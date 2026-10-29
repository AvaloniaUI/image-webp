//! One bitstream decoded a band of rows at a time as libwebp's `WebPDecode` decodes it into an
//! RGB or RGBA buffer, at its own size or scaled with libwebp's rescaler: the stills of a file
//! and the frames a demuxer hands out.

use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Take};
use std::sync::{Arc, Mutex, PoisonError};

use crate::ahead::{Ahead, Produce, Runner};
use crate::decoder::DecodingError;
use crate::extended::{read_alpha_chunk, AlphaRows};
use crate::headers::Headers;
use crate::lossless::{LosslessBands, LosslessDecoder};
use crate::lossy::{BandDecoder, Channel};
use crate::rescaler::{self, Rescaler};
use crate::{ReadSeek, UpsamplingMethod};

const BUFFERED_BAND_ROWS: usize = 16;
/// The most pixels a lossless RGB frame is decoded whole: its RGBA then takes no more than the
/// window of a band decode.
const LOSSLESS_WHOLE_RGB: usize = 2 << 20;

/// Decodes the bitstream [`Headers`] found in some data, top to bottom, reading the data as
/// it goes: a lossy bitstream a window of each partition at a time.
pub struct FrameDecoder<'a> {
    width: u32,
    height: u32,
    bpp: usize,
    source: Source<'a>,
    /// The size of the bitstream, which the output is scaled from.
    bitstream: (u32, u32),
}

enum Source<'a> {
    Rgb(BandDecoder<'a, 3>, Option<Alpha<'a>>),
    Rgba(BandDecoder<'a, 4>, Option<Alpha<'a>>),
    Scaled(Box<ScaledLossy<'a>>),
    /// RGBA bands handed out as they are, or as RGB through `band`.
    Lossless { bands: Lossless<'a>, band: Vec<u8> },
    ScaledLossless(Box<ScaledLossless<'a>>),
}

type Lossless<'a> = LosslessBands<Take<Box<dyn BufRead + Send + 'a>>>;

impl<'a> FrameDecoder<'a> {
    /// Starts decoding the bitstream `headers` found in `data`, which runs to the end of
    /// `data` as libwebp reads it, to RGBA (`alpha`) or RGB, scaled to `scaled` where given.
    pub fn new(
        data: &'a [u8],
        headers: &Headers,
        scaled: Option<(u32, u32)>,
        alpha: bool,
    ) -> Result<Self, DecodingError> {
        Self::with_upsampling(data, headers, scaled, alpha, UpsamplingMethod::Bilinear)
    }

    /// [`new`](Self::new) with lossy chroma upsampled by `upsampling` at full size; libwebp's
    /// default is [`UpsamplingMethod::Bilinear`].
    pub fn with_upsampling(
        data: &'a [u8],
        headers: &Headers,
        scaled: Option<(u32, u32)>,
        alpha: bool,
        upsampling: UpsamplingMethod,
    ) -> Result<Self, DecodingError> {
        Self::read(Cursor::new(data), 0, data.len() as u64, headers, scaled, alpha, upsampling)
    }

    /// [`new`](Self::new) over the `len` bytes `r` reads from `start` on, which
    /// [`headers::read_decoding`](crate::headers::read_decoding) found `headers` in.
    pub fn from_reader<R: BufRead + Seek + Send + 'a>(
        r: R,
        start: u64,
        len: u64,
        headers: &Headers,
        scaled: Option<(u32, u32)>,
        alpha: bool,
    ) -> Result<Self, DecodingError> {
        Self::read(r, start, len, headers, scaled, alpha, UpsamplingMethod::Bilinear)
    }

    pub(crate) fn read<R: BufRead + Seek + Send + 'a>(
        mut r: R,
        start: u64,
        len: u64,
        headers: &Headers,
        scaled: Option<(u32, u32)>,
        alpha: bool,
        upsampling: UpsamplingMethod,
    ) -> Result<Self, DecodingError> {
        let (width, height) = (headers.width, headers.height);
        let (out_width, out_height) = scaled.unwrap_or((width, height));
        if out_width == 0 || out_height == 0 {
            return Err(DecodingError::InconsistentImageSizes);
        }
        let bpp = if alpha { 4 } else { 3 };
        let bitstream = start + headers.image.start as u64;
        let end = start + len;
        let source = if headers.is_lossless {
            r.seek(SeekFrom::Start(bitstream))?;
            let r: Box<dyn BufRead + Send + 'a> = Box::new(r);
            let bands = LosslessDecoder::new(r.take(end - bitstream)).frame(width, height)?;
            match scaled {
                Some(size) => Source::ScaledLossless(Box::new(ScaledLossless::new(bands, (width, height), size, bpp))),
                None => Source::Lossless { bands, band: Vec::new() },
            }
        } else {
            // The alpha plane is read in step with the colour, from where its chunk is.
            let (r, alpha_rows): (Box<dyn ReadSeek + Send + 'a>, _) = match &headers.alpha {
                Some(range) => {
                    let shared = Shared::new(r);
                    let alpha = decode_alpha(shared.clone(), start + range.start as u64, range.len(), width, height)?;
                    (Box::new(shared), Some(Alpha::Rows(Box::new(alpha))))
                }
                None => (Box::new(r), None),
            };
            let check = |(w, h): (u16, u16)| match (u32::from(w), u32::from(h)) == (width, height) {
                true => Ok(()),
                false => Err(DecodingError::InconsistentImageSizes),
            };
            match scaled {
                Some(size) => {
                    let mut bands = BandDecoder::new(r, bitstream, end, UpsamplingMethod::Simple)?;
                    check(bands.dimensions())?;
                    if rescaler::bypasses_filter(width, height, size) {
                        bands.bypass_filter();
                    }
                    Source::Scaled(Box::new(ScaledLossy::new(bands, alpha_rows, alpha, (width, height), size, bpp)))
                }
                None if alpha => {
                    let bands = BandDecoder::new(r, bitstream, end, upsampling)?;
                    check(bands.dimensions())?;
                    Source::Rgba(bands, alpha_rows)
                }
                None => {
                    let bands = BandDecoder::new(r, bitstream, end, upsampling)?;
                    check(bands.dimensions())?;
                    // libwebp decodes the alpha plane of an RGB decode too, for its errors.
                    Source::Rgb(bands, alpha_rows)
                }
            }
        };
        Ok(Self { width: out_width, height: out_height, bpp, source, bitstream: (width, height) })
    }

    /// The output size.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The most rows a band has.
    pub fn max_band_rows(&self) -> u32 {
        let rows = match self.source {
            Source::Rgb(..) | Source::Rgba(..) => crate::lossy::MAX_BAND_ROWS,
            _ => BUFFERED_BAND_ROWS,
        };
        (rows as u32).min(self.height)
    }

    /// 4 for RGBA, 3 for RGB.
    pub fn bytes_per_pixel(&self) -> usize {
        self.bpp
    }

    /// Decodes the whole frame into `buf`, `width × height × bytes_per_pixel` bytes in the
    /// layout of the bands, straight into it instead of through a band: for a caller that
    /// holds the whole frame anyway. A lossless frame is decoded whole and its transforms
    /// undone in place, unless it is RGB and larger than the window a band decode holds;
    /// scaled frames and frames whose bands have started go through bands.
    pub fn read_into(&mut self, buf: &mut [u8]) -> Result<(), DecodingError> {
        let (w, h) = (self.width as usize, self.height as usize);
        if buf.len() != w * h * self.bpp {
            return Err(DecodingError::ImageTooLarge);
        }
        let direct = match &mut self.source {
            Source::Rgb(bands, alpha) => {
                while let Some((first, end)) = bands.next_band_into(buf)? {
                    // libwebp decodes the alpha plane of an RGB decode too, for its errors.
                    if let Some(alpha) = alpha {
                        for _ in first..end {
                            alpha.next_row()?;
                        }
                    }
                }
                true
            }
            Source::Rgba(bands, alpha) => {
                while let Some((first, end)) = bands.next_band_into(buf)? {
                    let rows = &mut buf[first * w * 4..end * w * 4];
                    match alpha {
                        Some(alpha) => {
                            for row in rows.chunks_exact_mut(w * 4) {
                                put_alpha(row, alpha.next_row()?);
                            }
                        }
                        None => rows.chunks_exact_mut(4).for_each(|px| px[3] = 255),
                    }
                }
                true
            }
            Source::Lossless { bands, .. } if bands.at_start() && self.bpp == 4 => {
                bands.read_into(buf)?;
                true
            }
            Source::Lossless { bands, .. } if bands.at_start() && w * h <= LOSSLESS_WHOLE_RGB => {
                let mut rgba = vec![0; w * h * 4];
                bands.read_into(&mut rgba)?;
                rgb_from_rgba(&rgba, buf);
                true
            }
            _ => false,
        };
        if !direct {
            let mut at = 0;
            while let Some((_, rows)) = self.next_band()? {
                buf[at..at + rows.len()].copy_from_slice(rows);
                at += rows.len();
            }
        }
        Ok(())
    }

    /// The next band: its first row and its whole rows, or `None` after the last.
    pub fn next_band(&mut self) -> Result<Option<(u32, &[u8])>, DecodingError> {
        let w = self.width as usize;
        match &mut self.source {
            Source::Rgb(bands, alpha) => {
                let Some((y0, rows)) = bands.next_band()? else {
                    return Ok(None);
                };
                if let Some(alpha) = alpha {
                    for _ in 0..rows.len() / (w * 3) {
                        alpha.next_row()?;
                    }
                }
                Ok(Some((y0, rows)))
            }
            Source::Rgba(bands, alpha) => {
                let Some((y0, rows)) = bands.next_band()? else {
                    return Ok(None);
                };
                if let Some(alpha) = alpha {
                    for row in rows.chunks_exact_mut(w * 4) {
                        put_alpha(row, alpha.next_row()?);
                    }
                }
                Ok(Some((y0, rows)))
            }
            Source::Scaled(s) => s.next_band(),
            Source::Lossless { bands, band } => {
                let Some((y, rows)) = bands.next_band()? else {
                    return Ok(None);
                };
                if self.bpp == 4 {
                    return Ok(Some((u32::from(y), rows)));
                }
                band.resize(rows.len() / 4 * 3, 0);
                rgb_from_rgba(rows, band);
                Ok(Some((u32::from(y), band)))
            }
            Source::ScaledLossless(s) => s.next_band(),
        }
    }
}

impl FrameDecoder<'_> {
    /// What the decode holds at most beyond decoding on its own once [`run_ahead`](Self::run_ahead)
    /// runs parts of it `ahead` units ahead; 0 where no part would run ahead.
    pub fn ahead_bytes(&self, ahead: usize) -> usize {
        let (bytes, alpha) = match &self.source {
            Source::Rgb(bands, alpha) => (bands.ahead_bytes(ahead), alpha.is_some()),
            Source::Rgba(bands, alpha) => (bands.ahead_bytes(ahead), alpha.is_some()),
            Source::Scaled(s) => (s.bands.ahead_bytes(ahead), s.alpha.is_some()),
            _ => return 0,
        };
        match bytes {
            0 => 0,
            _ if alpha => bytes + (self.alpha_ahead(ahead) + 2) * self.alpha_unit_bytes(),
            _ => bytes,
        }
    }

    /// The size of the alpha plane, that of the bitstream.
    fn alpha_width(&self) -> usize {
        self.bitstream.0 as usize
    }

    fn alpha_height(&self) -> usize {
        self.bitstream.1 as usize
    }

    /// The bytes of a unit of alpha rows decoded ahead: whole rows, about [`ALPHA_AHEAD_BYTES`].
    fn alpha_unit_bytes(&self) -> usize {
        let width = self.alpha_width();
        (ALPHA_AHEAD_BYTES / width).max(1) * width
    }

    /// The units of alpha rows decoded ahead: `ahead`, and at least [`ALPHA_AHEAD_ROWS`] rows.
    fn alpha_ahead(&self, ahead: usize) -> usize {
        let rows = self.alpha_unit_bytes() / self.alpha_width();
        ahead.max(ALPHA_AHEAD_ROWS.div_ceil(rows))
    }
}

impl FrameDecoder<'static> {
    /// Runs the parts of a lossy decode that go on their own, reading the macroblocks and
    /// decoding the alpha plane, on `runner`'s helpers while it has idle ones, at most `ahead`
    /// units ahead of the rows taken. The pixels and errors are those of the decode on its own.
    /// Before the first band.
    pub fn run_ahead(&mut self, runner: Runner, ahead: usize) {
        if self.ahead_bytes(ahead) == 0 {
            return;
        }
        let (per_unit, alpha_ahead) = (self.alpha_unit_bytes(), self.alpha_ahead(ahead));
        let (width, height) = (self.alpha_width(), self.alpha_height());
        let alpha = match &mut self.source {
            Source::Rgb(bands, alpha) => {
                bands.run_ahead(runner.clone(), ahead);
                alpha
            }
            Source::Rgba(bands, alpha) => {
                bands.run_ahead(runner.clone(), ahead);
                alpha
            }
            Source::Scaled(s) => {
                s.bands.run_ahead(runner.clone(), ahead);
                &mut s.alpha
            }
            _ => return,
        };
        if let Some(Alpha::Rows(rows)) = alpha.take() {
            let rows = *rows;
            let producer = AlphaProducer { rows, width, left: height, per_unit: per_unit / width, done: false };
            *alpha = Some(Alpha::Ahead(Box::new(AlphaAhead {
                ahead: Ahead::new(producer, runner, alpha_ahead),
                unit: AlphaUnit::default(),
                at: 0,
                width,
            })));
        }
    }
}

/// About how many bytes of alpha rows a unit decoded ahead holds.
const ALPHA_AHEAD_BYTES: usize = 16 << 10;
/// The fewest alpha rows decoded ahead: with fewer, the helper stops and is called again
/// several times in each band of rows the colour decode hands out.
const ALPHA_AHEAD_ROWS: usize = 8;

/// The alpha plane of a lossy bitstream, its rows decoded in step with the colour or ahead of it.
enum Alpha<'a> {
    Rows(Box<AlphaRows<'a>>),
    Ahead(Box<AlphaAhead>),
}

impl Alpha<'_> {
    fn next_row(&mut self) -> Result<&[u8], DecodingError> {
        match self {
            Alpha::Rows(rows) => rows.next_row(),
            Alpha::Ahead(ahead) => ahead.next_row(),
        }
    }
}

#[derive(Default)]
struct AlphaUnit {
    rows: Vec<u8>,
    error: Option<DecodingError>,
}

/// Decodes the rows of an alpha plane `per_unit` at a time.
struct AlphaProducer {
    rows: AlphaRows<'static>,
    width: usize,
    left: usize,
    per_unit: usize,
    done: bool,
}

impl Produce for AlphaProducer {
    type Unit = AlphaUnit;

    fn produce(&mut self, spare: Option<AlphaUnit>) -> Option<AlphaUnit> {
        if self.done || self.left == 0 {
            return None;
        }
        let mut unit = spare.unwrap_or_default();
        unit.rows.clear();
        unit.rows.reserve_exact(self.per_unit * self.width);
        unit.error = None;
        for _ in 0..self.per_unit.min(self.left) {
            match self.rows.next_row() {
                Ok(row) => unit.rows.extend_from_slice(row),
                Err(e) => {
                    unit.error = Some(e);
                    self.done = true;
                    break;
                }
            }
            self.left -= 1;
        }
        Some(unit)
    }
}

/// The rows an [`AlphaProducer`] decodes ahead, handed out one at a time.
struct AlphaAhead {
    ahead: Ahead<AlphaProducer>,
    unit: AlphaUnit,
    at: usize,
    width: usize,
}

impl AlphaAhead {
    fn next_row(&mut self) -> Result<&[u8], DecodingError> {
        while self.at * self.width == self.unit.rows.len() {
            if let Some(e) = self.unit.error.take() {
                return Err(e);
            }
            let spent = std::mem::take(&mut self.unit);
            self.unit = self.ahead.next(Some(spent)).ok_or(DecodingError::BitStreamError)?;
            self.at = 0;
        }
        let row = &self.unit.rows[self.at * self.width..][..self.width];
        self.at += 1;
        Ok(row)
    }
}

/// Sets the alpha of the RGBA pixels `row` to `alpha`.
fn put_alpha(row: &mut [u8], alpha: &[u8]) {
    for (px, &a) in row.as_chunks_mut::<4>().0.iter_mut().zip(alpha) {
        px[3] = a;
    }
}

/// Drops the alpha of the RGBA pixels `rgba` into `rgb`, four pixels a step as three words.
fn rgb_from_rgba(rgba: &[u8], rgb: &mut [u8]) {
    let (quads, rest) = rgba.as_chunks::<16>();
    let (triples, tail) = rgb.as_chunks_mut::<12>();
    for (q, t) in quads.iter().zip(triples) {
        let w = |i: usize| u32::from_le_bytes(q[4 * i..4 * i + 4].try_into().unwrap());
        let (a, b, c, d) = (w(0), w(1), w(2), w(3));
        t[..4].copy_from_slice(&(a & 0xff_ffff | b << 24).to_le_bytes());
        t[4..8].copy_from_slice(&(b >> 8 & 0xffff | c << 16).to_le_bytes());
        t[8..].copy_from_slice(&(c >> 16 & 0xff | d << 8).to_le_bytes());
    }
    for (px, p) in tail.chunks_exact_mut(3).zip(rest.chunks_exact(4)) {
        px.copy_from_slice(&p[..3]);
    }
}

/// The ALPH payload of `size` bytes at `at`, with libwebp's header checks.
fn decode_alpha<'a, R: Read + Seek + Send + 'a>(mut r: R, at: u64, size: usize, width: u32, height: u32) -> Result<AlphaRows<'a>, DecodingError> {
    r.seek(SeekFrom::Start(at))?;
    let mut payload: Box<dyn BufRead + Send + 'a> = Box::new(BufReader::new(r.take(size as u64)));
    let Some(&info) = payload.fill_buf()?.first().filter(|_| size > 1) else {
        return Err(DecodingError::InvalidCompressionMethod);
    };
    let (method, pre, reserved) = (info & 3, (info >> 4) & 3, info >> 6);
    if method > 1 || pre > 1 || reserved != 0 {
        return Err(DecodingError::InvalidCompressionMethod);
    }
    if method == 0 && size - 1 < width as usize * height as usize {
        return Err(DecodingError::InvalidCompressionMethod);
    }
    read_alpha_chunk(payload, width as u16, height as u16)
}

/// A reader read at two places in turn, each handle at a position of its own.
struct Shared<R> {
    inner: Arc<Mutex<R>>,
    pos: u64,
}

impl<R> Shared<R> {
    fn new(inner: R) -> Self {
        Self { inner: Arc::new(Mutex::new(inner)), pos: 0 }
    }
}

impl<R> Clone for Shared<R> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner), pos: self.pos }
    }
}

impl<R: Read + Seek> Read for Shared<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut r = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        r.seek(SeekFrom::Start(self.pos))?;
        let n = r.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl<R: Seek> Seek for Shared<R> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.pos = match pos {
            SeekFrom::Start(p) => p,
            SeekFrom::Current(d) => self.pos.checked_add_signed(d).ok_or(io::ErrorKind::InvalidInput)?,
            SeekFrom::End(_) => self.inner.lock().unwrap_or_else(PoisonError::into_inner).seek(pos)?,
        };
        Ok(self.pos)
    }
}

/// libwebp's alpha multiply: `round(x * a / 255)` forward, `x * 255 / a` back, in 24-bit
/// fixed point.
fn mult(x: u8, scale: u32) -> u8 {
    ((u32::from(x) * scale + (1 << 23)) >> 24) as u8
}

fn premultiply(row: &mut [u8], inverse: bool) {
    for px in row.chunks_exact_mut(4) {
        match px[3] {
            255 => {}
            0 => px.fill(0),
            a => {
                let scale = match inverse {
                    false => u32::from(a) * ((1 << 24) / 255),
                    true => (255 << 24) / u32::from(a),
                };
                for c in &mut px[..3] {
                    *c = mult(*c, scale);
                }
            }
        }
    }
}

/// A lossless picture scaled as libwebp's VP8L decoder scales it: each row premultiplied,
/// scaled in four channels, unpremultiplied, a band of decoded rows at a time.
struct ScaledLossless<'a> {
    bands: Lossless<'a>,
    rescaler: Rescaler,
    src_row: Vec<u8>,
    dst_row: Vec<u8>,
    out: Vec<u8>,
    handed: usize,
    bpp: usize,
    next_row: u32,
    done: bool,
}

impl<'a> ScaledLossless<'a> {
    fn new(bands: Lossless<'a>, (w, h): (u32, u32), (ow, oh): (u32, u32), bpp: usize) -> Self {
        Self {
            bands,
            rescaler: Rescaler::new(w as usize, h as usize, ow as usize, oh as usize, 4),
            src_row: vec![0; w as usize * 4],
            dst_row: vec![0; ow as usize * 4],
            out: Vec::new(),
            handed: 0,
            bpp,
            next_row: 0,
            done: false,
        }
    }

    fn export(rescaler: &mut Rescaler, dst_row: &mut [u8], out: &mut Vec<u8>, bpp: usize) {
        while rescaler.has_pending_output() {
            rescaler.export(dst_row);
            premultiply(dst_row, true);
            match bpp {
                4 => out.extend_from_slice(dst_row),
                _ => {
                    let at = out.len();
                    out.resize(at + dst_row.len() / 4 * 3, 0);
                    rgb_from_rgba(dst_row, &mut out[at..]);
                }
            }
        }
    }

    fn next_band(&mut self) -> Result<Option<(u32, &[u8])>, DecodingError> {
        let stride = self.rescaler.dst_row_len() / 4 * self.bpp;
        self.out.drain(..self.handed * stride);
        while self.out.len() < BUFFERED_BAND_ROWS * stride && !self.done {
            let (rescaler, dst_row, out) = (&mut self.rescaler, &mut self.dst_row, &mut self.out);
            Self::export(rescaler, dst_row, out, self.bpp);
            match self.bands.next_band()? {
                None => self.done = true,
                Some((_, rows)) => {
                    for row in rows.chunks_exact(self.src_row.len()) {
                        Self::export(rescaler, dst_row, out, self.bpp);
                        self.src_row.copy_from_slice(row);
                        premultiply(&mut self.src_row, false);
                        rescaler.import(&self.src_row);
                    }
                }
            }
        }
        Self::export(&mut self.rescaler, &mut self.dst_row, &mut self.out, self.bpp);
        self.handed = (self.out.len() / stride).min(BUFFERED_BAND_ROWS);
        if self.handed == 0 {
            return Ok(None);
        }
        let first = self.next_row;
        self.next_row += self.handed as u32;
        Ok(Some((first, &self.out[..self.handed * stride])))
    }
}

/// A lossy bitstream scaled as libwebp's RGB rescaling does it: luma, both chroma planes and
/// alpha each scaled to the output size, then converted per pixel.
struct ScaledLossy<'a> {
    bands: BandDecoder<'a, 3>,
    alpha: Option<Alpha<'a>>,
    /// Whether the alpha plane is scaled and handed out, not only decoded.
    keep_alpha: bool,
    alpha_rows: usize,
    scalers: [Rescaler; 4],
    queues: [VecDeque<Vec<u8>>; 4],
    spare: Vec<Vec<u8>>,
    out: Vec<u8>,
    bpp: usize,
    next_row: u32,
    done: bool,
}

impl<'a> ScaledLossy<'a> {
    fn new(bands: BandDecoder<'a, 3>, alpha: Option<Alpha<'a>>, keep_alpha: bool, (w, h): (u32, u32), (ow, oh): (u32, u32), bpp: usize) -> Self {
        let (w, h, ow, oh) = (w as usize, h as usize, ow as usize, oh as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        Self {
            bands,
            keep_alpha: keep_alpha && alpha.is_some(),
            alpha,
            alpha_rows: 0,
            scalers: [
                Rescaler::new(w, h, ow, oh, 1),
                Rescaler::new(cw, ch, ow, oh, 1),
                Rescaler::new(cw, ch, ow, oh, 1),
                Rescaler::new(w, h, ow, oh, 1),
            ],
            queues: Default::default(),
            spare: Vec::new(),
            out: Vec::new(),
            bpp,
            next_row: 0,
            done: false,
        }
    }

    fn push(&mut self, plane: usize, row: &[u8]) {
        self.drain(plane);
        self.scalers[plane].import(row);
        self.drain(plane);
    }

    fn drain(&mut self, plane: usize) {
        let r = &mut self.scalers[plane];
        while r.has_pending_output() {
            let mut out = self.spare.pop().unwrap_or_default();
            out.resize(r.dst_row_len(), 0);
            r.export(&mut out);
            self.queues[plane].push_back(out);
        }
    }

    fn ready(&self) -> usize {
        let n = self.queues[..3].iter().map(VecDeque::len).min().unwrap_or(0);
        match self.keep_alpha {
            true => n.min(self.queues[3].len()),
            false => n,
        }
    }

    fn next_band(&mut self) -> Result<Option<(u32, &[u8])>, DecodingError> {
        loop {
            let n = self.ready().min(BUFFERED_BAND_ROWS);
            if n > 0 {
                let ow = self.scalers[0].dst_row_len();
                let stride = ow * self.bpp;
                self.out.resize(n * stride, 0);
                for out in self.out.chunks_exact_mut(stride) {
                    let [y, u, v] = [0, 1, 2].map(|p| self.queues[p].pop_front().unwrap());
                    match self.bpp {
                        4 => crate::lossy::fill_row_yuv::<4>(out, &y, &u, &v),
                        _ => crate::lossy::fill_row_yuv::<3>(out, &y, &u, &v),
                    }
                    if self.bpp == 4 {
                        let a = self.keep_alpha.then(|| self.queues[3].pop_front().unwrap());
                        for (i, px) in out.chunks_exact_mut(4).enumerate() {
                            px[3] = a.as_ref().map_or(255, |a| a[i]);
                        }
                        self.spare.extend(a);
                    }
                    self.spare.extend([y, u, v]);
                }
                let first = self.next_row;
                self.next_row += n as u32;
                return Ok(Some((first, &self.out)));
            }
            if self.done {
                return Ok(None);
            }
            let mut rows: [Vec<Vec<u8>>; 3] = Default::default();
            let end = self.bands.next_planes(|plane, row| {
                let p = match plane {
                    Channel::Y => 0,
                    Channel::U => 1,
                    Channel::V => 2,
                };
                rows[p].push(row.to_vec());
            })?;
            for (p, rows) in rows.iter().enumerate() {
                for row in rows {
                    self.push(p, row);
                }
            }
            match end {
                None => self.done = true,
                Some(end) => {
                    if let Some(mut alpha) = self.alpha.take() {
                        let mut decoded = Ok(());
                        for _ in self.alpha_rows..end {
                            match alpha.next_row() {
                                Ok(row) if self.keep_alpha => self.push(3, row),
                                Ok(_) => {}
                                Err(e) => {
                                    decoded = Err(e);
                                    break;
                                }
                            }
                        }
                        self.alpha = Some(alpha);
                        decoded?;
                    }
                    self.alpha_rows = self.alpha_rows.max(end);
                }
            }
        }
    }
}
