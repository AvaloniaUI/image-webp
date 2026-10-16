//! One bitstream decoded a band of rows at a time as libwebp's `WebPDecode` decodes it into an
//! RGB or RGBA buffer, at its own size or scaled with libwebp's rescaler: the stills of a file
//! and the frames a demuxer hands out.

use std::collections::VecDeque;
use std::io::{BufRead, Cursor, Read, Seek, SeekFrom, Take};

use crate::decoder::DecodingError;
use crate::extended::{read_alpha_chunk, unfilter_alpha_row, AlphaChunk};
use crate::headers::Headers;
use crate::lossless::{LosslessBands, LosslessDecoder};
use crate::lossy::{BandDecoder, Channel};
use crate::rescaler::{self, Rescaler};
use crate::UpsamplingMethod;

const BUFFERED_BAND_ROWS: usize = 16;

/// Decodes the bitstream [`Headers`] found in some data, top to bottom, reading the data as
/// it goes: a lossy bitstream a window of each partition at a time.
pub struct FrameDecoder<'a> {
    width: u32,
    height: u32,
    bpp: usize,
    source: Source<'a>,
}

enum Source<'a> {
    Rgb(BandDecoder<'a, 3>),
    Rgba(BandDecoder<'a, 4>, Option<AlphaChunk>),
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
            let alpha_chunk = match &headers.alpha {
                Some(range) => Some(decode_alpha(&mut r, start + range.start as u64, range.len(), width, height)?),
                None => None,
            };
            let check = |(w, h): (u16, u16)| match (u32::from(w), u32::from(h)) == (width, height) {
                true => Ok(()),
                false => Err(DecodingError::InconsistentImageSizes),
            };
            match scaled {
                Some(size) => {
                    let mut bands = BandDecoder::new(Box::new(r), bitstream, end, UpsamplingMethod::Simple)?;
                    check(bands.dimensions())?;
                    if rescaler::bypasses_filter(width, height, size) {
                        bands.bypass_filter();
                    }
                    Source::Scaled(Box::new(ScaledLossy::new(bands, alpha_chunk.filter(|_| alpha), (width, height), size, bpp)))
                }
                None if alpha => {
                    let bands = BandDecoder::new(Box::new(r), bitstream, end, upsampling)?;
                    check(bands.dimensions())?;
                    Source::Rgba(bands, alpha_chunk)
                }
                None => {
                    let bands = BandDecoder::new(Box::new(r), bitstream, end, upsampling)?;
                    check(bands.dimensions())?;
                    // libwebp decodes the alpha plane of an RGB decode too, for its errors.
                    drop(alpha_chunk);
                    Source::Rgb(bands)
                }
            }
        };
        Ok(Self { width: out_width, height: out_height, bpp, source })
    }

    /// The output size.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The most rows a band has.
    pub fn max_band_rows(&self) -> u32 {
        let rows = match self.source {
            Source::Rgb(_) | Source::Rgba(..) => crate::lossy::MAX_BAND_ROWS,
            _ => BUFFERED_BAND_ROWS,
        };
        (rows as u32).min(self.height)
    }

    /// 4 for RGBA, 3 for RGB.
    pub fn bytes_per_pixel(&self) -> usize {
        self.bpp
    }

    /// The next band: its first row and its whole rows, or `None` after the last.
    pub fn next_band(&mut self) -> Result<Option<(u32, &[u8])>, DecodingError> {
        let w = self.width as usize;
        match &mut self.source {
            Source::Rgb(bands) => Ok(bands.next_band()?.map(|(y, rows)| (y, &*rows))),
            Source::Rgba(bands, alpha) => {
                let Some((y0, rows)) = bands.next_band()? else {
                    return Ok(None);
                };
                if let Some(alpha) = alpha {
                    for (y, row) in (y0 as usize..).zip(rows.chunks_exact_mut(w * 4)) {
                        let cur = unfiltered_row(alpha, y, w);
                        for (px, &a) in row.chunks_exact_mut(4).zip(cur) {
                            px[3] = a;
                        }
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
                band.clear();
                band.extend(rows.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]));
                Ok(Some((u32::from(y), band)))
            }
            Source::ScaledLossless(s) => s.next_band(),
        }
    }

}

/// Unfilters alpha row `y` in place, rows above it already unfiltered.
fn unfiltered_row(alpha: &mut AlphaChunk, y: usize, w: usize) -> &[u8] {
    let (above, rest) = alpha.data.split_at_mut(y * w);
    let prev = (y > 0).then(|| &above[(y - 1) * w..]);
    let cur = &mut rest[..w];
    unfilter_alpha_row(alpha.filtering_method, prev, cur);
    cur
}

/// The ALPH payload of `size` bytes at `at`, with libwebp's header checks.
fn decode_alpha<R: BufRead + Seek>(r: &mut R, at: u64, size: usize, width: u32, height: u32) -> Result<AlphaChunk, DecodingError> {
    r.seek(SeekFrom::Start(at))?;
    let mut payload = r.take(size as u64);
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
    read_alpha_chunk(&mut payload, width as u16, height as u16)
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
                _ => out.extend(dst_row.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]])),
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
    alpha: Option<AlphaChunk>,
    width: usize,
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
    fn new(bands: BandDecoder<'a, 3>, alpha: Option<AlphaChunk>, (w, h): (u32, u32), (ow, oh): (u32, u32), bpp: usize) -> Self {
        let (w, h, ow, oh) = (w as usize, h as usize, ow as usize, oh as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        Self {
            bands,
            alpha,
            width: w,
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
        match self.alpha {
            Some(_) => n.min(self.queues[3].len()),
            None => n,
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
                        let a = self.alpha.as_ref().map(|_| self.queues[3].pop_front().unwrap());
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
                        for y in self.alpha_rows..end {
                            let row = unfiltered_row(&mut alpha, y, self.width).to_vec();
                            self.push(3, &row);
                        }
                        self.alpha = Some(alpha);
                    }
                    self.alpha_rows = self.alpha_rows.max(end);
                }
            }
        }
    }
}
