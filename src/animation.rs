//! Encoding of animated WebP images.
//!
//! Frames come in one at a time on a canvas of their own size. Each is coded as the rectangle
//! that changed since the previous one, over the previous canvas or over that canvas with the
//! previous rectangle cleared, blended where the unchanged pixels can be made transparent, and
//! every `kmax` frames as a key frame. A frame identical to the previous one lengthens it
//! instead. Frames are written as soon as the next frame has settled how they dispose, so only
//! the two canvases, the rectangle being coded and the undecided coded frames are held.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.

use std::collections::VecDeque;
use std::io::{self, Seek, SeekFrom, Write};

use crate::encoder::{alpha_chunk_payload, chunk_size, write_chunk, write_chunk_parts};
use crate::lossless::{vp8l, EncoderParams};
use crate::lossy::{fill_row_yuv, upsample_chroma_row, vp8enc};
use crate::{ColorType, EncodingError, WebPEncoder};

const MAX_DURATION: u32 = 1 << 24;
const ANIMATION_FLAG: u8 = 1 << 1;
const ALPHA_FLAG: u8 = 1 << 4;

/// How an animation is cut into frames. The frames themselves are coded as
/// [`EncoderParams`] say.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AnimationParams {
    /// The smallest number of frames between two key frames, as libwebp's `kmin`. Past it a
    /// frame becomes a key frame where that costs less than coding it as a change.
    pub kmin: u32,
    /// The largest number of frames between two key frames, as libwebp's `kmax`. Defaults to 10
    /// with `kmin` 9: every tenth frame is a key frame.
    pub kmax: u32,
    /// How many times the animation plays, 0 for ever.
    pub loop_count: u16,
    /// The background colour hint of the `ANIM` chunk as `[blue, green, red, alpha]`.
    pub background: [u8; 4],
    /// Lossy frames pass through YUV 4:2:0 and back before anything looks at them, as libwebp's
    /// animation encoder gets them from libvips: the chroma is smoothed twice, which makes the
    /// files 5-10% smaller on detailed content for about 0.3 dB. On by default.
    pub subsample_first: bool,
}

impl Default for AnimationParams {
    fn default() -> Self {
        Self {
            kmin: 9,
            kmax: 10,
            loop_count: 0,
            background: [0xff; 4],
            subsample_first: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Rect {
    fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    fn area(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    fn snap_to_even(&mut self) {
        self.width += self.x & 1;
        self.height += self.y & 1;
        self.x &= !1;
        self.y &= !1;
    }

    fn columns(self) -> std::ops::Range<usize> {
        self.x as usize..(self.x + self.width) as usize
    }

    fn rows(self) -> std::ops::Range<usize> {
        self.y as usize..(self.y + self.height) as usize
    }
}

/// A coded frame: a `VP8L` payload, or a `VP8 ` payload with its `ALPH` chunk when some pixel
/// is not opaque.
struct FrameData {
    lossless: bool,
    parts: Vec<Vec<u8>>,
    alpha: Option<Vec<u8>>,
    has_alpha: bool,
}

impl FrameData {
    /// The bytes of the frame's chunks, which is what the candidates are compared by.
    fn chunks_len(&self) -> usize {
        let frame: usize = self.parts.iter().map(Vec::len).sum();
        let alpha = self
            .alpha
            .as_ref()
            .map_or(0, |a| chunk_size(a.len()) as usize);
        chunk_size(frame) as usize + alpha
    }

    fn write<W: Write>(&self, mut w: W) -> io::Result<()> {
        if let Some(alpha) = &self.alpha {
            write_chunk(&mut w, b"ALPH", alpha)?;
        }
        let parts: Vec<&[u8]> = self.parts.iter().map(Vec::as_slice).collect();
        let fourcc = if self.lossless { b"VP8L" } else { b"VP8 " };
        write_chunk_parts(&mut w, fourcc, &parts)
    }
}

/// Where a frame is coded and how it goes over the canvas.
struct Plan {
    rect: Rect,
    disposed: Option<Rect>,
    blend: bool,
}

/// The frame as it would be stored: coded at `rect`, blended over the canvas or not, with the
/// canvas disposed to transparent after it or not.
struct Coded {
    rect: Rect,
    blend: bool,
    dispose_to_background: bool,
    data: FrameData,
}

/// A frame whose place in the file is not settled: its variant as a change (`sub`) and as a
/// key frame, one of which is written once the following frames have decided.
struct Pending {
    sub: Option<Coded>,
    key: Option<Coded>,
    is_key: bool,
    duration: u32,
}

impl Pending {
    fn chosen(&mut self) -> &mut Coded {
        match self.is_key {
            true => self.key.as_mut(),
            false => self.sub.as_mut(),
        }
        .expect("a pending frame has its chosen variant")
    }
}

/// Animated WebP encoder.
///
/// Frames are added with [`add_frame`](Self::add_frame) and the file is completed with
/// [`finish`](Self::finish). The writer is seeked back to at the end to fill in the sizes.
pub struct AnimationEncoder<W> {
    writer: W,
    params: EncoderParams,
    animation: AnimationParams,
    icc_profile: Vec<u8>,
    exif_metadata: Vec<u8>,
    xmp_metadata: Vec<u8>,
    width: u32,
    height: u32,
    /// The previous frame as given, ARGB, transparent before the first.
    prev: Vec<u32>,
    /// The frame being added, ARGB.
    curr: Vec<u32>,
    prev_rect: Rect,
    pending: VecDeque<Pending>,
    /// How many of `pending` are settled, from the front.
    flush_count: usize,
    /// The frame in `pending` that is the key frame so far since `kmin`, undecided until `kmax`.
    keyframe: Option<usize>,
    best_delta: i64,
    since_key_frame: u32,
    prev_candidate_undecided: bool,
    frames_in: u32,
    frames_out: u32,
    started: bool,
    has_alpha: bool,
}

impl<W: Write + Seek> AnimationEncoder<W> {
    /// Create a new encoder that writes an animation on a `width` x `height` canvas to `w`.
    pub fn new(w: W, width: u32, height: u32) -> Result<Self, EncodingError> {
        if width == 0 || width > 16384 || height == 0 || height > 16384 {
            return Err(EncodingError::InvalidDimensions);
        }
        let pixels = width as usize * height as usize;
        Ok(Self {
            writer: w,
            params: EncoderParams::default(),
            animation: AnimationParams::default(),
            icc_profile: Vec::new(),
            exif_metadata: Vec::new(),
            xmp_metadata: Vec::new(),
            width,
            height,
            prev: vec![0; pixels],
            curr: vec![0; pixels],
            prev_rect: Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            pending: VecDeque::new(),
            flush_count: 0,
            keyframe: None,
            best_delta: i64::MAX,
            since_key_frame: 0,
            prev_candidate_undecided: false,
            frames_in: 0,
            frames_out: 0,
            started: false,
            has_alpha: false,
        })
    }

    /// Set the ICC profile to use for the image.
    pub fn set_icc_profile(&mut self, icc_profile: Vec<u8>) {
        self.icc_profile = icc_profile;
    }

    /// Set the EXIF metadata to use for the image.
    pub fn set_exif_metadata(&mut self, exif_metadata: Vec<u8>) {
        self.exif_metadata = exif_metadata;
    }

    /// Set the XMP metadata to use for the image.
    pub fn set_xmp_metadata(&mut self, xmp_metadata: Vec<u8>) {
        self.xmp_metadata = xmp_metadata;
    }

    /// Set the `EncoderParams` the frames are coded with.
    pub fn set_params(&mut self, params: EncoderParams) {
        self.params = params;
    }

    /// Set the `AnimationParams` to use. `kmin` and `kmax` are brought into libwebp's bounds:
    /// `kmax` 0 disables key frames, `kmin` stays below `kmax` and at least `kmax / 2 + 1`.
    pub fn set_animation_params(&mut self, params: AnimationParams) {
        let mut p = params;
        if p.kmax == 1 {
            (p.kmin, p.kmax) = (0, 0);
        } else {
            if p.kmax == 0 {
                p.kmax = u32::MAX;
            }
            if p.kmin >= p.kmax {
                p.kmin = p.kmax - 1;
            } else {
                let limit = p.kmax / 2 + 1;
                if p.kmin < limit && limit < p.kmax {
                    p.kmin = limit;
                }
            }
            if p.kmax - p.kmin > 30 {
                p.kmin = p.kmax - 30;
            }
        }
        self.animation = p;
    }

    /// Adds the next frame, `width` x `height` pixels of `color`, shown for `duration` ms.
    ///
    /// # Panics
    ///
    /// Panics if the data is not of the canvas's dimensions.
    pub fn add_frame(
        &mut self,
        data: &[u8],
        color: ColorType,
        duration: u32,
    ) -> Result<(), EncodingError> {
        import(data, color, &mut self.curr);
        if self.params.use_lossy && self.animation.subsample_first {
            chroma_roundtrip(&mut self.curr, self.width, self.height);
        }
        let duration = duration.min(MAX_DURATION - 1);
        let kept = self.cache_frame(duration)?;
        self.flush_frames()?;
        if kept {
            std::mem::swap(&mut self.prev, &mut self.curr);
        }
        self.frames_in += 1;
        Ok(())
    }

    /// Writes the last frames and the metadata and fills in the sizes. An animation of one
    /// frame, or whose frames never change, is written as a still.
    pub fn finish(mut self) -> Result<W, EncodingError> {
        if self.frames_in == 0 {
            return Err(EncodingError::InvalidDimensions);
        }
        self.flush_count = self.pending.len();
        if self.frames_out + self.pending.len() as u32 == 1 {
            self.write_still()?;
            return Ok(self.writer);
        }
        self.flush_frames()?;
        if !self.exif_metadata.is_empty() {
            write_chunk(&mut self.writer, b"EXIF", &self.exif_metadata)?;
        }
        if !self.xmp_metadata.is_empty() {
            write_chunk(&mut self.writer, b"XMP ", &self.xmp_metadata)?;
        }
        let end = self.writer.stream_position()?;
        self.writer.seek(SeekFrom::Start(4))?;
        self.writer.write_all(&((end - 8) as u32).to_le_bytes())?;
        self.writer.seek(SeekFrom::Start(20))?;
        self.writer.write_all(&[self.vp8x_flags()])?;
        self.writer.seek(SeekFrom::Start(end))?;
        Ok(self.writer)
    }

    /// libwebp's `CacheFrame`: codes the frame as a change and, past `kmin` frames since the
    /// last key frame, as a key frame too, and settles which earlier frames can be written.
    /// Returns whether the frame was kept rather than merged into the previous one.
    fn cache_frame(&mut self, duration: u32) -> Result<bool, EncodingError> {
        if self.frames_in == 0 {
            let plan = self
                .plan_frame(true)
                .expect("the first frame is never empty");
            let key = self.code_frame(plan, true)?;
            self.pending.push_back(Pending {
                sub: None,
                key: Some(key),
                is_key: true,
                duration,
            });
            self.flush_count = 0;
            self.since_key_frame = 0;
            self.prev_candidate_undecided = false;
            return Ok(true);
        }
        self.since_key_frame += 1;
        let Some(plan) = self.plan_frame(false) else {
            self.since_key_frame -= 1;
            let last = self
                .pending
                .back_mut()
                .expect("a frame precedes a merged one");
            last.duration = (last.duration + duration).min(MAX_DURATION - 1);
            return Ok(false);
        };
        if self.since_key_frame <= self.animation.kmin {
            let sub = self.code_frame(plan, false)?;
            self.pending.push_back(Pending {
                sub: Some(sub),
                key: None,
                is_key: false,
                duration,
            });
            self.flush_count = self.pending.len() - 1;
            self.prev_candidate_undecided = false;
            return Ok(true);
        }
        let forced = self.since_key_frame >= self.animation.kmax;
        // A forced key frame wins whatever the change would cost, so that is not coded.
        let sub = match forced {
            true => None,
            false => Some(self.code_frame(plan, false)?),
        };
        let prev_rect_sub = self.prev_rect;
        let key_plan = self.plan_frame(true).expect("a key frame is never empty");
        let key = self.code_frame(key_plan, true)?;
        let prev_rect_key = self.prev_rect;
        let delta = sub.as_ref().map_or(i64::MIN, |s| {
            key.data.chunks_len() as i64 - s.data.chunks_len() as i64
        });
        let is_key = delta <= self.best_delta;
        self.pending.push_back(Pending {
            sub,
            key: Some(key),
            is_key,
            duration,
        });
        let position = self.pending.len() - 1;
        if is_key {
            if let Some(old) = self.keyframe {
                self.pending[old].is_key = false;
            }
            self.prev_candidate_undecided = true;
            self.keyframe = Some(position);
            self.best_delta = delta;
            self.flush_count = position;
        } else {
            self.prev_candidate_undecided = false;
        }
        if forced {
            self.flush_count = position;
            self.since_key_frame = 0;
            self.keyframe = None;
            self.best_delta = i64::MAX;
        }
        if !self.prev_candidate_undecided {
            self.prev_rect = if is_key { prev_rect_key } else { prev_rect_sub };
        }
        Ok(true)
    }

    /// libwebp's `SetFrame` up to the coding: where the frame changes the previous canvas, with
    /// the previous frame's rectangle disposed to transparent where that changes less, and
    /// whether it can blend. `None` for a frame that does not change the canvas.
    fn plan_frame(&self, is_key: bool) -> Option<Plan> {
        let first = self.frames_in == 0;
        let max_diff = self.max_diff();
        let mut rect = Rect {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        };
        if !is_key || first {
            rect = self.minimize_change_rectangle(None, max_diff);
        }
        if rect.is_empty() {
            if !first {
                return None;
            }
            rect = Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            };
        }
        rect.snap_to_even();
        let mut disposed = None;
        if !is_key && !self.prev_candidate_undecided {
            let mut bg = self.minimize_change_rectangle(Some(self.prev_rect), max_diff);
            if !bg.is_empty() {
                bg.snap_to_even();
                if bg.area() < rect.area() {
                    rect = bg;
                    disposed = Some(self.prev_rect);
                }
            }
        }
        let blend = !is_key && self.blending_possible(rect, disposed, max_diff);
        Some(Plan {
            rect,
            disposed,
            blend,
        })
    }

    fn max_diff(&self) -> i32 {
        match self.params.use_lossy {
            false => 0,
            true => quality_to_max_diff(f32::from(self.params.lossy_quality.min(100))),
        }
    }

    /// Codes the frame as planned. A change settles how the previous frame disposes.
    fn code_frame(&mut self, plan: Plan, is_key: bool) -> Result<Coded, EncodingError> {
        let Plan {
            rect,
            disposed,
            blend,
        } = plan;
        let data = match self.params.use_lossy {
            false => {
                let mut sub = self.cut(rect);
                if blend {
                    self.increase_transparency(rect, disposed, &mut sub);
                }
                encode_lossless(&self.params, sub, rect.width, rect.height)
            }
            true => {
                let flat = match blend {
                    true => self.flatten_similar_blocks(rect, disposed, self.max_diff()),
                    false => Flattened::default(),
                };
                let stride = self.width as usize;
                encode_lossy(&self.params, rect.width, rect.height, blend, |y, row| {
                    let y = rect.y as usize + y;
                    argb_to_rgba(&self.curr[y * stride..][rect.columns()], row);
                    flat.apply(y, rect.x as usize, row);
                })?
            }
        };
        if !is_key {
            if let Some(prev) = self.pending.back_mut() {
                prev.chosen().dispose_to_background = disposed.is_some();
            }
        }
        self.prev_rect = rect;
        Ok(Coded {
            rect,
            blend,
            dispose_to_background: false,
            data,
        })
    }

    /// The row `y` of the previous canvas, and the columns of it the previous frame's
    /// rectangle `disposed` to transparent.
    fn prev_row(&self, y: usize, disposed: Option<Rect>) -> (&[u32], std::ops::Range<usize>) {
        let stride = self.width as usize;
        let row = &self.prev[y * stride..][..stride];
        let cleared = match disposed {
            Some(r) if r.rows().contains(&y) => r.columns(),
            _ => 0..0,
        };
        (row, cleared)
    }

    /// Row `y` of the previous canvas with `disposed` cleared, and of the frame being added.
    fn rows(&self, y: usize, disposed: Option<Rect>) -> RowPair<'_> {
        let stride = self.width as usize;
        let (prev, cleared) = self.prev_row(y, disposed);
        RowPair {
            prev,
            cleared,
            curr: &self.curr[y * stride..][..stride],
        }
    }

    /// libwebp's `MinimizeChangeRectangle` over the whole canvas: the bounds of the pixels that
    /// changed from the previous canvas, empty when nothing changed. Trimming unchanged
    /// columns and then rows leaves exactly these bounds, so they are found row by row.
    fn minimize_change_rectangle(&self, disposed: Option<Rect>, max_diff: i32) -> Rect {
        match max_diff {
            0 => self.changed_bounds(disposed, |a, b| a != b),
            _ => self.changed_bounds(disposed, |a, b| !similar(a, b, max_diff)),
        }
    }

    fn changed_bounds(
        &self,
        disposed: Option<Rect>,
        differs: impl Fn(u32, u32) -> bool + Copy,
    ) -> Rect {
        let (w, h) = (self.width as usize, self.height as usize);
        let Some((top, mut left)) =
            (0..h).find_map(|y| Some((y, self.rows(y, disposed).first(0..w, differs)?)))
        else {
            return Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            };
        };
        let mut right = self
            .rows(top, disposed)
            .last(left..w, differs)
            .expect("the row has a change");
        let mut bottom = top;
        for y in (top + 1..h).rev() {
            let row = self.rows(y, disposed);
            if let Some(x) = row.last(0..w, differs) {
                bottom = y;
                right = right.max(x);
                left = row.first(0..left, differs).unwrap_or(left);
                break;
            }
        }
        for y in top + 1..bottom {
            let row = self.rows(y, disposed);
            left = row.first(0..left, differs).unwrap_or(left);
            right = row.last(right + 1..w, differs).unwrap_or(right);
        }
        Rect {
            x: left as u32,
            y: top as u32,
            width: (right - left + 1) as u32,
            height: (bottom - top + 1) as u32,
        }
    }

    /// Whether blending the frame over the previous canvas can give every pixel of `rect`: a
    /// pixel that is not opaque has to be there already.
    fn blending_possible(&self, rect: Rect, disposed: Option<Rect>, max_diff: i32) -> bool {
        match max_diff {
            0 => self.blending_possible_with(rect, disposed, |a, b| a != b),
            _ => self.blending_possible_with(rect, disposed, |a, b| !similar(a, b, max_diff)),
        }
    }

    fn blending_possible_with(
        &self,
        rect: Rect,
        disposed: Option<Rect>,
        differs: impl Fn(u32, u32) -> bool + Copy,
    ) -> bool {
        rect.rows().all(|y| {
            self.rows(y, disposed)
                .first(rect.columns(), |p, c| (c >> 24 != 0xff) & differs(p, c))
                .is_none()
        })
    }

    fn cut(&self, rect: Rect) -> Vec<u32> {
        let stride = self.width as usize;
        let mut sub = Vec::with_capacity(rect.area() as usize);
        for y in rect.rows() {
            sub.extend_from_slice(&self.curr[y * stride..][rect.columns()]);
        }
        sub
    }

    /// Pixels equal to the previous canvas become transparent, as blending restores them.
    fn increase_transparency(&self, rect: Rect, disposed: Option<Rect>, sub: &mut [u32]) {
        for (dy, y) in rect.rows().enumerate() {
            let (row, cleared) = self.prev_row(y, disposed);
            let out = &mut sub[dy * rect.width as usize..][..rect.width as usize];
            for (dx, x) in rect.columns().enumerate() {
                let prev = if cleared.contains(&x) { 0 } else { row[x] };
                if out[dx] == prev && out[dx] != 0 {
                    out[dx] = 0;
                }
            }
        }
    }

    /// 8x8 blocks of opaque pixels similar to the previous canvas become transparent blocks of
    /// their average colour, which cost little and blend to what the canvas shows.
    fn flatten_similar_blocks(
        &self,
        rect: Rect,
        disposed: Option<Rect>,
        max_diff: i32,
    ) -> Flattened {
        let stride = self.width as usize;
        let y_start = (rect.y as usize + 8) & !7;
        let y_end = (rect.y + rect.height) as usize & !7;
        let x_start = (rect.x as usize + 8) & !7;
        let x_end = (rect.x + rect.width) as usize & !7;
        let mut flat = Flattened {
            x: x_start,
            y: y_start,
            columns: x_end.saturating_sub(x_start) / 8,
            colors: Vec::new(),
        };
        let flattens = |p: u32, c: u32| (c >> 24 == 0xff) & similar(p, c, max_diff);
        for by in (y_start..y_end).step_by(8) {
            for bx in (x_start..x_end).step_by(8) {
                let all = (by..by + 8).all(|y| {
                    self.rows(y, disposed)
                        .first(bx..bx + 8, |p, c| !flattens(p, c))
                        .is_none()
                });
                if !all {
                    flat.colors.push(KEEP);
                    continue;
                }
                let mut sums = [0u32; 3];
                for y in by..by + 8 {
                    for &src in &self.curr[y * stride + bx..][..8] {
                        sums[0] += (src >> 16) & 0xff;
                        sums[1] += (src >> 8) & 0xff;
                        sums[2] += src & 0xff;
                    }
                }
                flat.colors
                    .push(((sums[0] / 64) << 16) | ((sums[1] / 64) << 8) | (sums[2] / 64));
            }
        }
        flat
    }

    /// Writes the settled frames, the file's header first.
    fn flush_frames(&mut self) -> Result<(), EncodingError> {
        while self.flush_count > 0 {
            let mut frame = self
                .pending
                .pop_front()
                .expect("flush_count counts pending frames");
            self.flush_count -= 1;
            if let Some(k) = self.keyframe.as_mut() {
                *k -= 1;
            }
            if !self.started {
                self.start()?;
            }
            let duration = frame.duration;
            let coded = frame.chosen();
            let data_len = coded.data.chunks_len();
            let r = coded.rect;
            self.writer.write_all(b"ANMF")?;
            self.writer
                .write_all(&(16 + data_len as u32).to_le_bytes())?;
            self.writer.write_all(&(r.x / 2).to_le_bytes()[..3])?;
            self.writer.write_all(&(r.y / 2).to_le_bytes()[..3])?;
            self.writer.write_all(&(r.width - 1).to_le_bytes()[..3])?;
            self.writer.write_all(&(r.height - 1).to_le_bytes()[..3])?;
            self.writer.write_all(&duration.to_le_bytes()[..3])?;
            let flags = u8::from(coded.dispose_to_background) | u8::from(!coded.blend) << 1;
            self.writer.write_all(&[flags])?;
            coded.data.write(&mut self.writer)?;
            self.has_alpha |= coded.data.has_alpha;
            self.frames_out += 1;
        }
        Ok(())
    }

    fn vp8x_flags(&self) -> u8 {
        let mut flags = ANIMATION_FLAG;
        if !self.xmp_metadata.is_empty() {
            flags |= 1 << 2;
        }
        if !self.exif_metadata.is_empty() {
            flags |= 1 << 3;
        }
        if self.has_alpha {
            flags |= ALPHA_FLAG;
        }
        if !self.icc_profile.is_empty() {
            flags |= 1 << 5;
        }
        flags
    }

    /// The RIFF header with a size to fill in, the `VP8X`, `ICCP` and `ANIM` chunks.
    fn start(&mut self) -> io::Result<()> {
        self.started = true;
        self.writer.write_all(b"RIFF")?;
        self.writer.write_all(&0u32.to_le_bytes())?;
        self.writer.write_all(b"WEBP")?;
        let mut vp8x = vec![self.vp8x_flags(), 0, 0, 0];
        vp8x.extend_from_slice(&(self.width - 1).to_le_bytes()[..3]);
        vp8x.extend_from_slice(&(self.height - 1).to_le_bytes()[..3]);
        write_chunk(&mut self.writer, b"VP8X", &vp8x)?;
        if !self.icc_profile.is_empty() {
            write_chunk(&mut self.writer, b"ICCP", &self.icc_profile)?;
        }
        let mut anim = self.animation.background.to_vec();
        anim.extend_from_slice(&self.animation.loop_count.to_le_bytes());
        write_chunk(&mut self.writer, b"ANIM", &anim)
    }

    /// The one frame as a still, as libwebp writes an animation that has one: the frame itself
    /// when it covers the canvas, else the canvas coded whole.
    fn write_still(&mut self) -> Result<(), EncodingError> {
        let mut frame = self.pending.pop_front().expect("the frame is pending");
        let coded = frame.chosen();
        let full = Rect {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        };
        let data = match coded.rect == full {
            true => std::mem::replace(
                &mut coded.data,
                FrameData {
                    lossless: true,
                    parts: Vec::new(),
                    alpha: None,
                    has_alpha: false,
                },
            ),
            false => match self.params.use_lossy {
                false => encode_lossless(&self.params, self.prev.clone(), self.width, self.height),
                true => {
                    let stride = self.width as usize;
                    encode_lossy(&self.params, self.width, self.height, false, |y, row| {
                        argb_to_rgba(&self.prev[y * stride..][..stride], row)
                    })?
                }
            },
        };
        let mut encoder = WebPEncoder::new(&mut self.writer);
        encoder.set_icc_profile(std::mem::take(&mut self.icc_profile));
        encoder.set_exif_metadata(std::mem::take(&mut self.exif_metadata));
        encoder.set_xmp_metadata(std::mem::take(&mut self.xmp_metadata));
        let fourcc = if data.lossless { b"VP8L" } else { b"VP8 " };
        let lossless_with_alpha = data.lossless && data.has_alpha;
        encoder.write_container(
            fourcc,
            data.parts,
            data.alpha,
            lossless_with_alpha,
            self.width,
            self.height,
        )
    }
}

/// The colour of a block that is not flattened; flattened ones are transparent, so never this.
const KEEP: u32 = u32::MAX;

/// The 8x8 blocks of a lossy frame that become transparent blocks of one colour, row by row
/// from the block at `x`, `y` on the canvas, `columns` blocks to a row.
#[derive(Default)]
struct Flattened {
    x: usize,
    y: usize,
    columns: usize,
    colors: Vec<u32>,
}

impl Flattened {
    /// Puts the flattened blocks over canvas row `y` of the frame, whose `row` of RGBA bytes
    /// starts at canvas column `x`.
    fn apply(&self, y: usize, x: usize, row: &mut [u8]) {
        if self.colors.is_empty() || y < self.y {
            return;
        }
        let start = (y - self.y) / 8 * self.columns;
        let Some(colors) = self.colors.get(start..start + self.columns) else {
            return;
        };
        for (i, &color) in colors.iter().enumerate() {
            if color != KEEP {
                let at = (self.x + i * 8 - x) * 4;
                for px in row[at..at + 32].chunks_exact_mut(4) {
                    px.copy_from_slice(&swap_red_blue(color).to_le_bytes());
                }
            }
        }
    }
}

/// A row of the previous canvas, whose columns `cleared` count as transparent, and the same
/// row of the frame being added.
struct RowPair<'a> {
    prev: &'a [u32],
    cleared: std::ops::Range<usize>,
    curr: &'a [u32],
}

impl RowPair<'_> {
    /// `cols` as the runs over the previous row, over its cleared span and over the row again.
    fn spans(&self, cols: std::ops::Range<usize>) -> [std::ops::Range<usize>; 3] {
        let c0 = self.cleared.start.clamp(cols.start, cols.end);
        let c1 = self.cleared.end.clamp(c0, cols.end);
        [cols.start..c0, c0..c1, c1..cols.end]
    }

    /// The first column of `cols` where `f(previous, current)`.
    fn first(
        &self,
        cols: std::ops::Range<usize>,
        f: impl Fn(u32, u32) -> bool + Copy,
    ) -> Option<usize> {
        let [a, z, b] = self.spans(cols);
        position(&self.prev[a.clone()], &self.curr[a.clone()], f)
            .map(|i| a.start + i)
            .or_else(|| {
                position(&self.curr[z.clone()], &self.curr[z.clone()], |_, c| f(0, c))
                    .map(|i| z.start + i)
            })
            .or_else(|| {
                position(&self.prev[b.clone()], &self.curr[b.clone()], f).map(|i| b.start + i)
            })
    }

    /// The last column of `cols` where `f(previous, current)`.
    fn last(
        &self,
        cols: std::ops::Range<usize>,
        f: impl Fn(u32, u32) -> bool + Copy,
    ) -> Option<usize> {
        let [a, z, b] = self.spans(cols);
        rposition(&self.prev[b.clone()], &self.curr[b.clone()], f)
            .map(|i| b.start + i)
            .or_else(|| {
                rposition(&self.curr[z.clone()], &self.curr[z.clone()], |_, c| f(0, c))
                    .map(|i| z.start + i)
            })
            .or_else(|| {
                rposition(&self.prev[a.clone()], &self.curr[a.clone()], f).map(|i| a.start + i)
            })
    }
}

const SCAN_CHUNK: usize = 16;

/// The first `i` where `f(a[i], b[i])`, tested a chunk at a time without branches.
#[inline(always)]
fn position(a: &[u32], b: &[u32], f: impl Fn(u32, u32) -> bool + Copy) -> Option<usize> {
    let any = |a: &[u32], b: &[u32]| a.iter().zip(b).fold(false, |any, (&x, &y)| any | f(x, y));
    let at = |a: &[u32], b: &[u32]| a.iter().zip(b).position(|(&x, &y)| f(x, y));
    let mut base = 0;
    let (mut ac, mut bc) = (a.chunks_exact(SCAN_CHUNK), b.chunks_exact(SCAN_CHUNK));
    for (x, y) in (&mut ac).zip(&mut bc) {
        if any(x, y) {
            return at(x, y).map(|i| base + i);
        }
        base += SCAN_CHUNK;
    }
    at(ac.remainder(), bc.remainder()).map(|i| base + i)
}

/// The last `i` where `f(a[i], b[i])`, tested a chunk at a time without branches.
#[inline(always)]
fn rposition(a: &[u32], b: &[u32], f: impl Fn(u32, u32) -> bool + Copy) -> Option<usize> {
    let any = |a: &[u32], b: &[u32]| a.iter().zip(b).fold(false, |any, (&x, &y)| any | f(x, y));
    let at = |a: &[u32], b: &[u32]| a.iter().zip(b).rposition(|(&x, &y)| f(x, y));
    let n = a.len();
    let head = n % SCAN_CHUNK;
    let (ac, bc) = (
        a[head..].rchunks_exact(SCAN_CHUNK),
        b[head..].rchunks_exact(SCAN_CHUNK),
    );
    let mut end = n;
    for (x, y) in ac.zip(bc) {
        end -= SCAN_CHUNK;
        if any(x, y) {
            return at(x, y).map(|i| end + i);
        }
    }
    at(&a[..head], &b[..head])
}

/// `data` of `color` as ARGB words.
fn import(data: &[u8], color: ColorType, argb: &mut [u32]) {
    match color {
        ColorType::L8 => {
            assert_eq!(data.len(), argb.len());
            for (p, &l) in argb.iter_mut().zip(data) {
                *p = 0xff000000 | (u32::from(l) * 0x010101);
            }
        }
        ColorType::La8 => {
            assert_eq!(data.len(), argb.len() * 2);
            for (p, la) in argb.iter_mut().zip(data.chunks_exact(2)) {
                *p = (u32::from(la[1]) << 24) | (u32::from(la[0]) * 0x010101);
            }
        }
        ColorType::Rgb8 => {
            assert_eq!(data.len(), argb.len() * 3);
            for (p, px) in argb.iter_mut().zip(data.chunks_exact(3)) {
                *p = 0xff000000 | u32::from(px[0]) << 16 | u32::from(px[1]) << 8 | u32::from(px[2]);
            }
        }
        ColorType::Rgba8 => {
            assert_eq!(data.len(), argb.len() * 4);
            for (p, px) in argb.iter_mut().zip(data.chunks_exact(4)) {
                *p = u32::from(px[3]) << 24
                    | u32::from(px[0]) << 16
                    | u32::from(px[1]) << 8
                    | u32::from(px[2]);
            }
        }
    }
}

/// ARGB words as RGBA bytes, a word at a time so that it vectorizes.
fn argb_to_rgba(argb: &[u32], rgba: &mut [u8]) {
    for (px, &p) in rgba.chunks_exact_mut(4).zip(argb) {
        px.copy_from_slice(&swap_red_blue(p).to_le_bytes());
    }
}

/// ARGB as the little-endian word of RGBA bytes, and back.
#[inline(always)]
fn swap_red_blue(p: u32) -> u32 {
    (p & 0xff00ff00) | (p >> 16) & 0xff | (p & 0xff) << 16
}

/// `argb` through YUV 4:2:0 and back, as `WebPPictureImportRGBA` and `WebPPictureYUVAToARGB`
/// take a lossy frame: the crate's encoder-side conversion and its decoder-side upsampling.
/// Each macroblock row is converted back as soon as the chroma below its rows is there, so
/// only the converter's strip is held.
fn chroma_roundtrip(argb: &mut [u32], width: u32, height: u32) {
    let (w, h) = (width as usize, height as usize);
    let uv_h = h.div_ceil(2);
    let canvas = std::cell::RefCell::new(argb);
    let mut src = vp8enc::csp::Source::new(w, h, 4, |y, row| {
        argb_to_rgba(&canvas.borrow()[y * w..][..w], row)
    });
    src.keep_alpha = false;
    let (mut u_full, mut v_full, mut rgba) = (vec![0u8; w], vec![0u8; w], vec![0u8; w * 4]);
    let strips = h.div_ceil(16);
    for mb_y in 0..strips {
        src.load(mb_y);
        let uv_w = src.uv_stride;
        // Strip row 0 holds the row above the strip: luma row 16 * mb_y - 1, chroma 8 * mb_y - 1.
        let (y_base, uv_base) = ((mb_y * 16) as isize - 1, (mb_y * 8) as isize - 1);
        let first = (mb_y * 16).saturating_sub(1);
        let end = if mb_y + 1 == strips {
            h
        } else {
            mb_y * 16 + 15
        };
        for y in first..end {
            let (near, far) = if y == 0 {
                (0, 0)
            } else if y + 1 == h && h % 2 == 0 {
                (uv_h - 1, uv_h - 1)
            } else {
                let k = y.div_ceil(2);
                if y % 2 == 1 {
                    (k - 1, k)
                } else {
                    (k, k - 1)
                }
            };
            let uv_row = |c: usize| (c as isize - uv_base) as usize * uv_w;
            let (near, far) = (uv_row(near), uv_row(far));
            upsample_chroma_row(&src.u[near..][..uv_w], &src.u[far..][..uv_w], &mut u_full);
            upsample_chroma_row(&src.v[near..][..uv_w], &src.v[far..][..uv_w], &mut v_full);
            let luma = &src.y[(y as isize - y_base) as usize * src.y_stride..][..w];
            fill_row_yuv::<4>(&mut rgba, luma, &u_full, &v_full);
            let mut canvas = canvas.borrow_mut();
            for (p, px) in canvas[y * w..][..w].iter_mut().zip(rgba.chunks_exact(4)) {
                let v = u32::from_le_bytes(px.try_into().unwrap());
                *p = (*p & 0xff000000) | swap_red_blue(v) & 0xffffff;
            }
        }
    }
}

/// libwebp's `QualityToMaxDiff`: how far apart lossy pixels may be to count as the same.
fn quality_to_max_diff(quality: f32) -> i32 {
    let val = (f64::from(quality) / 100.).sqrt();
    (31. * (1. - val) + val + 0.5) as i32
}

/// libwebp's `PixelsAreSimilar`, without branches so that scans over it vectorize.
#[inline(always)]
fn similar(src: u32, dst: u32, max_diff: i32) -> bool {
    let dst_a = (dst >> 24) as i32;
    let limit = max_diff * 255;
    let close = |c: u32| {
        let s = ((src >> c) & 0xff) as i32;
        let d = ((dst >> c) & 0xff) as i32;
        (s - d).abs() * dst_a <= limit
    };
    ((src >> 24) as i32 == dst_a) & close(0) & close(8) & close(16)
}

/// Codes `argb` (`width` x `height`) as a `VP8L` payload.
fn encode_lossless(params: &EncoderParams, argb: Vec<u32>, width: u32, height: u32) -> FrameData {
    let config = vp8l::Config {
        quality: f32::from(params.lossless_quality.min(100)),
        method: i32::from(params.lossless_method.min(6)),
        exact: params.lossless_exact,
        use_predictor: params.use_predictor_transform,
        near_lossless: i32::from(params.near_lossless.min(100)),
    };
    let payload = vp8l::encode_argb(config, argb, width, height);
    let has_alpha = payload[4] & 0x10 != 0;
    FrameData {
        lossless: true,
        parts: vec![payload],
        alpha: None,
        has_alpha,
    }
}

/// Codes the `width` x `height` RGBA rows `read_row(y, row)` writes as a `VP8 ` payload with
/// the `ALPH` chunk of its alpha plane. A `blended` frame is coded without the loop filter, as
/// libwebp codes it, so its transparent blocks stay cheap.
fn encode_lossy(
    params: &EncoderParams,
    width: u32,
    height: u32,
    blended: bool,
    mut read_row: impl FnMut(usize, &mut [u8]),
) -> Result<FrameData, EncodingError> {
    let mut config = vp8enc::Config::new(
        f32::from(params.lossy_quality.min(100)),
        i32::from(params.lossy_method.min(6)),
    );
    config.keep_strips = true;
    if blended {
        config.filter_strength = 0;
    }
    let (parts, (has_alpha, alpha)) = vp8enc::encode_frame_lossy_with(
        config,
        width,
        height,
        ColorType::Rgba8,
        true,
        |y, row| read_row(y as usize, row),
    )?;
    let alpha = alpha
        .filter(|_| has_alpha)
        .map(|a| alpha_chunk_payload(&a, width, height, params));
    Ok(FrameData {
        lossless: false,
        parts,
        alpha,
        has_alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_diff_follows_quality() {
        assert_eq!(quality_to_max_diff(100.), 1);
        assert_eq!(quality_to_max_diff(75.), 5);
        assert_eq!(quality_to_max_diff(0.), 31);
    }

    #[test]
    fn rects_snap_to_even_offsets() {
        let mut r = Rect {
            x: 3,
            y: 5,
            width: 4,
            height: 6,
        };
        r.snap_to_even();
        assert_eq!(
            r,
            Rect {
                x: 2,
                y: 4,
                width: 5,
                height: 7
            }
        );
    }
}
