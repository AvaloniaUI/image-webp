//! RGB(A) to YUV420(A) as libwebp's `WebPPictureImportRGB(A)` does it without sharp YUV:
//! chroma averaged over 2x2 pixels in a 0.8 gamma, weighted by alpha where it is not opaque.

use std::sync::OnceLock;

use super::tables::INV_ALPHA;

const YUV_FIX: i32 = 16;
const YUV_HALF: i32 = 1 << (YUV_FIX - 1);
const GAMMA_FIX: i32 = 12;
const GAMMA_TAB_FIX: i32 = 7;
const GAMMA_TAB_SIZE: usize = 1 << (GAMMA_FIX - GAMMA_TAB_FIX);
const GAMMA_SCALE: i32 = (1 << GAMMA_FIX) - 1;
const GAMMA_TAB_SCALE: i32 = 1 << GAMMA_TAB_FIX;
const GAMMA_TAB_ROUNDER: i32 = GAMMA_TAB_SCALE >> 1;
const ALPHA_FIX: u32 = 19;

const SUM4_SIZE: usize = 4 << GAMMA_FIX;

struct GammaTables {
    to_linear: [u16; 256],
    to_gamma: [i32; GAMMA_TAB_SIZE + 1],
    /// `gamma(v, 0)` for every sum of four linear values.
    sum4_to_gamma: Box<[u16; SUM4_SIZE]>,
}

fn gamma_tables() -> &'static GammaTables {
    static TABLES: OnceLock<GammaTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        const GAMMA: f64 = 0.80;
        let scale = f64::from(1 << GAMMA_TAB_FIX) / f64::from(GAMMA_SCALE);
        let norm = 1. / 255.;
        let mut to_linear = [0u16; 256];
        for (v, t) in to_linear.iter_mut().enumerate() {
            *t = ((norm * v as f64).powf(GAMMA) * f64::from(GAMMA_SCALE) + 0.5) as u16;
        }
        let mut to_gamma = [0i32; GAMMA_TAB_SIZE + 1];
        for (v, t) in to_gamma.iter_mut().enumerate() {
            *t = (255. * (scale * v as f64).powf(1. / GAMMA) + 0.5) as i32;
        }
        let mut tables = GammaTables {
            to_linear,
            to_gamma,
            sum4_to_gamma: Box::new([0; SUM4_SIZE]),
        };
        for v in 0..=4 * GAMMA_SCALE as usize {
            tables.sum4_to_gamma[v] = tables.gamma(v as u32, 0) as u16;
        }
        tables
    })
}

impl GammaTables {
    #[inline(always)]
    fn linear(&self, v: u8) -> u32 {
        u32::from(self.to_linear[usize::from(v)])
    }

    #[inline(always)]
    fn gamma(&self, base_value: u32, shift: u32) -> i32 {
        let v = (base_value << shift) as i32;
        let tab_pos = (v >> (GAMMA_TAB_FIX + 2)) as usize;
        let x = v & ((GAMMA_TAB_SCALE << 2) - 1);
        let v0 = self.to_gamma[tab_pos];
        let v1 = self.to_gamma[tab_pos + 1];
        let y = v1 * x + v0 * ((GAMMA_TAB_SCALE << 2) - x);
        (y + GAMMA_TAB_ROUNDER) >> GAMMA_TAB_FIX
    }
}

/// Packed RGB(A) rows converted to YUV420 one macroblock row (16 rows) at a time, as the
/// encoder reaches it. Row 0 of the strip buffers holds the row above the strip.
pub(crate) struct Source<'a> {
    read_row: Box<dyn FnMut(usize, &mut [u8]) + 'a>,
    step: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) y_stride: usize,
    pub(crate) uv_stride: usize,
    pub(crate) y: Vec<u8>,
    pub(crate) u: Vec<u8>,
    pub(crate) v: Vec<u8>,
    strip: Option<usize>,
    tmp: Vec<[u16; 4]>,
    /// The packed rows of the current strip.
    rgb: Vec<u8>,
    /// The alpha plane, once a pixel that is not opaque has been read and if `keep_alpha`.
    alpha: Option<Vec<u8>>,
    pub(crate) keep_alpha: bool,
    has_alpha: bool,
    /// Rows whose alpha has been looked at.
    alpha_rows: usize,
    /// The alpha of the band `cleanup_band` works on.
    band_alpha: Vec<u8>,
    /// Whether each row of the strip in `rgb` is opaque.
    opaque_rows: [bool; 16],
    cache: Option<StripCache>,
}

/// The converted rows of the strips loaded so far, for the passes after the first.
#[derive(Default)]
struct StripCache {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    strips: usize,
}

impl<'a> Source<'a> {
    /// Rows of packed RGB (`step` 3) or RGBA (`step` 4) pixels, which `read_row(y, row)`
    /// writes into `row`. Rows are read top to bottom, once per pass over the picture.
    pub(crate) fn new(
        width: usize,
        height: usize,
        step: usize,
        read_row: impl FnMut(usize, &mut [u8]) + 'a,
    ) -> Self {
        let uv_width = width.div_ceil(2);
        Self {
            read_row: Box::new(read_row),
            step,
            width,
            height,
            y_stride: width,
            uv_stride: uv_width,
            y: vec![0; width * 17],
            u: vec![0; uv_width * 9],
            v: vec![0; uv_width * 9],
            strip: None,
            tmp: vec![[0; 4]; uv_width],
            rgb: vec![0; width * step * 16],
            alpha: None,
            keep_alpha: true,
            has_alpha: false,
            alpha_rows: 0,
            band_alpha: Vec::new(),
            opaque_rows: [true; 16],
            cache: None,
        }
    }

    /// Keeps the converted picture (1.5 bytes a pixel) so that later passes copy strips
    /// instead of reading and converting the rows again.
    pub(crate) fn keep_strips(&mut self) {
        self.cache = Some(StripCache::default());
    }

    /// Whether some pixel is not opaque, and then the alpha plane if `keep_alpha`. Valid once
    /// every row has been read.
    pub(crate) fn take_alpha(&mut self) -> (bool, Option<Vec<u8>>) {
        debug_assert_eq!(self.alpha_rows, self.height);
        (self.has_alpha, self.alpha.take())
    }

    /// Makes macroblock row `mb_y` (and the row above it) available in `y`, `u` and `v`.
    pub(crate) fn load(&mut self, mb_y: usize) {
        if self.strip == Some(mb_y) {
            return;
        }
        if mb_y > 0 {
            if self.strip != Some(mb_y - 1) {
                self.load(mb_y - 1);
            }
            let (ys, uvs) = (self.y_stride, self.uv_stride);
            self.y.copy_within(16 * ys..17 * ys, 0);
            self.u.copy_within(8 * uvs..9 * uvs, 0);
            self.v.copy_within(8 * uvs..9 * uvs, 0);
        }
        let y0 = mb_y * 16;
        let y1 = (y0 + 16).min(self.height);
        let (ys, uvs) = (self.y_stride, self.uv_stride);
        let (y_len, uv_len) = ((y1 - y0) * ys, (y1 - y0).div_ceil(2) * uvs);
        if let Some(c) = self.cache.as_ref().filter(|c| mb_y < c.strips) {
            self.y[ys..][..y_len].copy_from_slice(&c.y[y0 * ys..][..y_len]);
            self.u[uvs..][..uv_len].copy_from_slice(&c.u[y0 / 2 * uvs..][..uv_len]);
            self.v[uvs..][..uv_len].copy_from_slice(&c.v[y0 / 2 * uvs..][..uv_len]);
            self.strip = Some(mb_y);
            return;
        }
        self.read_rows(y0, y1);
        self.convert_rows(y0, y1);
        if self.step == 4 {
            // A no-op on opaque bands.
            for band in (y0..y1).step_by(8) {
                self.cleanup_band(y0, band, (y1 - band).min(8));
            }
        }
        if let Some(c) = self.cache.as_mut().filter(|c| mb_y == c.strips) {
            c.y.extend_from_slice(&self.y[ys..][..y_len]);
            c.u.extend_from_slice(&self.u[uvs..][..uv_len]);
            c.v.extend_from_slice(&self.v[uvs..][..uv_len]);
            c.strips += 1;
        }
        self.strip = Some(mb_y);
    }

    fn read_rows(&mut self, y0: usize, y1: usize) {
        let stride = self.width * self.step;
        for row in y0..y1 {
            let line = &mut self.rgb[(row - y0) * stride..][..stride];
            (self.read_row)(row, line);
            let is_opaque = self.step == 3 || opaque(line);
            self.opaque_rows[row - y0] = is_opaque;
            if self.step == 4 && row == self.alpha_rows {
                if !self.has_alpha && !is_opaque {
                    self.has_alpha = true;
                    if self.keep_alpha {
                        self.alpha = Some(vec![0xff; self.width * self.height]);
                    }
                }
                if let Some(alpha) = self.alpha.as_mut() {
                    alpha_of(line, &mut alpha[row * self.width..][..self.width]);
                }
                self.alpha_rows += 1;
            }
        }
        if self.step == 3 {
            self.alpha_rows = self.alpha_rows.max(y1);
        }
    }

    /// Converts rows `y0..y1` (`y0` even) from `rgb` into strip rows from 1.
    fn convert_rows(&mut self, y0: usize, y1: usize) {
        let gt = gamma_tables();
        let (width, step) = (self.width, self.step);
        let stride = width * step;
        for row in (y0..y1).step_by(2) {
            let two = row + 1 < y1;
            let rgb_stride = if two { stride } else { 0 };
            let src = &self.rgb[(row - y0) * stride..];
            for dy in 0..if two { 2 } else { 1 } {
                let line = &src[dy * stride..];
                let out = &mut self.y[(row + dy - y0 + 1) * width..][..width];
                if step == 3 {
                    convert_y::<3>(line, out);
                } else {
                    convert_y::<4>(line, out);
                }
            }
            let rows_have_alpha = !self.opaque_rows[row - y0..][..if two { 2 } else { 1 }]
                .iter()
                .all(|&o| o);
            if rows_have_alpha {
                accumulate_rgba(gt, src, rgb_stride, width, &mut self.tmp);
            } else {
                accumulate_rgb(gt, src, step, rgb_stride, width, &mut self.tmp);
            }
            let uv_row = ((row - y0) / 2 + 1) * self.uv_stride;
            let n = self.tmp.len();
            let (u, v) = (&mut self.u[uv_row..][..n], &mut self.v[uv_row..][..n]);
            convert_uv(&self.tmp, u, v);
        }
    }

    /// libwebp's `WebPCleanupTransparentArea` for the 8-row band at `band` (`height` rows):
    /// flattens fully transparent 8x8 blocks and sets transparent luma to the average of the
    /// visible pixels in partly transparent ones.
    fn cleanup_band(&mut self, y0: usize, band: usize, height: usize) {
        const SIZE: usize = 8;
        let width = self.width;
        let strip_row = band - y0 + 1;
        if self.opaque_rows[band - y0..][..height].iter().all(|&o| o) {
            return;
        }
        let rows = &self.rgb[(band - y0) * width * 4..][..height * width * 4];
        let mut alpha = std::mem::take(&mut self.band_alpha);
        alpha.resize(rows.len() / 4, 0);
        alpha_of(rows, &mut alpha);
        let (ys, uvs) = (self.y_stride, self.uv_stride);
        let luma = &mut self.y[strip_row * ys..];
        let mut x = 0;
        if height == SIZE {
            let uv_row = (strip_row - 1) / 2 + 1;
            let mut values = [0u8; 3];
            let mut need_reset = true;
            while x + SIZE <= width {
                if smoothen_block(&alpha[x..], width, &mut luma[x..], ys, SIZE, SIZE) {
                    if need_reset {
                        values = [
                            luma[x],
                            self.u[uv_row * uvs + x / 2],
                            self.v[uv_row * uvs + x / 2],
                        ];
                        need_reset = false;
                    }
                    flatten(&mut luma[x..], values[0], ys, SIZE);
                    flatten(
                        &mut self.u[uv_row * uvs + x / 2..],
                        values[1],
                        uvs,
                        SIZE / 2,
                    );
                    flatten(
                        &mut self.v[uv_row * uvs + x / 2..],
                        values[2],
                        uvs,
                        SIZE / 2,
                    );
                } else {
                    need_reset = true;
                }
                x += SIZE;
            }
        } else {
            while x + SIZE <= width {
                smoothen_block(&alpha[x..], width, &mut luma[x..], ys, SIZE, height);
                x += SIZE;
            }
        }
        if x < width {
            smoothen_block(&alpha[x..], width, &mut luma[x..], ys, width - x, height);
        }
        self.band_alpha = alpha;
    }
}

/// Whether every pixel of the RGBA `rows` is opaque. Whole words rather than every fourth
/// byte, which vectorizes.
fn opaque(rows: &[u8]) -> bool {
    let all = rows.chunks_exact(4).fold(u32::MAX, |acc, p| {
        acc & u32::from_le_bytes(p.try_into().unwrap())
    });
    all >> 24 == 0xff
}

/// The alpha bytes of the RGBA `rows`.
fn alpha_of(rows: &[u8], alpha: &mut [u8]) {
    for (a, p) in alpha.iter_mut().zip(rows.chunks_exact(4)) {
        *a = (u32::from_le_bytes(p.try_into().unwrap()) >> 24) as u8;
    }
}

fn convert_uv(tmp: &[[u16; 4]], u: &mut [u8], v: &mut [u8]) {
    for ((t, u), v) in tmp.iter().zip(u).zip(v) {
        // At most 4 * 255, so i16 is lossless and lets the products become pmaddwd.
        let (r, g, b) = (
            i32::from(t[0] as i16),
            i32::from(t[1] as i16),
            i32::from(t[2] as i16),
        );
        *u = clip_uv(-9719 * r - 19081 * g + 28800 * b);
        *v = clip_uv(28800 * r - 24116 * g - 4684 * b);
    }
}

#[inline(always)]
fn clip_uv(uv: i32) -> u8 {
    ((uv + (YUV_HALF << 2) + (128 << (YUV_FIX + 2))) >> (YUV_FIX + 2)).clamp(0, 255) as u8
}

/// `rgb_to_y` in f32, exact because every intermediate is an integer below 2^24 and the
/// floor is taken without a float to int conversion, which SSE2 cannot do saturating.
#[inline(always)]
fn rgb_to_y_f32(r: u8, g: u8, b: u8) -> u8 {
    const MAGIC: f32 = (1 << 23) as f32;
    let luma = 16839. * f32::from(r) + 33059. * f32::from(g) + 6420. * f32::from(b);
    let x = (luma + (YUV_HALF + (16 << YUV_FIX)) as f32) * (1. / (1 << YUV_FIX) as f32);
    let y = x + MAGIC;
    let rounded_up = (y - MAGIC > x) as u32;
    (y.to_bits() - MAGIC.to_bits() - rounded_up) as u8
}

fn convert_y<const STEP: usize>(line: &[u8], out: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    let (line, out) = {
        let done = super::neon::rgb_to_y_row::<STEP>(line, out);
        (&line[done * STEP..], &mut out[done..])
    };
    #[cfg(target_arch = "x86_64")]
    let (line, out) = {
        let done = match STEP {
            4 => super::sse2::rgba_to_y_row(line, out),
            _ => super::sse2::rgb_to_y_row(line, out),
        };
        (&line[done * STEP..], &mut out[done..])
    };
    let Some(last) = out.len().checked_sub(1) else {
        return;
    };
    let line = &line[..out.len() * STEP];
    // Each pixel read as one word, the last one by itself as its word would pass the row end.
    for (i, o) in out[..last].iter_mut().enumerate() {
        let w = u32::from_le_bytes(line[i * STEP..][..4].try_into().unwrap());
        *o = rgb_to_y_f32(w as u8, (w >> 8) as u8, (w >> 16) as u8);
    }
    let p = &line[last * STEP..];
    out[last] = rgb_to_y_f32(p[0], p[1], p[2]);
}

fn accumulate_rgb(
    gt: &GammaTables,
    src: &[u8],
    step: usize,
    rgb_stride: usize,
    width: usize,
    dst: &mut [[u16; 4]],
) {
    let r0 = &src[..width * step];
    let r1 = &src[rgb_stride..][..width * step];
    if step == 3 {
        accumulate_rows::<3>(gt, r0, r1, &mut dst[..width.div_ceil(2)]);
    } else {
        accumulate_rows::<4>(gt, r0, r1, &mut dst[..width.div_ceil(2)]);
    }
}

#[inline(always)]
fn sum4<const STEP: usize>(gt: &GammaTables, a: &[u8], b: &[u8], c: usize) -> u16 {
    let sum = gt.linear(a[c]) + gt.linear(a[c + STEP]) + gt.linear(b[c]) + gt.linear(b[c + STEP]);
    gt.sum4_to_gamma[sum as usize & (SUM4_SIZE - 1)]
}

#[inline(always)]
fn accumulate_rows<const STEP: usize>(
    gt: &GammaTables,
    r0: &[u8],
    r1: &[u8],
    dst: &mut [[u16; 4]],
) {
    let pairs = r0.chunks_exact(2 * STEP).zip(r1.chunks_exact(2 * STEP));
    for (d, (a, b)) in dst.iter_mut().zip(pairs) {
        *d = [
            sum4::<STEP>(gt, a, b, 0),
            sum4::<STEP>(gt, a, b, 1),
            sum4::<STEP>(gt, a, b, 2),
            0,
        ];
    }
    if r0.len() % (2 * STEP) != 0 {
        let (a, b) = (&r0[r0.len() - STEP..], &r1[r1.len() - STEP..]);
        let s = |c: usize| gt.gamma(gt.linear(a[c]) + gt.linear(b[c]), 1) as u16;
        dst[dst.len() - 1] = [s(0), s(1), s(2), 0];
    }
}

/// The alpha-weighted average of channel `c` over the 2x2 pixels `a` (top) and `b` (bottom)
/// whose alpha sums to `total_a`.
#[inline(always)]
fn weighted4(gt: &GammaTables, a: &[u8], b: &[u8], c: usize, total_a: u32) -> u16 {
    let sum = u32::from(a[3]) * gt.linear(a[c])
        + u32::from(a[7]) * gt.linear(a[c + 4])
        + u32::from(b[3]) * gt.linear(b[c])
        + u32::from(b[7]) * gt.linear(b[c + 4]);
    gt.gamma((sum * INV_ALPHA[total_a as usize]) >> (ALPHA_FIX - 2), 0) as u16
}

fn accumulate_rgba(
    gt: &GammaTables,
    src: &[u8],
    rgb_stride: usize,
    width: usize,
    dst: &mut [[u16; 4]],
) {
    let r0 = &src[..width * 4];
    let r1 = &src[rgb_stride..][..width * 4];
    let pairs = r0.chunks_exact(8).zip(r1.chunks_exact(8));
    for (d, (a, b)) in dst.iter_mut().zip(pairs) {
        let total = u32::from(a[3]) + u32::from(a[7]) + u32::from(b[3]) + u32::from(b[7]);
        *d = if total == 4 * 0xff || total == 0 {
            [
                sum4::<4>(gt, a, b, 0),
                sum4::<4>(gt, a, b, 1),
                sum4::<4>(gt, a, b, 2),
                total as u16,
            ]
        } else {
            [
                weighted4(gt, a, b, 0, total),
                weighted4(gt, a, b, 1, total),
                weighted4(gt, a, b, 2, total),
                total as u16,
            ]
        };
    }
    if width & 1 != 0 {
        let (a, b) = (&r0[r0.len() - 4..], &r1[r1.len() - 4..]);
        let total = 2 * (u32::from(a[3]) + u32::from(b[3]));
        let s = |c: usize| match total {
            0 | 0x3fc => gt.gamma(gt.linear(a[c]) + gt.linear(b[c]), 1) as u16,
            _ => {
                let sum =
                    2 * (u32::from(a[3]) * gt.linear(a[c]) + u32::from(b[3]) * gt.linear(b[c]));
                gt.gamma((sum * INV_ALPHA[total as usize]) >> (ALPHA_FIX - 2), 0) as u16
            }
        };
        dst[width / 2] = [s(0), s(1), s(2), total as u16];
    }
}

fn smoothen_block(
    a: &[u8],
    a_stride: usize,
    luma: &mut [u8],
    y_stride: usize,
    width: usize,
    height: usize,
) -> bool {
    if width == 8 {
        // Fully transparent or fully visible blocks, told from eight alpha bytes at a time.
        let (mut any, mut zero_byte) = (0u64, 0u64);
        for y in 0..height {
            let r = u64::from_le_bytes(a[y * a_stride..][..8].try_into().unwrap());
            any |= r;
            zero_byte |= r.wrapping_sub(0x0101_0101_0101_0101) & !r & 0x8080_8080_8080_8080;
        }
        if any == 0 {
            return true;
        }
        if zero_byte == 0 {
            return false;
        }
    }
    let (mut sum, mut count) = (0u32, 0u32);
    for y in 0..height {
        let (a, luma) = (&a[y * a_stride..][..width], &luma[y * y_stride..][..width]);
        for (&a, &l) in a.iter().zip(luma) {
            let visible = u32::from(a != 0);
            count += visible;
            sum += visible * u32::from(l);
        }
    }
    if count > 0 && (count as usize) < width * height {
        let avg = (sum / count) as u8;
        for y in 0..height {
            let (a, luma) = (
                &a[y * a_stride..][..width],
                &mut luma[y * y_stride..][..width],
            );
            for (&a, l) in a.iter().zip(luma) {
                *l = if a == 0 { avg } else { *l };
            }
        }
    }
    count == 0
}

fn flatten(p: &mut [u8], v: u8, stride: usize, size: usize) {
    for y in 0..size {
        p[y * stride..y * stride + size].fill(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_to_y(r: i32, g: i32, b: i32) -> u8 {
        let luma = 16839 * r + 33059 * g + 6420 * b;
        ((luma + YUV_HALF + (16 << YUV_FIX)) >> YUV_FIX) as u8
    }

    #[test]
    fn convert_y_matches_reference() {
        let mut seed = 1u32;
        let line: Vec<u8> = (0..4 * 140)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (seed >> 23) as u8
            })
            .collect();
        for width in 0..140 {
            let mut out = vec![0u8; width];
            convert_y::<3>(&line, &mut out);
            for (i, &y) in out.iter().enumerate() {
                let p = &line[3 * i..];
                assert_eq!(
                    y,
                    rgb_to_y(p[0].into(), p[1].into(), p[2].into()),
                    "rgb {width}"
                );
            }
            convert_y::<4>(&line, &mut out);
            for (i, &y) in out.iter().enumerate() {
                let p = &line[4 * i..];
                assert_eq!(
                    y,
                    rgb_to_y(p[0].into(), p[1].into(), p[2].into()),
                    "rgba {width}"
                );
            }
        }
    }

    #[test]
    fn rgb_to_y_f32_is_exact() {
        for r in 0..=255u8 {
            for g in 0..=255u8 {
                for b in 0..=255u8 {
                    assert_eq!(
                        rgb_to_y_f32(r, g, b),
                        rgb_to_y(i32::from(r), i32::from(g), i32::from(b))
                    );
                }
            }
        }
    }
}
