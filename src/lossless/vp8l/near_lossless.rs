//! Near-lossless preprocessing: pixels are moved within a bound before the lossless encode so
//! that their residuals pack better. Two stages as libwebp's `near_lossless_enc.c` and
//! `predictor_enc.c`: a picture smoothing that discretizes every pixel not on a smooth patch
//! (used when the stream has no predictor transform), and a quantization of each predictor
//! residual bounded by the pixel's surroundings (used when it has one).

use super::entropy::sub_pixels;

const MIN_DIM: usize = 64;
const MASK_ALPHA: u32 = 0xff000000;

/// `VP8LNearLosslessBits`: how many low bits a `quality` of 0..=100 may lose, 0 at 100.
pub(crate) fn limit_bits(quality: i32) -> u32 {
    (5 - quality.clamp(0, 100) / 20) as u32
}

/// Quantization step of the residual stage: `1 << limit_bits`, 1 when near-lossless is off.
pub(crate) fn max_quantization(quality: i32) -> u32 {
    1 << limit_bits(quality)
}

fn closest_discretized(a: u32, bits: u32) -> u32 {
    let mask = (1u32 << bits) - 1;
    let biased = a + (mask >> 1) + ((a >> bits) & 1);
    if biased > 0xff {
        0xff
    } else {
        biased & !mask
    }
}

fn closest_discretized_argb(a: u32, bits: u32) -> u32 {
    closest_discretized(a >> 24, bits) << 24
        | closest_discretized((a >> 16) & 0xff, bits) << 16
        | closest_discretized((a >> 8) & 0xff, bits) << 8
        | closest_discretized(a & 0xff, bits)
}

fn is_near(a: u32, b: u32, limit: i32) -> bool {
    (0..4).all(|k| {
        let delta = ((a >> (k * 8)) & 0xff) as i32 - ((b >> (k * 8)) & 0xff) as i32;
        delta < limit && delta > -limit
    })
}

/// One smoothing pass over rows of ARGB pixels: a pixel whose four neighbours are all within
/// `1 << bits` keeps its value, any other loses its `bits` low bits to the closest step. The
/// first and last rows and columns are kept. Rows come out one row behind the rows put in.
struct Pass {
    bits: u32,
    rows: [Vec<u32>; 3],
    held: usize,
}

impl Pass {
    fn new(width: usize, bits: u32) -> Pass {
        Pass {
            bits,
            rows: [vec![0; width], vec![0; width], vec![0; width]],
            held: 0,
        }
    }

    /// Pushes `row` in; returns the row before it, smoothed with `row` as its lower neighbour.
    fn push(&mut self, row: &[u32], out: &mut [u32]) -> bool {
        self.rows.rotate_left(1);
        self.rows[2].copy_from_slice(row);
        self.held += 1;
        if self.held < 2 {
            return false;
        }
        if self.held == 2 {
            out.copy_from_slice(&self.rows[1]);
            return true;
        }
        self.smooth(out);
        true
    }

    fn finish(&mut self, out: &mut [u32]) -> bool {
        if self.held == 0 {
            return false;
        }
        out.copy_from_slice(&self.rows[2]);
        true
    }

    fn smooth(&self, out: &mut [u32]) {
        let [prev, cur, next] = &self.rows;
        let limit = 1i32 << self.bits;
        let n = cur.len();
        out[0] = cur[0];
        out[n - 1] = cur[n - 1];
        for x in 1..n - 1 {
            let c = cur[x];
            out[x] = if is_near(c, cur[x - 1], limit)
                && is_near(c, cur[x + 1], limit)
                && is_near(c, prev[x], limit)
                && is_near(c, next[x], limit)
            {
                c
            } else {
                closest_discretized_argb(c, self.bits)
            };
        }
    }
}

/// The picture stage as a chain of passes with `limit_bits(quality)` down to 1 bit, fed a row
/// at a time; `VP8ApplyNearLossless` on rows instead of the whole picture. Each pass keeps
/// three rows, so a row comes out `limit_bits` rows after it went in. `None` when nothing is
/// to be done: quality 100, or a picture under 64 pixels each way or under 3 rows.
pub(crate) struct NearLossless {
    passes: Vec<Pass>,
    buffers: Vec<Vec<u32>>,
    width: usize,
}

impl NearLossless {
    pub(crate) fn new(width: usize, height: usize, quality: i32) -> Option<NearLossless> {
        let bits = limit_bits(quality);
        if bits == 0 || (width < MIN_DIM && height < MIN_DIM) || height < 3 {
            return None;
        }
        let passes = (1..=bits).rev().map(|b| Pass::new(width, b)).collect();
        let buffers = (0..bits).map(|_| vec![0; width]).collect();
        Some(NearLossless {
            passes,
            buffers,
            width,
        })
    }

    /// Pushes a row in; `out` gets each finished row.
    pub(crate) fn push(&mut self, row: &[u32], mut out: impl FnMut(&[u32])) {
        debug_assert_eq!(row.len(), self.width);
        let mut buffers = std::mem::take(&mut self.buffers);
        let mut input: &[u32] = row;
        let mut done = true;
        for (pass, buf) in self.passes.iter_mut().zip(buffers.iter_mut()) {
            if !pass.push(input, buf) {
                done = false;
                break;
            }
            input = buf;
        }
        if done {
            out(input);
        }
        self.buffers = buffers;
    }

    /// Drains the rows still held, in order, to `out`.
    pub(crate) fn finish(mut self, mut out: impl FnMut(&[u32])) {
        let passes = self.passes.len();
        let mut buffers = std::mem::take(&mut self.buffers);
        let mut flush = vec![0u32; self.width];
        for first in 0..passes {
            if !self.passes[first].finish(&mut flush) {
                continue;
            }
            let mut input: &[u32] = &flush;
            let mut done = true;
            for (pass, buf) in self.passes[first + 1..].iter_mut().zip(buffers.iter_mut()) {
                if !pass.push(input, buf) {
                    done = false;
                    break;
                }
                input = buf;
            }
            if done {
                out(input);
            }
        }
    }

    /// The picture stage over a whole picture, in place.
    pub(crate) fn apply(argb: &mut [u32], width: usize, height: usize, quality: i32) {
        let Some(mut chain) = NearLossless::new(width, height, quality) else {
            return;
        };
        let mut y_out = 0;
        let mut row = vec![0u32; width];
        for y in 0..height {
            row.copy_from_slice(&argb[y * width..(y + 1) * width]);
            chain.push(&row, |done| {
                argb[y_out * width..(y_out + 1) * width].copy_from_slice(done);
                y_out += 1;
            });
        }
        chain.finish(|done| {
            argb[y_out * width..(y_out + 1) * width].copy_from_slice(done);
            y_out += 1;
        });
        debug_assert_eq!(y_out, height);
    }
}

fn max_diff_between_pixels(p1: u32, p2: u32) -> u8 {
    let d = |s: u32| (((p1 >> s) & 0xff) as i32 - ((p2 >> s) & 0xff) as i32).unsigned_abs();
    d(24).max(d(16)).max(d(8)).max(d(0)) as u8
}

fn add_green_to_blue_and_red(argb: u32) -> u32 {
    let green = (argb >> 8) & 0xff;
    let red_blue = (argb & 0x00ff00ff) + ((green << 16) | green);
    (argb & 0xff00ff00) | (red_blue & 0x00ff00ff)
}

/// `MaxDiffsForRow`: the largest component difference between each pixel of the row at
/// `argb[at..at + width]` and its four neighbours, into `max_diffs[1..width - 1]`; `stride`
/// is the row above's and below's distance.
pub(crate) fn max_diffs_for_row(
    width: usize,
    stride: usize,
    argb: &[u32],
    at: usize,
    max_diffs: &mut [u8],
    subtract_green: bool,
) {
    if width <= 2 {
        return;
    }
    let argb = &argb[at - stride..];
    let (up, argb, down) = (&argb[..width], &argb[stride..], &argb[2 * stride..]);
    let fix = |p: u32| {
        if subtract_green {
            add_green_to_blue_and_red(p)
        } else {
            p
        }
    };
    let mut current = fix(argb[0]);
    let mut right = fix(argb[1]);
    for x in 1..width - 1 {
        let up = fix(up[x]);
        let down = fix(down[x]);
        let left = current;
        current = right;
        right = fix(argb[x + 1]);
        max_diffs[x] = max_diff_between_pixels(current, up)
            .max(max_diff_between_pixels(current, down))
            .max(max_diff_between_pixels(current, left))
            .max(max_diff_between_pixels(current, right));
    }
}

fn component(value: u8, predict: u8, boundary: u8, quantization: i32) -> u8 {
    let residual = (i32::from(value) - i32::from(predict)) & 0xff;
    let boundary_residual = (i32::from(boundary) - i32::from(predict)) & 0xff;
    let lower = residual & !(quantization - 1);
    let upper = lower + quantization;
    let bias = i32::from(((i32::from(boundary) - i32::from(value)) & 0xff) < boundary_residual);
    if residual - lower < upper - residual + bias {
        if residual > boundary_residual && lower <= boundary_residual {
            (lower + (quantization >> 1)) as u8
        } else {
            lower as u8
        }
    } else if residual <= boundary_residual && upper > boundary_residual {
        (lower + (quantization >> 1)) as u8
    } else {
        upper as u8
    }
}

fn diff(a: u8, b: u8) -> u8 {
    a.wrapping_sub(b)
}

/// `NearLossless` of predictor_enc.c: the residual of `value` against `predict`, each component
/// quantized to a step of at most `max_quantization`, halved while it reaches `max_diff` (the
/// pixel's largest neighbour difference), so that flat areas keep their steps small. An alpha of
/// 0 or 255 is kept exactly.
pub(crate) fn residual(
    value: u32,
    predict: u32,
    max_quantization: u32,
    max_diff: u8,
    subtract_green: bool,
) -> u32 {
    if max_diff <= 2 {
        return sub_pixels(value, predict);
    }
    let mut quantization = max_quantization as i32;
    while quantization >= i32::from(max_diff) {
        quantization >>= 1;
    }
    let (va, vr, vg, vb) = (
        (value >> 24) as u8,
        (value >> 16) as u8,
        (value >> 8) as u8,
        value as u8,
    );
    let (pa, pr, pg, pb) = (
        (predict >> 24) as u8,
        (predict >> 16) as u8,
        (predict >> 8) as u8,
        predict as u8,
    );
    let a = if va == 0 || va == 0xff {
        diff(va, pa)
    } else {
        component(va, pa, 0xff, quantization)
    };
    let g = component(vg, pg, 0xff, quantization);
    let (new_green, green_diff) = if subtract_green {
        let new_green = pg.wrapping_add(g);
        (new_green, diff(new_green, vg))
    } else {
        (0, 0)
    };
    let r = component(diff(vr, green_diff), pr, 0xff - new_green, quantization);
    let b = component(diff(vb, green_diff), pb, 0xff - new_green, quantization);
    u32::from(a) << 24 | u32::from(r) << 16 | u32::from(g) << 8 | u32::from(b)
}

/// `VP8LAddPixels`: `a + b` per component.
pub(crate) fn add_pixels(a: u32, b: u32) -> u32 {
    let alpha_and_green = (a & 0xff00ff00).wrapping_add(b & 0xff00ff00);
    let red_and_blue = (a & 0x00ff00ff).wrapping_add(b & 0x00ff00ff);
    (alpha_and_green & 0xff00ff00) | (red_and_blue & 0x00ff00ff)
}

/// Whether `pixel` is fully transparent, whose RGB a non-exact encode replaces.
pub(crate) fn transparent(pixel: u32) -> bool {
    pixel & MASK_ALPHA == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(argb: &[u32], width: usize, height: usize, quality: i32) -> Vec<u32> {
        let bits = limit_bits(quality);
        let mut out = argb.to_vec();
        if bits == 0 || (width < MIN_DIM && height < MIN_DIM) || height < 3 {
            return out;
        }
        for b in (1..=bits).rev() {
            let src = out.clone();
            let limit = 1i32 << b;
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let c = src[y * width + x];
                    let smooth = is_near(c, src[y * width + x - 1], limit)
                        && is_near(c, src[y * width + x + 1], limit)
                        && is_near(c, src[(y - 1) * width + x], limit)
                        && is_near(c, src[(y + 1) * width + x], limit);
                    out[y * width + x] = if smooth {
                        c
                    } else {
                        closest_discretized_argb(c, b)
                    };
                }
            }
        }
        out
    }

    #[test]
    fn rows_come_out_as_the_whole_picture_pass() {
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        for &(w, h) in &[(70usize, 5usize), (3, 64), (64, 3), (65, 9), (10, 10)] {
            let argb: Vec<u32> = (0..w * h)
                .map(|i| {
                    let smooth = 0xff000000 | ((i % w) as u32 * 2) << 8 | (i / w) as u32;
                    if next() % 5 == 0 {
                        next()
                    } else {
                        smooth
                    }
                })
                .collect();
            for q in [0, 19, 20, 55, 79, 80, 99, 100] {
                let mut ours = argb.clone();
                NearLossless::apply(&mut ours, w, h, q);
                assert_eq!(ours, reference(&argb, w, h, q), "{w}x{h} quality {q}");
            }
        }
    }

    #[test]
    fn quantization_steps() {
        assert_eq!(limit_bits(100), 0);
        assert_eq!(limit_bits(99), 1);
        assert_eq!(limit_bits(80), 1);
        assert_eq!(limit_bits(79), 2);
        assert_eq!(limit_bits(0), 5);
        assert_eq!(max_quantization(40), 8);
        assert_eq!(residual(0x80402010, 0x80402010, 8, 2, false), 0);
        assert_eq!(
            residual(0xff807060, 0xff807060, 8, 0, true),
            sub_pixels(0xff807060, 0xff807060)
        );
    }
}
