//! Spatial prediction and cross-colour transforms (libwebp's `enc/predictor_enc.c`).

use super::entropy::*;
use super::near_lossless::{self, add_pixels};

const HISTO_SIZE: usize = 4 * 256;
const SPATIAL_PREDICTOR_BIAS: i64 = 15 << LOG_2_PRECISION_BITS;
const PRED_LOW_EFFORT: usize = 11;
const MASK_ALPHA: u32 = 0xff000000;
const NUM_PRED_MODES: usize = 14;
const ARGB_BLACK: u32 = 0xff000000;
pub(crate) const MIN_TRANSFORM_BITS: u32 = 2;
pub(crate) const MAX_TRANSFORM_BITS: u32 = MIN_TRANSFORM_BITS + (1 << 3) - 1;

#[inline]
fn average2(a0: u32, a1: u32) -> u32 {
    (((a0 ^ a1) & 0xfefefefe) >> 1) + (a0 & a1)
}

#[inline]
fn clip255(a: u32) -> u32 {
    if a < 256 {
        a
    } else {
        !a >> 24
    }
}

#[inline]
fn add_subtract_component_full(a: i32, b: i32, c: i32) -> u32 {
    clip255((a + b - c) as u32)
}

#[inline]
fn clamped_add_subtract_full(c0: u32, c1: u32, c2: u32) -> u32 {
    let ch = |s: u32| {
        add_subtract_component_full(
            ((c0 >> s) & 0xff) as i32,
            ((c1 >> s) & 0xff) as i32,
            ((c2 >> s) & 0xff) as i32,
        )
    };
    (ch(24) << 24) | (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

#[inline]
fn add_subtract_component_half(a: i32, b: i32) -> u32 {
    clip255((a + (a - b) / 2) as u32)
}

#[inline]
fn clamped_add_subtract_half(c0: u32, c1: u32, c2: u32) -> u32 {
    let ave = average2(c0, c1);
    let ch =
        |s: u32| add_subtract_component_half(((ave >> s) & 0xff) as i32, ((c2 >> s) & 0xff) as i32);
    (ch(24) << 24) | (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

#[inline]
fn sub3(a: i32, b: i32, c: i32) -> i32 {
    let pb = b - c;
    let pa = a - c;
    pb.abs() - pa.abs()
}

#[inline]
fn select(a: u32, b: u32, c: u32) -> u32 {
    let ch = |s: u32| {
        sub3(
            ((a >> s) & 0xff) as i32,
            ((b >> s) & 0xff) as i32,
            ((c >> s) & 0xff) as i32,
        )
    };
    let pa_minus_pb = ch(24) + ch(16) + ch(8) + ch(0);
    if pa_minus_pb <= 0 {
        a
    } else {
        b
    }
}

/// `VP8LPredictors[M]` with `left` = `row[x - 1]` and `top[-1..=1]` from `upper[x - 1..=x + 1]`.
#[inline(always)]
pub(super) fn predict<const M: usize>(left: u32, tl: u32, t: u32, tr: u32) -> u32 {
    match M {
        0 => ARGB_BLACK,
        1 => left,
        2 => t,
        3 => tr,
        4 => tl,
        5 => average2(average2(left, tr), t),
        6 => average2(left, tl),
        7 => average2(left, t),
        8 => average2(tl, t),
        9 => average2(t, tr),
        10 => average2(average2(left, tl), average2(t, tr)),
        11 => select(t, left, tl),
        12 => clamped_add_subtract_full(left, t, tl),
        13 => clamped_add_subtract_half(left, t, tl),
        _ => ARGB_BLACK,
    }
}

/// Residuals of `current[x0..x0 + out.len()]` (x0 > 0) against mode `M`.
fn sub_batch<const M: usize>(current: &[u32], upper: &[u32], x0: usize, out: &mut [u32]) {
    let n = out.len();
    let cur = &current[x0 - 1..x0 + n];
    let up = &upper[x0 - 1..x0 + n + 1];
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let done = super::simd::sub_batch::<M>(cur, up, out);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let done = 0;
    for i in done..n {
        out[i] = sub_pixels(
            cur[i + 1],
            predict::<M>(cur[i], up[i], up[i + 1], up[i + 2]),
        );
    }
}

/// Non-exact residuals of `current[x0..x0 + out.len()]` (x0 > 0, y > 0) against mode `M`,
/// with the RGB of transparent pixels replaced by the prediction's.
fn sub_batch_inexact<const M: usize>(
    current: &mut [u32],
    upper: &[u32],
    x0: usize,
    out: &mut [u32],
) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let (cur, up) = (&current[x0 - 1 + i..x0 + n], &upper[x0 - 1 + i..x0 + n + 1]);
            i += super::simd::sub_batch_opaque::<M>(cur, up, &mut out[i..]);
            if i >= n {
                break;
            }
        }
        let x = x0 + i;
        let o = &mut out[i];
        i += 1;
        let predict_v = predict::<M>(current[x - 1], upper[x - 1], upper[x], upper[x + 1]);
        let mut residual = sub_pixels(current[x], predict_v);
        if current[x] & MASK_ALPHA == 0 {
            residual &= MASK_ALPHA;
            current[x] = predict_v & !MASK_ALPHA;
        }
        *o = residual;
    }
}

macro_rules! dispatch_mode {
    ($mode:expr, $f:ident, $($arg:expr),*) => {
        match $mode {
            0 => $f::<0>($($arg),*),
            1 => $f::<1>($($arg),*),
            2 => $f::<2>($($arg),*),
            3 => $f::<3>($($arg),*),
            4 => $f::<4>($($arg),*),
            5 => $f::<5>($($arg),*),
            6 => $f::<6>($($arg),*),
            7 => $f::<7>($($arg),*),
            8 => $f::<8>($($arg),*),
            9 => $f::<9>($($arg),*),
            10 => $f::<10>($($arg),*),
            11 => $f::<11>($($arg),*),
            12 => $f::<12>($($arg),*),
            13 => $f::<13>($($arg),*),
            _ => $f::<0>($($arg),*),
        }
    };
}

fn predict_mode(mode: usize, left: u32, tl: u32, t: u32, tr: u32) -> u32 {
    dispatch_mode!(mode, predict, left, tl, t, tr)
}

/// `PredictBatch`: residuals of `current[x_start..x_start + num_pixels]` into `out`.
fn predict_batch(
    mode: usize,
    mut x_start: usize,
    y: usize,
    num_pixels: usize,
    current: &[u32],
    upper: &[u32],
    out: &mut [u32],
) {
    let mut out = &mut out[..num_pixels];
    if x_start == 0 {
        out[0] = if y == 0 {
            sub_pixels(current[0], ARGB_BLACK)
        } else {
            sub_pixels(current[0], upper[0])
        };
        x_start += 1;
        out = &mut out[1..];
    }
    if y == 0 {
        sub_batch::<1>(current, upper, x_start, out);
    } else {
        dispatch_mode!(mode, sub_batch, current, upper, x_start, out);
    }
}

/// `GetResidual`: the residuals of `current_row[x_start..x_end]`, quantized within
/// `max_diffs` where `res` asks for near-lossless.
#[allow(clippy::too_many_arguments)]
fn get_residual(
    width: usize,
    height: usize,
    upper_row: &mut [u32],
    current_row: &mut [u32],
    max_diffs: &[u8],
    mode: usize,
    x_start: usize,
    x_end: usize,
    y: usize,
    res: Residuals,
    out: &mut [u32],
) {
    if res.exact {
        predict_batch(
            mode,
            x_start,
            y,
            x_end - x_start,
            current_row,
            upper_row,
            out,
        );
        return;
    }
    if res.max_quantization > 1 && mode != 0 && y != 0 && y != height - 1 {
        for x in x_start..x_end {
            let predict_v = if x == 0 {
                upper_row[0]
            } else {
                predict_mode(
                    mode,
                    current_row[x - 1],
                    upper_row[x - 1],
                    upper_row[x],
                    upper_row[x + 1],
                )
            };
            let mut residual = if x == 0 || x == width - 1 {
                sub_pixels(current_row[x], predict_v)
            } else {
                let r = near_lossless::residual(
                    current_row[x],
                    predict_v,
                    res.max_quantization,
                    max_diffs[x],
                    res.subtract_green,
                );
                current_row[x] = add_pixels(predict_v, r);
                r
            };
            if near_lossless::transparent(current_row[x]) {
                residual &= MASK_ALPHA;
                current_row[x] = predict_v & !MASK_ALPHA;
                if x == 0 {
                    upper_row[width] = current_row[0];
                }
            }
            out[x - x_start] = residual;
        }
        return;
    }
    let mut out = &mut out[..x_end - x_start];
    let mut x0 = x_start;
    if x0 == 0 {
        let predict_v = if y == 0 { ARGB_BLACK } else { upper_row[0] };
        let mut residual = sub_pixels(current_row[0], predict_v);
        if current_row[0] & MASK_ALPHA == 0 {
            residual &= MASK_ALPHA;
            current_row[0] = predict_v & !MASK_ALPHA;
            if y != 0 {
                upper_row[width] = current_row[0];
            }
        }
        out[0] = residual;
        out = &mut out[1..];
        x0 = 1;
    }
    if y == 0 {
        sub_batch_inexact::<1>(current_row, upper_row, x0, out);
    } else {
        dispatch_mode!(mode, sub_batch_inexact, current_row, upper_row, x0, out);
    }
}

/// The 15 decaying weights of `PredictionCostBias` from `exp_val`: each is the previous one
/// times 6/10, rounded.
const fn bias_weights(exp_val: u64) -> [u64; 16] {
    let mut w = [0u64; 16];
    w[1] = exp_val << LOG_2_PRECISION_BITS;
    let mut i = 2;
    while i < 16 {
        w[i] = (6 * w[i - 1] + 5) / 10;
        i += 1;
    }
    w
}

const SPATIAL_BIAS_WEIGHTS: [u64; 16] = bias_weights(94);
const CROSS_COLOR_BIAS_WEIGHTS: [u64; 16] = bias_weights(240);

/// `PredictionCostBias`, with its weights precomputed.
fn prediction_cost_bias(counts: &[u32], weight_0: u64, weights: &[u64; 16]) -> i64 {
    let mut bits = (weight_0 * u64::from(counts[0])) << LOG_2_PRECISION_BITS;
    for i in 1..16 {
        let c = u64::from(counts[i].wrapping_add(counts[256 - i]));
        bits = bits.wrapping_add(div_round(weights[i].wrapping_mul(c) as i64, 100) as u64);
    }
    -div_round(bits as i64, 10)
}

fn as256(s: &[u32]) -> &[u32; 256] {
    s[..256].try_into().unwrap()
}

fn prediction_cost_spatial_histogram(
    accumulated: &[SlogHisto; 4],
    tile: &[u32],
    mode: usize,
    left_mode: usize,
    above_mode: usize,
) -> i64 {
    let mut retval = 0i64;
    for i in 0..4 {
        retval += prediction_cost_bias(&tile[i * 256..], 1, &SPATIAL_BIAS_WEIGHTS);
        retval =
            retval.wrapping_add(accumulated[i].combined_entropy(as256(&tile[i * 256..])) as i64);
    }
    if mode == left_mode {
        retval -= SPATIAL_PREDICTOR_BIAS;
    }
    if mode == above_mode {
        retval -= SPATIAL_PREDICTOR_BIAS;
    }
    retval
}

/// `UpdateHisto` over a tile's residuals. A channel that is constant across the tile is
/// counted with one add; the others rotate between four histograms so that runs of one
/// value do not serialize on a single bin.
#[inline(always)]
fn update_histo(histo: &mut [u32; HISTO_SIZE], residuals: &[u32]) {
    let first = residuals[0];
    let diff = residuals.iter().fold(0, |acc, &r| acc | (r ^ first));
    for (c, plane) in histo.chunks_exact_mut(256).enumerate() {
        let shift = 24 - 8 * c;
        let bin = |r: u32| ((r >> shift) & 0xff) as usize;
        if (diff >> shift) & 0xff == 0 {
            plane[bin(first)] += residuals.len() as u32;
        } else {
            let mut more = [[0u32; 256]; 3];
            let quads = residuals.chunks_exact(4);
            for &r in quads.remainder() {
                plane[bin(r)] += 1;
            }
            for q in quads {
                plane[bin(q[0])] += 1;
                more[0][bin(q[1])] += 1;
                more[1][bin(q[2])] += 1;
                more[2][bin(q[3])] += 1;
            }
            for (i, a) in plane.iter_mut().enumerate() {
                *a += more[0][i] + more[1][i] + more[2][i];
            }
        }
    }
}

/// How residuals are taken: `exact` keeps transparent pixels' RGB, a `max_quantization` above
/// 1 quantizes them near-losslessly (after subtract-green when `subtract_green`).
#[derive(Clone, Copy)]
pub(crate) struct Residuals {
    pub(crate) exact: bool,
    pub(crate) max_quantization: u32,
    pub(crate) subtract_green: bool,
}

struct Scratch {
    rows: Vec<u32>,
    residuals: Vec<u32>,
    max_diffs: [Vec<u8>; 2],
    width: usize,
    swapped: bool,
}

impl Scratch {
    fn new(width: usize) -> Self {
        Self {
            rows: vec![0; 2 * (width + 1)],
            residuals: Vec::new(),
            max_diffs: [vec![0; width], vec![0; width]],
            width,
            swapped: false,
        }
    }
    fn swap(&mut self) {
        self.swapped = !self.swapped;
    }
    /// (upper, current)
    fn rows(&mut self) -> (&mut [u32], &mut [u32]) {
        let (a, b) = self.rows.split_at_mut(self.width + 1);
        if self.swapped {
            (b, a)
        } else {
            (a, b)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_residuals_for_tile(
    width: usize,
    height: usize,
    tile_x: usize,
    tile_y: usize,
    min_bits: u32,
    update_up_to_index: usize,
    all_argb: &mut [u32],
    scratch: &mut Scratch,
    argb: &[u32],
    res: Residuals,
) {
    let start_x = tile_x << min_bits;
    let start_y = tile_y << min_bits;
    let tile_size = 1 << min_bits;
    let max_y = tile_size.min(height - start_y);
    let max_x = tile_size.min(width - start_x);
    let have_left = usize::from(start_x > 0);
    let context_start_x = start_x - have_left;
    let context_width = max_x + have_left + usize::from(max_x < width - start_x);
    let mut residuals = core::mem::take(&mut scratch.residuals);
    residuals.resize(max_x * max_y, 0);
    let mut max_diffs = core::mem::take(&mut scratch.max_diffs);
    scratch.swapped = false;
    for mode in 0..NUM_PRED_MODES {
        if start_y > 0 {
            let n = max_x + have_left + 1;
            let src = (start_y - 1) * width + context_start_x;
            let (_, current) = scratch.rows();
            current[context_start_x..context_start_x + n].copy_from_slice(&argb[src..src + n]);
        }
        for relative_y in 0..max_y {
            let y = start_y + relative_y;
            scratch.swap();
            let n = max_x + have_left + usize::from(y + 1 < height);
            let src = y * width + context_start_x;
            let (upper, current) = scratch.rows();
            current[context_start_x..context_start_x + n].copy_from_slice(&argb[src..src + n]);
            if res.max_quantization > 1 && y >= 1 && y + 1 < height {
                near_lossless::max_diffs_for_row(
                    context_width,
                    width,
                    argb,
                    src,
                    &mut max_diffs[0][context_start_x..],
                    res.subtract_green,
                );
            }
            get_residual(
                width,
                height,
                upper,
                current,
                &max_diffs[0],
                mode,
                start_x,
                start_x + max_x,
                y,
                res,
                &mut residuals[relative_y * max_x..],
            );
        }
        for subsampling_index in 0..=update_up_to_index {
            let off = (subsampling_index * NUM_PRED_MODES + mode) * HISTO_SIZE;
            let histo: &mut [u32; HISTO_SIZE] =
                (&mut all_argb[off..off + HISTO_SIZE]).try_into().unwrap();
            crate::wide::wide(|| update_histo(histo, &residuals));
        }
    }
    scratch.residuals = residuals;
    scratch.max_diffs = max_diffs;
}

fn copy_image_with_prediction(
    width: usize,
    height: usize,
    bits: u32,
    modes: &[u32],
    scratch: &mut Scratch,
    argb: &mut [u32],
    low_effort: bool,
    res: Residuals,
) {
    let tiles_per_row = sub_sample_size(width as u32, bits) as usize;
    scratch.swapped = false;
    let [mut current_max_diffs, mut lower_max_diffs] = core::mem::take(&mut scratch.max_diffs);
    for y in 0..height {
        scratch.swap();
        let n = width + usize::from(y + 1 < height);
        let (upper, current) = scratch.rows();
        current[..n].copy_from_slice(&argb[y * width..y * width + n]);
        if low_effort {
            let out = &mut argb[y * width..(y + 1) * width];
            predict_batch(PRED_LOW_EFFORT, 0, y, width, current, upper, out);
        } else {
            if res.max_quantization > 1 {
                core::mem::swap(&mut current_max_diffs, &mut lower_max_diffs);
                if y + 2 < height {
                    near_lossless::max_diffs_for_row(
                        width,
                        width,
                        argb,
                        (y + 1) * width,
                        &mut lower_max_diffs,
                        res.subtract_green,
                    );
                }
            }
            let out = &mut argb[y * width..(y + 1) * width];
            let mut x = 0;
            while x < width {
                let mode =
                    ((modes[(y >> bits) * tiles_per_row + (x >> bits)] >> 8) & 0xff) as usize;
                let x_end = (x + (1 << bits)).min(width);
                get_residual(
                    width,
                    height,
                    upper,
                    current,
                    &current_max_diffs,
                    mode,
                    x,
                    x_end,
                    y,
                    res,
                    &mut out[x..x_end],
                );
                x = x_end;
            }
        }
    }
    scratch.max_diffs = [current_max_diffs, lower_max_diffs];
}

/// `VP8LOptimizeSampling`: subsamples `image` further while its blocks are uniform.
pub(crate) fn optimize_sampling(
    image: &mut [u32],
    full_width: u32,
    full_height: u32,
    bits: u32,
    max_bits: u32,
) -> u32 {
    let mut width = sub_sample_size(full_width, bits) as usize;
    let mut height = sub_sample_size(full_height, bits) as usize;
    let mut best_bits = bits;
    while best_bits < max_bits {
        let new_square_size = 1 << (best_bits + 1 - bits);
        let square_size = 1 << (best_bits - bits);
        let mut is_good = true;
        let mut y = 0;
        while y + square_size < height {
            if image[y * width..(y + 1) * width]
                != image[(y + square_size) * width..(y + square_size + 1) * width]
            {
                is_good = false;
                break;
            }
            y += new_square_size;
        }
        if is_good {
            best_bits += 1;
        } else {
            break;
        }
    }
    if best_bits == bits {
        return bits;
    }
    while best_bits > bits {
        let square_size = 1 << (best_bits - bits);
        let mut is_good = true;
        'outer: for y in 0..height {
            let mut x = 0;
            while x < width {
                for i in x + 1..(x + square_size).min(width) {
                    if image[y * width + i] != image[y * width + x] {
                        is_good = false;
                        break 'outer;
                    }
                }
                x += square_size;
            }
        }
        if is_good {
            break;
        }
        best_bits -= 1;
    }
    if best_bits == bits {
        return bits;
    }
    let old_width = width;
    let square_size = 1 << (best_bits - bits);
    width = sub_sample_size(full_width, best_bits) as usize;
    height = sub_sample_size(full_height, best_bits) as usize;
    for y in 0..height {
        for x in 0..width {
            image[y * width + x] = image[square_size * (y * old_width + x)];
        }
    }
    best_bits
}

/// Returns the best bits and the modes at that sampling.
#[allow(clippy::too_many_arguments)]
fn get_best_predictors_and_sub_sampling(
    width: usize,
    height: usize,
    min_bits: u32,
    max_bits: u32,
    scratch: &mut Scratch,
    argb: &[u32],
    res: Residuals,
    all_modes: &mut [Vec<u32>],
) -> u32 {
    let tiles_per_row = sub_sample_size(width as u32, min_bits) as usize;
    let tiles_per_col = sub_sample_size(height as u32, min_bits) as usize;
    let max_subsampling_index = (max_bits - min_bits) as usize;
    let mut all_argb = vec![0u32; (max_subsampling_index + 1) * NUM_PRED_MODES * HISTO_SIZE];
    let mut all_accumulated: Vec<[SlogHisto; 4]> = (0..=max_subsampling_index)
        .map(|_| Default::default())
        .collect();
    let mut all_pred_histos = vec![0u32; (max_subsampling_index + 1) * NUM_PRED_MODES];
    let max_tile_size = 1usize << max_subsampling_index;
    let update_up_to_index = (4.min(max_bits).max(min_bits) - min_bits) as usize;
    let (mut local_tile_x, mut local_tile_y) = (0usize, 0usize);
    let (mut max_tile_x, mut max_tile_y) = (0usize, 0usize);
    let (mut tile_x, mut tile_y) = (0usize, 0usize);

    while tile_y < tiles_per_col {
        compute_residuals_for_tile(
            width,
            height,
            tile_x,
            tile_y,
            min_bits,
            update_up_to_index,
            &mut all_argb,
            scratch,
            argb,
            res,
        );
        let mut subsampling_index = 0usize;
        loop {
            let super_tile_x = tile_x >> subsampling_index;
            let super_tile_y = tile_y >> subsampling_index;
            let super_tiles_per_row =
                sub_sample_size(width as u32, min_bits + subsampling_index as u32) as usize;
            get_best_predictor_for_tile(
                &all_argb,
                subsampling_index,
                super_tile_x,
                super_tile_y,
                super_tiles_per_row,
                &mut all_accumulated[subsampling_index],
                all_modes,
                &mut all_pred_histos,
            );
            if subsampling_index == max_subsampling_index {
                break;
            }
            subsampling_index += 1;
            if subsampling_index > update_up_to_index && subsampling_index <= max_subsampling_index
            {
                let n = HISTO_SIZE * NUM_PRED_MODES;
                let (lo, hi) = all_argb.split_at_mut(subsampling_index * n);
                for (o, &a) in hi[..n].iter_mut().zip(&lo[(subsampling_index - 1) * n..]) {
                    *o = o.wrapping_add(a);
                }
            }
            let s = 1usize << subsampling_index;
            if !((tile_x == tiles_per_row - 1 || (local_tile_x + 1) % s == 0)
                && (tile_y == tiles_per_col - 1 || (local_tile_y + 1) % s == 0))
            {
                subsampling_index -= 1;
                break;
            }
        }
        all_argb[..HISTO_SIZE * NUM_PRED_MODES * (subsampling_index + 1)].fill(0);
        if subsampling_index == max_subsampling_index {
            if tile_x == tiles_per_row - 1 {
                max_tile_x = 0;
                max_tile_y += 1;
            } else {
                max_tile_x += 1;
            }
            local_tile_x = 0;
            local_tile_y = 0;
        } else {
            let mut coord_x = local_tile_x >> subsampling_index;
            let mut coord_y = local_tile_y >> subsampling_index;
            if tile_x == tiles_per_row - 1 && coord_x % 2 == 0 {
                coord_y += 1;
            } else if coord_x % 2 == 0 {
                coord_x += 1;
            } else {
                coord_y += 1;
                coord_x -= 1;
            }
            local_tile_x = coord_x << subsampling_index;
            local_tile_y = coord_y << subsampling_index;
        }
        tile_x = max_tile_x * max_tile_size + local_tile_x;
        tile_y = max_tile_y * max_tile_size + local_tile_y;
    }

    let mut best_cost = i64::MAX;
    let mut best = 0;
    for subsampling_index in 0..=max_subsampling_index {
        let accumulated = &all_accumulated[subsampling_index];
        let mut cost = shannon_entropy(
            &all_pred_histos[subsampling_index * NUM_PRED_MODES..][..NUM_PRED_MODES],
        ) as i64;
        for plane in accumulated {
            cost = cost.wrapping_add(shannon_entropy(plane.histogram()) as i64);
        }
        if cost < best_cost {
            best_cost = cost;
            best = subsampling_index;
        }
    }
    let best_bits = min_bits + best as u32;
    let best_bits = optimize_sampling(
        &mut all_modes[best],
        width as u32,
        height as u32,
        best_bits,
        MAX_TRANSFORM_BITS,
    );
    all_modes.swap(0, best);
    best_bits
}

#[allow(clippy::too_many_arguments)]
fn get_best_predictor_for_tile(
    all_argb: &[u32],
    subsampling_index: usize,
    tile_x: usize,
    tile_y: usize,
    tiles_per_row: usize,
    accumulated: &mut [SlogHisto; 4],
    all_modes: &mut [Vec<u32>],
    all_pred_histos: &mut [u32],
) {
    let modes = &mut all_modes[subsampling_index];
    let left_mode = if tile_x > 0 {
        ((modes[tile_y * tiles_per_row + tile_x - 1] >> 8) & 0xff) as usize
    } else {
        0xff
    };
    let above_mode = if tile_y > 0 {
        ((modes[(tile_y - 1) * tiles_per_row + tile_x] >> 8) & 0xff) as usize
    } else {
        0xff
    };
    let mut best_diff = i64::MAX;
    let mut best_mode = 0usize;
    for mode in 0..NUM_PRED_MODES {
        let off = (subsampling_index * NUM_PRED_MODES + mode) * HISTO_SIZE;
        let histo = &all_argb[off..off + HISTO_SIZE];
        let cur_diff =
            prediction_cost_spatial_histogram(accumulated, histo, mode, left_mode, above_mode);
        if cur_diff < best_diff {
            best_diff = cur_diff;
            best_mode = mode;
        }
    }
    let off = (subsampling_index * NUM_PRED_MODES + best_mode) * HISTO_SIZE;
    let (planes, _) = all_argb[off..off + HISTO_SIZE].as_chunks::<256>();
    for (h, plane) in accumulated.iter_mut().zip(planes) {
        h.add(plane);
    }
    modes[tile_y * tiles_per_row + tile_x] = ARGB_BLACK | ((best_mode as u32) << 8);
    all_pred_histos[subsampling_index * NUM_PRED_MODES + best_mode] += 1;
}

/// `VP8LResidualImage`: chooses the predictors into `image` and replaces `argb` by the
/// residuals. Returns the predictor bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_image(
    width: usize,
    height: usize,
    min_bits: u32,
    max_bits: u32,
    low_effort: bool,
    argb: &mut [u32],
    image: &mut [u32],
    res: Residuals,
) -> u32 {
    let mut scratch = Scratch::new(width);
    let best_bits;
    if low_effort {
        let n = (sub_sample_size(width as u32, max_bits) * sub_sample_size(height as u32, max_bits))
            as usize;
        image[..n].fill(ARGB_BLACK | ((PRED_LOW_EFFORT as u32) << 8));
        best_bits = max_bits;
    } else {
        let mut all_modes: Vec<Vec<u32>> = (min_bits..=max_bits)
            .map(|bits| {
                vec![
                    0u32;
                    (sub_sample_size(width as u32, bits) * sub_sample_size(height as u32, bits))
                        as usize
                ]
            })
            .collect();
        best_bits = get_best_predictors_and_sub_sampling(
            width,
            height,
            min_bits,
            max_bits,
            &mut scratch,
            argb,
            res,
            &mut all_modes,
        );
        let n = (sub_sample_size(width as u32, best_bits)
            * sub_sample_size(height as u32, best_bits)) as usize;
        image[..n].copy_from_slice(&all_modes[0][..n]);
    }
    copy_image_with_prediction(
        width,
        height,
        best_bits,
        image,
        &mut scratch,
        argb,
        low_effort,
        res,
    );
    best_bits
}

#[derive(Clone, Copy, Default)]
struct Multipliers {
    green_to_red: u8,
    green_to_blue: u8,
    red_to_blue: u8,
}

impl Multipliers {
    fn from_code(c: u32) -> Self {
        Self {
            green_to_red: c as u8,
            green_to_blue: (c >> 8) as u8,
            red_to_blue: (c >> 16) as u8,
        }
    }
    fn to_code(self) -> u32 {
        0xff000000
            | (u32::from(self.red_to_blue) << 16)
            | (u32::from(self.green_to_blue) << 8)
            | u32::from(self.green_to_red)
    }
}

#[inline]
fn color_transform_delta(color_pred: i8, color: i8) -> i32 {
    (i32::from(color_pred) * i32::from(color)) >> 5
}

#[inline]
#[cfg(test)]
fn transform_color_red(green_to_red: u8, argb: u32) -> u8 {
    let green = (argb >> 8) as u8 as i8;
    let new_red = (argb >> 16) as i32 - color_transform_delta(green_to_red as i8, green);
    (new_red & 0xff) as u8
}

#[inline]
#[cfg(test)]
fn transform_color_blue(green_to_blue: u8, red_to_blue: u8, argb: u32) -> u8 {
    let green = (argb >> 8) as u8 as i8;
    let red = (argb >> 16) as u8 as i8;
    let mut new_blue = (argb & 0xff) as i32;
    new_blue -= color_transform_delta(green_to_blue as i8, green);
    new_blue -= color_transform_delta(red_to_blue as i8, red);
    (new_blue & 0xff) as u8
}

#[inline(always)]
fn transform_color(m: Multipliers, data: &mut [u32]) {
    for p in data {
        let argb = *p;
        let green = (argb >> 8) as u8 as i8;
        let red = (argb >> 16) as u8 as i8;
        let mut new_red = i32::from(red as u8);
        let mut new_blue = (argb & 0xff) as i32;
        new_red -= color_transform_delta(m.green_to_red as i8, green);
        new_red &= 0xff;
        new_blue -= color_transform_delta(m.green_to_blue as i8, green);
        new_blue -= color_transform_delta(m.red_to_blue as i8, red);
        new_blue &= 0xff;
        *p = (argb & 0xff00ff00) | ((new_red as u32) << 16) | new_blue as u32;
    }
}

fn prediction_cost_cross_color(accumulated: &SlogHisto, counts: &[u32; 256]) -> i64 {
    (accumulated.combined_entropy(counts) as i64).wrapping_add(prediction_cost_bias(
        counts,
        3,
        &CROSS_COLOR_BIAS_WEIGHTS,
    ))
}

/// The tile's green, red and blue channels, one byte per pixel.
#[derive(Default)]
struct Tile {
    green: Vec<i8>,
    red: Vec<i8>,
    blue: Vec<u8>,
    transformed: Vec<u8>,
}

impl Tile {
    fn load(&mut self, argb: &[u32], stride: usize, width: usize, height: usize) {
        self.green.clear();
        self.red.clear();
        self.blue.clear();
        for y in 0..height {
            let row = &argb[y * stride..y * stride + width];
            self.green.extend(row.iter().map(|&p| (p >> 8) as u8 as i8));
            self.red.extend(row.iter().map(|&p| (p >> 16) as u8 as i8));
            self.blue.extend(row.iter().map(|&p| p as u8));
        }
        self.transformed.resize(self.green.len(), 0);
    }

    /// Histogram of `transformed`, counted in four interleaved sub-histograms so that
    /// runs of one value do not serialize on a single bin.
    #[inline(always)]
    fn histogram(&self) -> [u32; 256] {
        let mut h = [[0u32; 256]; 4];
        let chunks = self.transformed.chunks_exact(4);
        let rest = chunks.remainder();
        for c in chunks {
            h[0][c[0] as usize] += 1;
            h[1][c[1] as usize] += 1;
            h[2][c[2] as usize] += 1;
            h[3][c[3] as usize] += 1;
        }
        for &v in rest {
            h[0][v as usize] += 1;
        }
        let mut out = [0u32; 256];
        for i in 0..256 {
            out[i] = h[0][i] + h[1][i] + h[2][i] + h[3][i];
        }
        out
    }

    fn red_histogram(&mut self, green_to_red: u8) -> [u32; 256] {
        let m = i16::from(green_to_red as i8);
        crate::wide::wide(|| {
            for ((o, &g), &r) in self.transformed.iter_mut().zip(&self.green).zip(&self.red) {
                *o = (r as u8).wrapping_sub(((m * i16::from(g)) >> 5) as u8);
            }
            self.histogram()
        })
    }

    fn blue_histogram(&mut self, green_to_blue: u8, red_to_blue: u8) -> [u32; 256] {
        let (mg, mr) = (i16::from(green_to_blue as i8), i16::from(red_to_blue as i8));
        crate::wide::wide(|| {
            for (((o, &g), &r), &b) in self
                .transformed
                .iter_mut()
                .zip(&self.green)
                .zip(&self.red)
                .zip(&self.blue)
            {
                *o = b
                    .wrapping_sub(((mg * i16::from(g)) >> 5) as u8)
                    .wrapping_sub(((mr * i16::from(r)) >> 5) as u8);
            }
            self.histogram()
        })
    }
}

const BIAS3: i64 = 3 << LOG_2_PRECISION_BITS;

fn cost_cross_color_red(
    tile: &mut Tile,
    prev_x: Multipliers,
    prev_y: Multipliers,
    green_to_red: i32,
    acc: &SlogHisto,
) -> i64 {
    let histo = tile.red_histogram(green_to_red as u8);
    let mut cur_diff = prediction_cost_cross_color(acc, &histo);
    if green_to_red as u8 == prev_x.green_to_red {
        cur_diff -= BIAS3;
    }
    if green_to_red as u8 == prev_y.green_to_red {
        cur_diff -= BIAS3;
    }
    if green_to_red == 0 {
        cur_diff -= BIAS3;
    }
    cur_diff
}

fn best_green_to_red(
    tile: &mut Tile,
    prev_x: Multipliers,
    prev_y: Multipliers,
    quality: i32,
    acc: &SlogHisto,
    best_tx: &mut Multipliers,
) {
    let max_iters = 4 + ((7 * quality) >> 8);
    let mut green_to_red_best = 0i32;
    let mut best_diff = cost_cross_color_red(tile, prev_x, prev_y, green_to_red_best, acc);
    for iter in 0..max_iters {
        let delta = 32 >> iter;
        let mut offset = -delta;
        while offset <= delta {
            let cur = offset + green_to_red_best;
            let cur_diff = cost_cross_color_red(tile, prev_x, prev_y, cur, acc);
            if cur_diff < best_diff {
                best_diff = cur_diff;
                green_to_red_best = cur;
            }
            offset += 2 * delta;
        }
    }
    best_tx.green_to_red = (green_to_red_best & 0xff) as u8;
}

fn cost_cross_color_blue(
    tile: &mut Tile,
    prev_x: Multipliers,
    prev_y: Multipliers,
    green_to_blue: i32,
    red_to_blue: i32,
    acc: &SlogHisto,
) -> i64 {
    let histo = tile.blue_histogram(green_to_blue as u8, red_to_blue as u8);
    let mut cur_diff = prediction_cost_cross_color(acc, &histo);
    if green_to_blue as u8 == prev_x.green_to_blue {
        cur_diff -= BIAS3;
    }
    if green_to_blue as u8 == prev_y.green_to_blue {
        cur_diff -= BIAS3;
    }
    if red_to_blue as u8 == prev_x.red_to_blue {
        cur_diff -= BIAS3;
    }
    if red_to_blue as u8 == prev_y.red_to_blue {
        cur_diff -= BIAS3;
    }
    if green_to_blue == 0 {
        cur_diff -= BIAS3;
    }
    if red_to_blue == 0 {
        cur_diff -= BIAS3;
    }
    cur_diff
}

fn best_green_red_to_blue(
    tile: &mut Tile,
    prev_x: Multipliers,
    prev_y: Multipliers,
    quality: i32,
    acc: &SlogHisto,
    best_tx: &mut Multipliers,
) {
    const OFFSET: [[i32; 2]; 8] = [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ];
    const DELTA_LUT: [i32; 7] = [16, 16, 8, 4, 2, 2, 2];
    let iters = if quality < 25 {
        1
    } else if quality > 50 {
        7
    } else {
        4
    };
    // The search revisits points; their cost depends only on (gb, rb).
    let mut seen = [(0i32, 0i32, 0i64); 1 + 7 * 8];
    let mut num_seen = 0;
    let mut cost = |tile: &mut Tile, gb: i32, rb: i32| {
        if let Some(&(_, _, c)) = seen[..num_seen]
            .iter()
            .find(|&&(g, r, _)| g == gb && r == rb)
        {
            return c;
        }
        let c = cost_cross_color_blue(tile, prev_x, prev_y, gb, rb, acc);
        seen[num_seen] = (gb, rb, c);
        num_seen += 1;
        c
    };
    let (mut gb_best, mut rb_best) = (0i32, 0i32);
    let mut best_diff = cost(tile, gb_best, rb_best);
    for iter in 0..iters {
        let delta = DELTA_LUT[iter];
        for axis in 0..8 {
            let gb = OFFSET[axis][0] * delta + gb_best;
            let rb = OFFSET[axis][1] * delta + rb_best;
            let cur_diff = cost(tile, gb, rb);
            if cur_diff < best_diff {
                best_diff = cur_diff;
                gb_best = gb;
                rb_best = rb;
            }
            if quality < 25 && iter == 4 {
                break;
            }
        }
        if delta == 2 && gb_best == 0 && rb_best == 0 {
            break;
        }
    }
    best_tx.green_to_blue = (gb_best & 0xff) as u8;
    best_tx.red_to_blue = (rb_best & 0xff) as u8;
}

/// `VP8LColorSpaceTransform`: chooses the cross-colour multipliers into `image` and
/// applies them to `argb`. Returns the transform bits.
pub(crate) fn color_space_transform(
    width: usize,
    height: usize,
    bits: u32,
    quality: i32,
    argb: &mut [u32],
    image: &mut [u32],
) -> u32 {
    let max_tile_size = 1usize << bits;
    let tile_xsize = sub_sample_size(width as u32, bits) as usize;
    let tile_ysize = sub_sample_size(height as u32, bits) as usize;
    let mut acc_red = [0u32; 256];
    let mut acc_blue = [0u32; 256];
    let (mut slog_red, mut slog_blue) = (SlogHisto::default(), SlogHisto::default());
    let mut prev_x = Multipliers::default();
    let mut prev_y = Multipliers::default();
    let mut tile = Tile::default();
    for tile_y in 0..tile_ysize {
        for tile_x in 0..tile_xsize {
            let tile_x_offset = tile_x * max_tile_size;
            let tile_y_offset = tile_y * max_tile_size;
            let all_x_max = (tile_x_offset + max_tile_size).min(width);
            let all_y_max = (tile_y_offset + max_tile_size).min(height);
            let offset = tile_y * tile_xsize + tile_x;
            if tile_y != 0 {
                prev_y = Multipliers::from_code(image[offset - tile_xsize]);
            }
            let mut best_tx = Multipliers::default();
            tile.load(
                &argb[tile_y_offset * width + tile_x_offset..],
                width,
                all_x_max - tile_x_offset,
                all_y_max - tile_y_offset,
            );
            slog_red.update(&acc_red);
            slog_blue.update(&acc_blue);
            best_green_to_red(&mut tile, prev_x, prev_y, quality, &slog_red, &mut best_tx);
            best_green_red_to_blue(&mut tile, prev_x, prev_y, quality, &slog_blue, &mut best_tx);
            prev_x = best_tx;
            image[offset] = prev_x.to_code();
            crate::wide::wide(|| {
                for y in tile_y_offset..all_y_max {
                    transform_color(
                        prev_x,
                        &mut argb[y * width + tile_x_offset..y * width + all_x_max],
                    );
                }
            });
            for y in tile_y_offset..all_y_max {
                let mut ix = y * width + tile_x_offset;
                let ix_end = ix + all_x_max - tile_x_offset;
                while ix < ix_end {
                    let pix = argb[ix];
                    let skip = (ix >= 2 && pix == argb[ix - 2] && pix == argb[ix - 1])
                        || (ix >= width + 2
                            && argb[ix - 2] == argb[ix - width - 2]
                            && argb[ix - 1] == argb[ix - width - 1]
                            && pix == argb[ix - width]);
                    if !skip {
                        acc_red[((pix >> 16) & 0xff) as usize] += 1;
                        acc_blue[(pix & 0xff) as usize] += 1;
                    }
                    ix += 1;
                }
            }
        }
    }
    optimize_sampling(image, width as u32, height as u32, bits, MAX_TRANSFORM_BITS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) {
        for i in 0..out.len() {
            out[i] = sub_pixels(
                cur[i + 1],
                predict::<M>(cur[i], up[i], up[i + 1], up[i + 2]),
            );
        }
    }

    fn check<const M: usize>(cur: &[u32], up: &[u32]) {
        let n = cur.len() - 1;
        let (mut a, mut b) = (vec![0; n], vec![0; n]);
        scalar::<M>(cur, up, &mut a);
        sub_batch::<M>(cur, up, 1, &mut b);
        assert_eq!(a, b, "mode {M}");
        let mut c = vec![0; n];
        let mut cur_mut = cur.to_vec();
        let mut cur_ref = cur.to_vec();
        sub_batch_inexact::<M>(&mut cur_mut, up, 1, &mut c);
        for i in 0..n {
            let x = i + 1;
            let p = predict::<M>(cur_ref[x - 1], up[x - 1], up[x], up[x + 1]);
            let mut r = sub_pixels(cur_ref[x], p);
            if cur_ref[x] >> 24 == 0 {
                r &= MASK_ALPHA;
                cur_ref[x] = p & !MASK_ALPHA;
            }
            assert_eq!(c[i], r, "inexact mode {M} at {i}");
        }
        assert_eq!(cur_mut, cur_ref);
    }

    #[test]
    fn predictor_batches_match_scalar() {
        let mut seed = 0x9e37_79b9u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for n in [1usize, 3, 4, 5, 8, 31, 64] {
            let cur: Vec<u32> = (0..=n)
                .map(|i| {
                    if i % 5 == 3 {
                        rnd() & 0x00ffffff
                    } else {
                        rnd()
                    }
                })
                .collect();
            let up: Vec<u32> = (0..n + 2).map(|_| rnd()).collect();
            check::<0>(&cur, &up);
            check::<1>(&cur, &up);
            check::<2>(&cur, &up);
            check::<3>(&cur, &up);
            check::<4>(&cur, &up);
            check::<5>(&cur, &up);
            check::<6>(&cur, &up);
            check::<7>(&cur, &up);
            check::<8>(&cur, &up);
            check::<9>(&cur, &up);
            check::<10>(&cur, &up);
            check::<11>(&cur, &up);
            check::<12>(&cur, &up);
            check::<13>(&cur, &up);
        }
    }

    #[test]
    fn wide_transforms_match_scalar() {
        let mut seed = 0x3c6e_f372u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for n in [1usize, 7, 31, 32, 33, 100, 1000] {
            let argb: Vec<u32> = (0..n)
                .map(|i| if i % 5 < 2 { 0xff10_2030 } else { rnd() })
                .collect();
            let m = Multipliers {
                green_to_red: rnd() as u8,
                green_to_blue: rnd() as u8,
                red_to_blue: rnd() as u8,
            };
            let mut got = argb.clone();
            crate::wide::wide(|| transform_color(m, &mut got));
            let want: Vec<u32> = argb
                .iter()
                .map(|&p| {
                    (p & 0xff00_ff00)
                        | (u32::from(transform_color_red(m.green_to_red, p)) << 16)
                        | u32::from(transform_color_blue(m.green_to_blue, m.red_to_blue, p))
                })
                .collect();
            assert_eq!(got, want, "transform_color {n}");
            let mut wide = [0u32; HISTO_SIZE];
            let mut narrow = [0u32; HISTO_SIZE];
            crate::wide::wide(|| update_histo(&mut wide, &argb));
            update_histo(&mut narrow, &argb);
            assert_eq!(wide, narrow, "update_histo {n}");
        }
    }

    #[test]
    fn prediction_cost_bias_matches_libwebp() {
        fn reference(counts: &[u32], weight_0: u64, mut exp_val: u64) -> i64 {
            let mut bits = (weight_0 * u64::from(counts[0])) << LOG_2_PRECISION_BITS;
            exp_val <<= LOG_2_PRECISION_BITS;
            for i in 1..16 {
                bits = bits.wrapping_add(div_round(
                    exp_val.wrapping_mul(u64::from(counts[i].wrapping_add(counts[256 - i]))) as i64,
                    100,
                ) as u64);
                exp_val = div_round((6 * exp_val) as i64, 10) as u64;
            }
            -div_round(bits as i64, 10)
        }
        let mut seed = 0xa54f_f53au32;
        for round in 0..2000 {
            let counts: Vec<u32> = (0..256)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed >> (seed % 32).max(round % 32)
                })
                .collect();
            assert_eq!(
                prediction_cost_bias(&counts, 1, &SPATIAL_BIAS_WEIGHTS),
                reference(&counts, 1, 94)
            );
            assert_eq!(
                prediction_cost_bias(&counts, 3, &CROSS_COLOR_BIAS_WEIGHTS),
                reference(&counts, 3, 240)
            );
        }
    }

    #[test]
    fn cross_color_histograms_match_scalar() {
        let mut seed = 0x1234_5678u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let (stride, width, height) = (40, 37, 9);
        let argb: Vec<u32> = (0..stride * height)
            .map(|i| if i % 7 < 3 { 0xff80_7f81 } else { rnd() })
            .collect();
        let mut tile = Tile::default();
        tile.load(&argb, stride, width, height);
        let pixels =
            || (0..height).flat_map(|y| argb[y * stride..y * stride + width].iter().copied());
        for m in [0u8, 1, 31, 32, 127, 128, 200, 255] {
            let mut want = [0u32; 256];
            pixels().for_each(|p| want[transform_color_red(m, p) as usize] += 1);
            assert_eq!(tile.red_histogram(m), want, "red {m}");
            for n in [0u8, 3, 128, 255] {
                let mut want = [0u32; 256];
                pixels().for_each(|p| want[transform_color_blue(m, n, p) as usize] += 1);
                let got = tile.blue_histogram(m, n);
                assert_eq!(got, want, "blue {m} {n}");
            }
        }
    }
}
