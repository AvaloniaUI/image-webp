//! `ALPH` chunk payloads (libwebp's `enc/alpha_enc.c`, `dsp/filters.c` and
//! `utils/filters_utils.c`) with the default `alpha_quality = 100` and
//! `alpha_filtering = 1` (fast filter choice).

use super::encode_alpha_plane;

const FILTER_NONE: usize = 0;
const FILTER_HORIZONTAL: usize = 1;
const FILTER_VERTICAL: usize = 2;
const FILTER_GRADIENT: usize = 3;
const FILTER_LAST: usize = 4;

#[inline]
fn gradient_predictor(a: u8, b: u8, c: u8) -> u8 {
    let g = i32::from(a) + i32::from(b) - i32::from(c);
    if g & !0xff == 0 {
        g as u8
    } else if g < 0 {
        0
    } else {
        255
    }
}

fn predict_line(src: &[u8], pred: &[u8], dst: &mut [u8]) {
    for ((d, &s), &p) in dst.iter_mut().zip(src).zip(pred) {
        *d = s.wrapping_sub(p);
    }
}

/// libwebp's `WebPFilters[filter]` over a `width`-wide plane.
fn apply_filter(filter: usize, data: &[u8], width: usize, height: usize, out: &mut [u8]) {
    out[0] = data[0];
    predict_line(&data[1..width], &data[..width - 1], &mut out[1..width]);
    for row in 1..height {
        let (cur, prev) = (
            &data[row * width..(row + 1) * width],
            &data[(row - 1) * width..row * width],
        );
        let o = &mut out[row * width..(row + 1) * width];
        match filter {
            FILTER_HORIZONTAL => {
                o[0] = cur[0].wrapping_sub(prev[0]);
                predict_line(&cur[1..], &cur[..width - 1], &mut o[1..]);
            }
            FILTER_VERTICAL => predict_line(cur, prev, o),
            _ => {
                o[0] = cur[0].wrapping_sub(prev[0]);
                for w in 1..width {
                    o[w] =
                        cur[w].wrapping_sub(gradient_predictor(cur[w - 1], prev[w], prev[w - 1]));
                }
            }
        }
    }
}

/// `WebPEstimateBestFilter`.
fn estimate_best_filter(data: &[u8], width: usize, height: usize) -> usize {
    const SMAX: usize = 16;
    let sdiff = |a: i32, b: i32| ((a - b).abs() >> 4) as usize;
    let mut bins = [[0u8; SMAX]; FILTER_LAST];
    let mut j = 2;
    while j + 1 < height {
        let p = &data[j * width..];
        let mut mean = i32::from(p[0]);
        let mut i = 2;
        while i + 1 < width {
            let pi = i32::from(p[i]);
            let up = i32::from(data[j * width + i - width]);
            let grad = gradient_predictor(
                p[i - 1],
                data[j * width + i - width],
                data[j * width + i - width - 1],
            );
            bins[FILTER_NONE][sdiff(pi, mean)] = 1;
            bins[FILTER_HORIZONTAL][sdiff(pi, i32::from(p[i - 1]))] = 1;
            bins[FILTER_VERTICAL][sdiff(pi, up)] = 1;
            bins[FILTER_GRADIENT][sdiff(pi, i32::from(grad))] = 1;
            mean = (3 * mean + pi + 2) >> 2;
            i += 2;
        }
        j += 2;
    }
    let mut best_filter = FILTER_NONE;
    let mut best_score = i32::MAX;
    for (filter, b) in bins.iter().enumerate() {
        let score: i32 = b
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0)
            .map(|(i, _)| i as i32)
            .sum();
        if score < best_score {
            best_score = score;
            best_filter = filter;
        }
    }
    best_filter
}

fn get_filter_map(alpha: &[u8], width: usize, height: usize, effort_level: i32) -> u32 {
    let try_filter_none = effort_level > 3;
    let mut seen = [false; 256];
    for &a in alpha {
        seen[a as usize] = true;
    }
    let num_colors = seen.iter().filter(|&&s| s).count();
    let filter = if num_colors <= 16 {
        FILTER_NONE
    } else {
        estimate_best_filter(alpha, width, height)
    };
    let mut bit_map = 1 << filter;
    if try_filter_none || num_colors > 192 {
        bit_map |= 1 << FILTER_NONE;
    }
    bit_map
}

fn encode_alpha_internal(
    alpha: &[u8],
    width: usize,
    height: usize,
    filter: usize,
    effort_level: i32,
    tmp: &mut [u8],
) -> Vec<u8> {
    let src: &[u8] = if filter == FILTER_NONE {
        alpha
    } else {
        apply_filter(filter, alpha, width, height, tmp);
        tmp
    };
    let compressed = encode_alpha_plane(src, width as u32, height as u32, effort_level, true);
    let (method, payload) = if compressed.len() > src.len() {
        (0u8, src)
    } else {
        (1u8, &compressed[..])
    };
    let mut out = Vec::with_capacity(1 + payload.len());
    out.push(method | (filter as u8) << 2);
    out.extend_from_slice(payload);
    out
}

/// The `ALPH` payload of an uncompressed alpha plane, as libwebp writes it with
/// `alpha_compression = 0`: no filter, no preprocessing.
pub(crate) fn encode_alpha_raw(alpha: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + alpha.len());
    out.push(0);
    out.extend_from_slice(alpha);
    out
}

/// The `ALPH` payload of a `width` x `height` alpha plane, as libwebp's lossy encoder
/// writes it at `method`.
pub(crate) fn encode_alpha(alpha: &[u8], width: u32, height: u32, method: u8) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let effort_level = i32::from(method.min(6));
    let try_map = get_filter_map(alpha, w, h, effort_level);
    if try_map == 1 << FILTER_NONE {
        return encode_alpha_internal(alpha, w, h, FILTER_NONE, effort_level, &mut []);
    }
    let mut tmp = vec![0u8; alpha.len()];
    let mut best: Option<Vec<u8>> = None;
    for filter in 0..FILTER_LAST {
        if try_map & (1 << filter) != 0 {
            let trial = encode_alpha_internal(alpha, w, h, filter, effort_level, &mut tmp);
            if best.as_ref().map_or(true, |b| trial.len() < b.len()) {
                best = Some(trial);
            }
        }
    }
    best.unwrap()
}
