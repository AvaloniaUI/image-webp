//! Fixed-point logarithms, entropy estimates and prefix codes (libwebp's
//! `dsp/lossless_enc.c` and `dsp/lossless_common.h`).

use super::tables::{LOG2_TABLE, PREFIX_ENCODE_CODE, PREFIX_EXTRA_BITS_VALUE, SLOG2_TABLE};

pub(crate) const LOG_2_PRECISION_BITS: u32 = 23;
const LOG_2_RECIPROCAL_FIXED_DOUBLE: f64 = 12102203.161561485379934310913085937500;
const LOG_2_RECIPROCAL_FIXED: u64 = 12102203;
const APPROX_LOG_WITH_CORRECTION_MAX: u32 = 65536;
const APPROX_LOG_MAX: u32 = 4096;
pub(crate) const NON_TRIVIAL_SYM: u16 = 0xffff;

#[inline]
pub(crate) fn bits_log2_floor(n: u32) -> u32 {
    31 ^ n.leading_zeros()
}

#[inline]
pub(crate) fn div_round(a: i64, b: i64) -> i64 {
    if (a < 0) == (b < 0) {
        (a + b / 2) / b
    } else {
        (a - b / 2) / b
    }
}

#[inline]
pub(crate) fn sub_sample_size(size: u32, bits: u32) -> u32 {
    (size + (1 << bits) - 1) >> bits
}

#[inline]
pub(crate) fn fast_log2(v: u32) -> u32 {
    if v < 256 {
        LOG2_TABLE[v as usize]
    } else {
        fast_log2_slow(v)
    }
}

#[inline]
pub(crate) fn fast_slog2(v: u32) -> u64 {
    if v < 256 {
        SLOG2_TABLE[v as usize]
    } else {
        fast_slog2_slow(v)
    }
}

#[cold]
fn fast_slog2_slow(v: u32) -> u64 {
    if v < APPROX_LOG_WITH_CORRECTION_MAX {
        let orig_v = u64::from(v);
        let log_cnt = bits_log2_floor(v) - 7;
        let y = 1u64 << log_cnt;
        let v = v >> log_cnt;
        let correction = LOG_2_RECIPROCAL_FIXED * (orig_v & (y - 1));
        orig_v * (u64::from(LOG2_TABLE[v as usize]) + (u64::from(log_cnt) << LOG_2_PRECISION_BITS))
            + correction
    } else {
        (LOG_2_RECIPROCAL_FIXED_DOUBLE * f64::from(v) * f64::from(v).ln() + 0.5) as u64
    }
}

#[cold]
fn fast_log2_slow(v: u32) -> u32 {
    if v < APPROX_LOG_WITH_CORRECTION_MAX {
        let orig_v = v;
        let log_cnt = bits_log2_floor(v) - 7;
        let y = 1u32 << log_cnt;
        let v = v >> log_cnt;
        let mut log_2 = LOG2_TABLE[v as usize] + (log_cnt << LOG_2_PRECISION_BITS);
        if orig_v >= APPROX_LOG_MAX {
            let correction = LOG_2_RECIPROCAL_FIXED * u64::from(orig_v & (y - 1));
            log_2 = log_2.wrapping_add(div_round(correction as i64, i64::from(orig_v)) as u32);
        }
        log_2
    } else {
        (LOG_2_RECIPROCAL_FIXED_DOUBLE * f64::from(v).ln() + 0.5) as u32
    }
}

/// `(code, extra_bits)` of `distance`.
#[inline]
pub(crate) fn prefix_encode_bits(distance: u32) -> (u32, u32) {
    if distance < 512 {
        let [code, extra] = PREFIX_ENCODE_CODE[distance as usize];
        (u32::from(code), u32::from(extra))
    } else {
        let d = distance - 1;
        let highest_bit = bits_log2_floor(d);
        let second_highest_bit = (d >> (highest_bit - 1)) & 1;
        (2 * highest_bit + second_highest_bit, highest_bit - 1)
    }
}

/// `(code, extra_bits, extra_bits_value)` of `distance`.
#[inline]
pub(crate) fn prefix_encode(distance: u32) -> (u32, u32, u32) {
    if distance < 512 {
        let [code, extra] = PREFIX_ENCODE_CODE[distance as usize];
        (
            u32::from(code),
            u32::from(extra),
            u32::from(PREFIX_EXTRA_BITS_VALUE[distance as usize]),
        )
    } else {
        let d = distance - 1;
        let highest_bit = bits_log2_floor(d);
        let second_highest_bit = (d >> (highest_bit - 1)) & 1;
        let extra_bits = highest_bit - 1;
        (
            2 * highest_bit + second_highest_bit,
            extra_bits,
            d & ((1 << extra_bits) - 1),
        )
    }
}

#[inline]
pub(crate) fn sub_pixels(a: u32, b: u32) -> u32 {
    let alpha_and_green = 0x00ff00ffu32
        .wrapping_add(a & 0xff00ff00)
        .wrapping_sub(b & 0xff00ff00);
    let red_and_blue = 0xff00ff00u32
        .wrapping_add(a & 0x00ff00ff)
        .wrapping_sub(b & 0x00ff00ff);
    (alpha_and_green & 0xff00ff00) | (red_and_blue & 0x00ff00ff)
}

#[derive(Clone, Copy, Default)]
pub(crate) struct BitEntropy {
    pub(crate) entropy: u64,
    pub(crate) sum: u32,
    pub(crate) nonzeros: i32,
    pub(crate) max_val: u32,
    pub(crate) nonzero_code: u32,
}

impl BitEntropy {
    fn new() -> Self {
        Self {
            entropy: 0,
            sum: 0,
            nonzeros: 0,
            max_val: 0,
            nonzero_code: u32::from(NON_TRIVIAL_SYM),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Streaks {
    pub(crate) counts: [i32; 2],
    pub(crate) streaks: [[i32; 2]; 2],
}

pub(crate) fn bits_entropy_unrefined(array: &[u32]) -> BitEntropy {
    let mut e = BitEntropy::new();
    for (i, &v) in array.iter().enumerate() {
        if v != 0 {
            e.sum = e.sum.wrapping_add(v);
            e.nonzero_code = i as u32;
            e.nonzeros += 1;
            e.entropy = e.entropy.wrapping_add(fast_slog2(v));
            if e.max_val < v {
                e.max_val = v;
            }
        }
    }
    e.entropy = fast_slog2(e.sum).wrapping_sub(e.entropy);
    e
}

const RUN_WORDS: usize = 24;

/// `(changed, nonzero)`: bit `j` set where `v[j + 1] != v[j]`, and where `v[j + 1] != 0`.
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn run_masks_scalar(v: &[u32; 65]) -> (u64, u64) {
    let (mut changed, mut nonzero) = (0u64, 0u64);
    for j in 0..64 {
        changed |= u64::from(v[j + 1] != v[j]) << j;
        nonzero |= u64::from(v[j + 1] != 0) << j;
    }
    (changed, nonzero)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::simd::run_masks;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use run_masks_scalar as run_masks;

/// `GetEntropyUnrefined` of `x` (or of `x + y` when `COMBINED`) without walking the runs.
/// Bit `i` of `starts` is set where a run starts and at `n`, bit `i` of `nonzero` where the
/// value is nonzero. A run is long (`streak > 3`) when no run starts in the 3 positions after
/// it, and a position is in a short run when one starts there or 1 or 2 before with no start
/// in between. The bit entropy terms of a run are the sums of its values' terms.
#[inline(always)]
fn entropy_unrefined<const COMBINED: bool>(x: &[u32], y: &[u32]) -> (BitEntropy, Streaks) {
    let n = x.len();
    assert!(n > 0 && n < RUN_WORDS * 64);
    let y = &y[..if COMBINED { n } else { 0 }];
    let words = n / 64 + 1;
    let mut starts = [0u64; RUN_WORDS + 1];
    let mut nonzero = [0u64; RUN_WORDS];
    let (mut sum, mut max_val, mut entropy) = (0u32, 0u32, 0u64);
    let mut v = [0u32; 65];
    v[64] = !(x[0].wrapping_add(if COMBINED { y[0] } else { 0 }));
    for k in 0..words {
        let base = k * 64;
        v[0] = v[64];
        let chunk: &mut [u32; 64] = (&mut v[1..]).try_into().unwrap();
        if base + 64 <= n {
            let xs: &[u32; 64] = x[base..base + 64].try_into().unwrap();
            if COMBINED {
                let ys: &[u32; 64] = y[base..base + 64].try_into().unwrap();
                for ((v, &x), &y) in chunk.iter_mut().zip(xs).zip(ys) {
                    *v = x.wrapping_add(y);
                }
            } else {
                *chunk = *xs;
            }
        } else {
            let (head, tail) = chunk.split_at_mut(n - base);
            for (j, v) in head.iter_mut().enumerate() {
                *v = if COMBINED {
                    x[base + j].wrapping_add(y[base + j])
                } else {
                    x[base + j]
                };
            }
            tail.fill(0);
        }
        #[cfg(target_arch = "x86_64")]
        let stats = super::avx2::chunk_stats(chunk);
        #[cfg(not(target_arch = "x86_64"))]
        let stats = None;
        if let Some((s, m, e)) = stats {
            sum = sum.wrapping_add(s);
            max_val = max_val.max(m);
            entropy = entropy.wrapping_add(e);
        } else {
            for &v in chunk.iter() {
                sum = sum.wrapping_add(v);
                max_val = max_val.max(v);
            }
            for &v in chunk.iter() {
                entropy = entropy.wrapping_add(fast_slog2(v));
            }
        }
        let (mut s, z) = run_masks(&v);
        if base + 64 > n {
            let end = n - base;
            s = (s & ((1u64 << end) - 1)) | (1u64 << end);
        }
        starts[k] = s;
        nonzero[k] = z;
    }

    let mut stats = Streaks::default();
    let mut nonzeros = 0u32;
    let (mut short_prev, mut starts_prev) = (0u64, 0u64);
    for k in 0..words {
        let (c, next) = (starts[k], starts[k + 1]);
        let shr = |s: u32| (c >> s) | (next << (64 - s));
        let long_start = c & !(shr(1) | shr(2) | shr(3));
        let short_start = c & !long_start;
        let in_short = short_start
            | (((short_start << 1) | (short_prev >> 63)) & !c)
            | (((short_start << 2) | (short_prev >> 62)) & !c & !((c << 1) | (starts_prev >> 63)));
        (short_prev, starts_prev) = (short_start, c);
        let valid = if k + 1 == words {
            (1u64 << (n % 64)) - 1
        } else {
            !0
        };
        let (nz, zero) = (nonzero[k] & valid, !nonzero[k] & valid);
        let (long_start, in_long, in_short) =
            (long_start & valid, !in_short & valid, in_short & valid);
        nonzeros += nz.count_ones();
        stats.counts[0] += (long_start & zero).count_ones() as i32;
        stats.counts[1] += (long_start & nz).count_ones() as i32;
        stats.streaks[0][0] += (in_short & zero).count_ones() as i32;
        stats.streaks[0][1] += (in_long & zero).count_ones() as i32;
        stats.streaks[1][0] += (in_short & nz).count_ones() as i32;
        stats.streaks[1][1] += (in_long & nz).count_ones() as i32;
    }

    let mut nonzero_code = u32::from(NON_TRIVIAL_SYM);
    if let Some(k) = (0..words).rev().find(|&k| nonzero[k] != 0) {
        let last_nonzero = 63 - nonzero[k].leading_zeros();
        let mut mask = u64::MAX >> (63 - last_nonzero);
        for k in (0..=k).rev() {
            let s = starts[k] & mask;
            if s != 0 {
                nonzero_code = (k * 64) as u32 + 63 - s.leading_zeros();
                break;
            }
            mask = u64::MAX;
        }
    }

    let e = BitEntropy {
        entropy: fast_slog2(sum).wrapping_sub(entropy),
        sum,
        nonzeros: nonzeros as i32,
        max_val,
        nonzero_code,
    };
    (e, stats)
}

pub(crate) fn get_entropy_unrefined(x: &[u32]) -> (BitEntropy, Streaks) {
    entropy_unrefined::<false>(x, x)
}

pub(crate) fn get_combined_entropy_unrefined(x: &[u32], y: &[u32]) -> (BitEntropy, Streaks) {
    entropy_unrefined::<true>(x, y)
}

pub(crate) fn bits_entropy_refine(e: &BitEntropy) -> u64 {
    let mix: u64;
    if e.nonzeros < 5 {
        if e.nonzeros <= 1 {
            return 0;
        }
        if e.nonzeros == 2 {
            return div_round(
                (99 * (u64::from(e.sum) << LOG_2_PRECISION_BITS)).wrapping_add(e.entropy) as i64,
                100,
            ) as u64;
        }
        mix = if e.nonzeros == 3 { 950 } else { 700 };
    } else {
        mix = 627;
    }
    let min_limit =
        u64::from(2u32.wrapping_mul(e.sum).wrapping_sub(e.max_val)) << LOG_2_PRECISION_BITS;
    let min_limit = div_round(
        (mix * min_limit).wrapping_add((1000 - mix).wrapping_mul(e.entropy)) as i64,
        1000,
    ) as u64;
    if e.entropy < min_limit {
        min_limit
    } else {
        e.entropy
    }
}

pub(crate) fn bits_entropy(array: &[u32]) -> u64 {
    bits_entropy_refine(&bits_entropy_unrefined(array))
}

/// `CombinedShannonEntropy_SSE2`: only the bins where `x` or `y` is nonzero contribute.
pub(crate) fn combined_shannon_entropy(x: &[u32; 256], y: &[u32; 256]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if let Some(e) = super::avx2::combined_shannon_entropy(x, y) {
        return e;
    }
    let mut retval = 0u64;
    let (mut sum_x, mut sum_xy) = (0u32, 0u32);
    for (x, y) in x.chunks_exact(16).zip(y.chunks_exact(16)) {
        let (x, y): (&[u32; 16], &[u32; 16]) = (x.try_into().unwrap(), y.try_into().unwrap());
        let mut used = 0u32;
        for j in 0..16 {
            used |= u32::from((x[j] | y[j]) != 0) << j;
        }
        while used != 0 {
            let j = used.trailing_zeros() as usize;
            let xy = x[j].wrapping_add(y[j]);
            sum_x = sum_x.wrapping_add(x[j]);
            sum_xy = sum_xy.wrapping_add(xy);
            retval = retval
                .wrapping_add(fast_slog2(x[j]))
                .wrapping_add(fast_slog2(xy));
            used &= used - 1;
        }
    }
    fast_slog2(sum_x)
        .wrapping_add(fast_slog2(sum_xy))
        .wrapping_sub(retval)
}

pub(crate) fn shannon_entropy(x: &[u32]) -> u64 {
    let mut retval = 0u64;
    let mut sum_x = 0u32;
    for &v in x {
        if v != 0 {
            sum_x = sum_x.wrapping_add(v);
            retval = retval.wrapping_add(fast_slog2(v));
        }
    }
    fast_slog2(sum_x).wrapping_sub(retval)
}

pub(crate) fn extra_cost(population: &[u32]) -> u32 {
    let length = population.len();
    let mut cost = population[4].wrapping_add(population[5]);
    for i in 2..length / 2 - 1 {
        cost = cost.wrapping_add(
            (i as u32).wrapping_mul(population[2 * i + 2].wrapping_add(population[2 * i + 3])),
        );
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_runs(x: &[u32]) -> (BitEntropy, Streaks) {
        let mut stats = Streaks::default();
        let mut e = BitEntropy::new();
        let (mut i_prev, mut prev) = (0i32, x[0]);
        for i in 1..=x.len() {
            let v = x.get(i).copied().unwrap_or(0);
            if v != prev || i == x.len() {
                let streak = i as i32 - i_prev;
                if prev != 0 {
                    e.sum = e.sum.wrapping_add(prev.wrapping_mul(streak as u32));
                    e.nonzeros += streak;
                    e.nonzero_code = i_prev as u32;
                    e.entropy = e
                        .entropy
                        .wrapping_add(fast_slog2(prev).wrapping_mul(streak as u64));
                    e.max_val = e.max_val.max(prev);
                }
                let nz = usize::from(prev != 0);
                let long = usize::from(streak > 3);
                stats.counts[nz] += long as i32;
                stats.streaks[nz][long] += streak;
                (prev, i_prev) = (v, i as i32);
            }
        }
        e.entropy = fast_slog2(e.sum).wrapping_sub(e.entropy);
        (e, stats)
    }

    fn key(r: (BitEntropy, Streaks)) -> (u64, u32, i32, u32, u32, [i32; 2], [[i32; 2]; 2]) {
        let (e, s) = r;
        (
            e.entropy,
            e.sum,
            e.nonzeros,
            e.max_val,
            e.nonzero_code,
            s.counts,
            s.streaks,
        )
    }

    #[test]
    fn run_masks_match_scalar() {
        let mut v = [0u32; 65];
        for seed in 0..2000u32 {
            for (j, v) in v.iter_mut().enumerate() {
                let h = (seed.wrapping_mul(2654435761) ^ (j as u32).wrapping_mul(40503))
                    .rotate_left(seed % 32);
                *v = if h % 3 == 0 {
                    0
                } else if h % 3 == 1 {
                    7
                } else {
                    h
                };
            }
            assert_eq!(run_masks(&v), run_masks_scalar(&v));
        }
    }

    #[test]
    fn entropy_runs_match_scalar() {
        let mut seed = 1u32;
        let mut next = |m: u32| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) % m
        };
        for len in [1usize, 3, 40, 63, 64, 65, 128, 256, 280, 281, 296, 1304] {
            for _ in 0..200 {
                let ranges = [2u32, 300, 70000, 1 << 24];
                let (rx, ry) = (ranges[next(4) as usize], ranges[next(4) as usize]);
                let mut x: Vec<u32> = (0..len)
                    .map(|_| if next(3) == 0 { 0 } else { next(rx) })
                    .collect();
                let y: Vec<u32> = (0..len)
                    .map(|_| if next(2) == 0 { 0 } else { next(ry) })
                    .collect();
                for i in 1..len {
                    if next(4) == 0 {
                        x[i] = x[i - 1];
                    }
                }
                assert!(key(get_entropy_unrefined(&x)) == key(reference_runs(&x)));
                let xy: Vec<u32> = x.iter().zip(&y).map(|(a, b)| a.wrapping_add(*b)).collect();
                assert!(key(get_combined_entropy_unrefined(&x, &y)) == key(reference_runs(&xy)));
                if len == 256 {
                    let (xa, ya): (&[u32; 256], &[u32; 256]) =
                        (x[..].try_into().unwrap(), y[..].try_into().unwrap());
                    let mut want = 0u64;
                    let (mut sx, mut sxy) = (0u32, 0u32);
                    for (&a, &b) in xa.iter().zip(ya) {
                        sx = sx.wrapping_add(a);
                        sxy = sxy.wrapping_add(a.wrapping_add(b));
                        want = want
                            .wrapping_add(fast_slog2(a))
                            .wrapping_add(fast_slog2(a.wrapping_add(b)));
                    }
                    want = fast_slog2(sx)
                        .wrapping_add(fast_slog2(sxy))
                        .wrapping_sub(want);
                    assert_eq!(combined_shannon_entropy(xa, ya), want);
                }
            }
        }
    }
}
