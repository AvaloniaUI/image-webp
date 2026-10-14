//! NEON versions of the spatial predictor residuals (libwebp's `PredictorSub*`) and of the
//! run masks of `entropy.rs`, bit-exact with the scalar ones and with the API of `sse2.rs`.
//! They handle whole groups of four pixels and return how many they did; the caller
//! finishes the rest.
#![allow(unsafe_code)]

use core::arch::aarch64::*;

#[inline(always)]
unsafe fn load(p: &[u32], i: usize) -> uint8x16_t {
    vreinterpretq_u8_u32(vld1q_u32(p.as_ptr().add(i)))
}

/// Per pixel: `t` if sum(|l - tl|) <= sum(|t - tl|) over the four channels, else `l`.
#[inline(always)]
unsafe fn select(t: uint8x16_t, l: uint8x16_t, tl: uint8x16_t) -> uint8x16_t {
    let sum = |d: uint8x16_t| vpaddlq_u16(vpaddlq_u8(d));
    let use_l = vcgtq_u32(sum(vabdq_u8(l, tl)), sum(vabdq_u8(t, tl)));
    vbslq_u8(vreinterpretq_u8_u32(use_l), l, t)
}

#[inline(always)]
unsafe fn clamped_add_subtract_full(l: uint8x16_t, t: uint8x16_t, tl: uint8x16_t) -> uint8x16_t {
    let half = |l: uint8x8_t, t: uint8x8_t, tl: uint8x8_t| {
        vqmovun_s16(vreinterpretq_s16_u16(vsubw_u8(vaddl_u8(l, t), tl)))
    };
    vcombine_u8(
        half(vget_low_u8(l), vget_low_u8(t), vget_low_u8(tl)),
        half(vget_high_u8(l), vget_high_u8(t), vget_high_u8(tl)),
    )
}

/// `a + (a - b) / 2`, the division rounding towards zero, saturated to 8 bits.
#[inline(always)]
unsafe fn add_half_difference(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let d = vreinterpretq_s16_u16(vsubl_u8(a, b));
    let toward_zero = vreinterpretq_s16_u16(vshrq_n_u16::<15>(vreinterpretq_u16_s16(d)));
    let half = vshrq_n_s16::<1>(vaddq_s16(d, toward_zero));
    vqmovun_s16(vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(a)), half))
}

#[inline(always)]
unsafe fn clamped_add_subtract_half(l: uint8x16_t, t: uint8x16_t, tl: uint8x16_t) -> uint8x16_t {
    let ave = vhaddq_u8(l, t);
    vcombine_u8(
        add_half_difference(vget_low_u8(ave), vget_low_u8(tl)),
        add_half_difference(vget_high_u8(ave), vget_high_u8(tl)),
    )
}

#[inline(always)]
unsafe fn predict<const M: usize>(
    l: uint8x16_t,
    tl: uint8x16_t,
    t: uint8x16_t,
    tr: uint8x16_t,
) -> uint8x16_t {
    match M {
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => vhaddq_u8(vhaddq_u8(l, tr), t),
        6 => vhaddq_u8(l, tl),
        7 => vhaddq_u8(l, t),
        8 => vhaddq_u8(tl, t),
        9 => vhaddq_u8(t, tr),
        10 => vhaddq_u8(vhaddq_u8(l, tl), vhaddq_u8(t, tr)),
        11 => select(t, l, tl),
        12 => clamped_add_subtract_full(l, t, tl),
        13 => clamped_add_subtract_half(l, t, tl),
        _ => vreinterpretq_u8_u32(vdupq_n_u32(0xff000000)),
    }
}

/// Residuals of `cur[i + 1]` for `i` in `0..out.len()`, with left `cur[i]` and top-left,
/// top and top-right `up[i..i + 3]`.
pub(super) fn sub_batch<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) -> usize {
    let n = out.len() & !3;
    assert!(cur.len() > n && up.len() >= n + 2);
    let mut i = 0;
    // SAFETY: NEON is part of aarch64; the loads read cur[i..i + 5] and up[i..i + 6] and the
    // store writes out[i..i + 4], all in bounds for i + 4 <= n by the assert above.
    unsafe {
        while i < n {
            let pred = predict::<M>(load(cur, i), load(up, i), load(up, i + 1), load(up, i + 2));
            let res = vsubq_u8(load(cur, i + 1), pred);
            vst1q_u32(out.as_mut_ptr().add(i), vreinterpretq_u32_u8(res));
            i += 4;
        }
    }
    n
}

/// `sub_batch` for groups of four pixels none of which is fully transparent, stopping at
/// the first group that has one. Returns how many pixels it did.
pub(super) fn sub_batch_opaque<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) -> usize {
    let n = out.len() & !3;
    assert!(cur.len() > n && up.len() >= n + 2);
    let mut i = 0;
    // SAFETY: as in `sub_batch`.
    unsafe {
        while i < n {
            let x = load(cur, i + 1);
            let alpha = vshrq_n_u32::<24>(vreinterpretq_u32_u8(x));
            if vminvq_u32(alpha) == 0 {
                break;
            }
            let pred = predict::<M>(load(cur, i), load(up, i), load(up, i + 1), load(up, i + 2));
            vst1q_u32(
                out.as_mut_ptr().add(i),
                vreinterpretq_u32_u8(vsubq_u8(x, pred)),
            );
            i += 4;
        }
    }
    i
}

/// Bit `j` set where byte `j` of `m` (all ones or zero) is set.
#[inline(always)]
unsafe fn movemask(m: uint8x16_t) -> u16 {
    const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let b = vandq_u8(m, vld1q_u8(BITS.as_ptr()));
    u16::from(vaddv_u8(vget_low_u8(b))) | (u16::from(vaddv_u8(vget_high_u8(b))) << 8)
}

/// `(changed, nonzero)`: bit `j` set where `v[j + 1] != v[j]`, and where `v[j + 1] != 0`.
pub(super) fn run_masks(v: &[u32; 65]) -> (u64, u64) {
    let (mut changed, mut nonzero) = (0u64, 0u64);
    // SAFETY: NEON is part of aarch64; the loads read v[16 * k..16 * k + 17] for k < 4.
    unsafe {
        let narrow = |a: uint32x4_t, b: uint32x4_t, c: uint32x4_t, d: uint32x4_t| {
            vcombine_u8(
                vmovn_u16(vcombine_u16(vmovn_u32(a), vmovn_u32(b))),
                vmovn_u16(vcombine_u16(vmovn_u32(c), vmovn_u32(d))),
            )
        };
        for k in 0..4 {
            let at = |i: usize| vld1q_u32(v.as_ptr().add(16 * k + i));
            let (c0, c1, c2, c3) = (at(1), at(5), at(9), at(13));
            let eq = narrow(
                vceqq_u32(c0, at(0)),
                vceqq_u32(c1, at(4)),
                vceqq_u32(c2, at(8)),
                vceqq_u32(c3, at(12)),
            );
            let is_zero = narrow(
                vceqzq_u32(c0),
                vceqzq_u32(c1),
                vceqzq_u32(c2),
                vceqzq_u32(c3),
            );
            changed |= u64::from(!movemask(eq)) << (16 * k);
            nonzero |= u64::from(!movemask(is_zero)) << (16 * k);
        }
    }
    (changed, nonzero)
}
