//! NEON kernels ported from libwebp's `dsp/enc_neon.c`, bit-exact with the scalar ones.
//! NEON is part of aarch64, so no runtime detection is needed. Each wrapper checks the
//! lengths its loads and stores rely on.
#![allow(unsafe_code)]

use core::arch::aarch64::*;

use super::dsp::{Matrix, BPS, MAX_LEVEL};

const KC1: i16 = 20091;
const KC2: i16 = (35468 / 2) as i16;

/// All 16 pixels of a 4x4 block, row after row. Reads 8 bytes of each row, with vector
/// loads rather than lane inserts from general registers.
#[inline(always)]
unsafe fn load4x4(p: &[u8]) -> uint8x16_t {
    let p = &p[..3 * BPS + 8];
    let row = |y: usize| vreinterpret_u32_u8(vld1_u8(p.as_ptr().add(y * BPS)));
    let r01 = vzip1_u32(row(0), row(1));
    let r23 = vzip1_u32(row(2), row(3));
    vreinterpretq_u8_u32(vcombine_u32(r01, r23))
}

#[inline(always)]
unsafe fn transform_pass(rows: int16x8x2_t) -> int16x8x2_t {
    let b1 = vcombine_s16(vget_high_s16(rows.0), vget_high_s16(rows.1));
    let c0 = vsraq_n_s16::<1>(b1, vqdmulhq_n_s16(b1, KC1));
    let c1 = vqdmulhq_n_s16(b1, KC2);
    let a = vqadd_s16(vget_low_s16(rows.0), vget_low_s16(rows.1));
    let b = vqsub_s16(vget_low_s16(rows.0), vget_low_s16(rows.1));
    let c = vqsub_s16(vget_low_s16(c1), vget_high_s16(c0));
    let d = vqadd_s16(vget_low_s16(c0), vget_high_s16(c1));
    let d0 = vcombine_s16(a, b);
    let d1 = vcombine_s16(d, c);
    let e0 = vqaddq_s16(d0, d1);
    let e_tmp = vqsubq_s16(d0, d1);
    let e1 = vcombine_s16(vget_high_s16(e_tmp), vget_low_s16(e_tmp));
    let tmp0 = vzipq_s16(e0, e1);
    let out = vzipq_s16(tmp0.0, tmp0.1);
    int16x8x2_t(out.0, out.1)
}

#[inline]
pub(crate) fn itransform_one(r: &[u8], input: &[i16], dst: &mut [u8]) {
    let input = &input[..16];
    let r = &r[..3 * BPS + 8];
    let dst = &mut dst[..3 * BPS + 4];
    unsafe {
        let rows = int16x8x2_t(vld1q_s16(input.as_ptr()), vld1q_s16(input.as_ptr().add(8)));
        let rows = transform_pass(transform_pass(rows));
        let px = vreinterpretq_u32_u8(load4x4(r));
        let dst01 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vget_low_u32(px))));
        let dst23 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vget_high_u32(px))));
        let out01 = vqmovun_s16(vrsraq_n_s16::<3>(dst01, rows.0));
        let out23 = vqmovun_s16(vrsraq_n_s16::<3>(dst23, rows.1));
        let (o01, o23) = (vreinterpret_u32_u8(out01), vreinterpret_u32_u8(out23));
        let rows = [
            vget_lane_u32::<0>(o01),
            vget_lane_u32::<1>(o01),
            vget_lane_u32::<0>(o23),
            vget_lane_u32::<1>(o23),
        ];
        for (y, row) in rows.into_iter().enumerate() {
            dst[y * BPS..y * BPS + 4].copy_from_slice(&row.to_le_bytes());
        }
    }
}

#[inline(always)]
unsafe fn transpose4x4_s16(
    a: int16x4_t,
    b: int16x4_t,
    c: int16x4_t,
    d: int16x4_t,
) -> (int16x8_t, int16x8_t) {
    let ab = vtrn_s16(a, b);
    let cd = vtrn_s16(c, d);
    let tmp02 = vtrn_s32(vreinterpret_s32_s16(ab.0), vreinterpret_s32_s16(cd.0));
    let tmp13 = vtrn_s32(vreinterpret_s32_s16(ab.1), vreinterpret_s32_s16(cd.1));
    (
        vreinterpretq_s16_s64(vcombine_s64(
            vreinterpret_s64_s32(tmp02.0),
            vreinterpret_s64_s32(tmp13.0),
        )),
        vreinterpretq_s16_s64(vcombine_s64(
            vreinterpret_s64_s32(tmp13.1),
            vreinterpret_s64_s32(tmp02.1),
        )),
    )
}

#[inline]
pub(crate) fn ftransform(src: &[u8], r: &[u8], out: &mut [i16]) {
    let out = &mut out[..16];
    unsafe {
        let s0 = load4x4(src);
        let r0 = load4x4(r);
        let d0d1 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s0), vget_low_u8(r0)));
        let d2d3 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s0), vget_high_u8(r0)));
        let (d0d1, d3d2) = transpose4x4_s16(
            vget_low_s16(d0d1),
            vget_high_s16(d0d1),
            vget_low_s16(d2d3),
            vget_high_s16(d2d3),
        );

        // First pass.
        let a0a1 = vaddq_s16(d0d1, d3d2);
        let a3a2 = vsubq_s16(d0d1, d3d2);
        let a0a1_2 = vshlq_n_s16::<3>(a0a1);
        let tmp0 = vadd_s16(vget_low_s16(a0a1_2), vget_high_s16(a0a1_2));
        let tmp2 = vsub_s16(vget_low_s16(a0a1_2), vget_high_s16(a0a1_2));
        let a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
        let a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
        let a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
        let a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);
        let tmp1 = vshrn_n_s32::<9>(vaddq_s32(a2_p_a3, vdupq_n_s32(1812)));
        let tmp3 = vshrn_n_s32::<9>(vaddq_s32(a3_m_a2, vdupq_n_s32(937)));
        let (d0d1, d3d2) = transpose4x4_s16(tmp0, tmp1, tmp2, tmp3);

        // Second pass.
        let a0a1 = vaddq_s16(d0d1, d3d2);
        let a3a2 = vsubq_s16(d0d1, d3d2);
        let a0_k7 = vadd_s16(vget_low_s16(a0a1), vdup_n_s16(7));
        let out0 = vshr_n_s16::<4>(vadd_s16(a0_k7, vget_high_s16(a0a1)));
        let out2 = vshr_n_s16::<4>(vsub_s16(a0_k7, vget_high_s16(a0a1)));
        let a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
        let a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
        let a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
        let a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);
        let tmp1 = vaddhn_s32(a2_p_a3, vdupq_n_s32(12000 + (1 << 16)));
        let out3 = vaddhn_s32(a3_m_a2, vdupq_n_s32(51000));
        let a3_eq_0 = vreinterpret_s16_u16(vceq_s16(vget_low_s16(a3a2), vdup_n_s16(0)));
        let out1 = vadd_s16(tmp1, a3_eq_0);
        vst1_s16(out.as_mut_ptr(), out0);
        vst1_s16(out.as_mut_ptr().add(4), out1);
        vst1_s16(out.as_mut_ptr().add(8), out2);
        vst1_s16(out.as_mut_ptr().add(12), out3);
    }
}

#[inline]
pub(crate) fn sse16xn(a: &[u8], b: &[u8], rows: usize) -> i32 {
    let a = &a[..(rows - 1) * BPS + 16];
    let b = &b[..(rows - 1) * BPS + 16];
    unsafe {
        let mut sum = vdupq_n_u32(0);
        for y in 0..rows {
            let abs_diff = vabdq_u8(
                vld1q_u8(a.as_ptr().add(y * BPS)),
                vld1q_u8(b.as_ptr().add(y * BPS)),
            );
            let prod1 = vmull_u8(vget_low_u8(abs_diff), vget_low_u8(abs_diff));
            let prod2 = vmull_u8(vget_high_u8(abs_diff), vget_high_u8(abs_diff));
            sum = vaddq_u32(sum, vaddq_u32(vpaddlq_u16(prod1), vpaddlq_u16(prod2)));
        }
        vaddvq_u32(sum) as i32
    }
}

#[inline]
pub(crate) fn sse4x4(a: &[u8], b: &[u8]) -> i32 {
    unsafe {
        let abs_diff = vabdq_u8(load4x4(a), load4x4(b));
        let prod1 = vmull_u8(vget_low_u8(abs_diff), vget_low_u8(abs_diff));
        let prod2 = vmull_u8(vget_high_u8(abs_diff), vget_high_u8(abs_diff));
        vaddvq_u32(vaddq_u32(vpaddlq_u16(prod1), vpaddlq_u16(prod2))) as i32
    }
}

/// `|weighted Hadamard sum of b - that of a| >> 5`.
#[inline]
pub(crate) fn tdisto4x4(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32 {
    let a = &a[..3 * BPS + 8];
    let b = &b[..3 * BPS + 8];
    unsafe {
        let ld = |p: &[u8], y: usize| vreinterpret_u32_u8(vld1_u8(p.as_ptr().add(y * BPS)));
        let row = |y: usize| vreinterpret_u8_u32(vzip1_u32(ld(a, y), ld(b, y)));
        let (i0, i1, i2, i3) = (row(0), row(1), row(2), row(3));

        // Vertical pass.
        let q_a0 = vreinterpretq_s16_u16(vaddl_u8(i0, i2));
        let q_a1 = vreinterpretq_s16_u16(vaddl_u8(i1, i3));
        let q_a2 = vreinterpretq_s16_u16(vsubl_u8(i1, i3));
        let q_a3 = vreinterpretq_s16_u16(vsubl_u8(i0, i2));
        let v = [
            vaddq_s16(q_a0, q_a1),
            vaddq_s16(q_a3, q_a2),
            vsubq_s16(q_a3, q_a2),
            vsubq_s16(q_a0, q_a1),
        ];

        // Transpose.
        let t0 = vtrnq_s16(v[0], v[1]);
        let t1 = vtrnq_s16(v[2], v[3]);
        let t2 = vtrnq_s32(vreinterpretq_s32_s16(t0.0), vreinterpretq_s32_s16(t1.0));
        let t3 = vtrnq_s32(vreinterpretq_s32_s16(t0.1), vreinterpretq_s32_s16(t1.1));
        let t = [
            vreinterpretq_s16_s32(t2.0),
            vreinterpretq_s16_s32(t3.0),
            vreinterpretq_s16_s32(t2.1),
            vreinterpretq_s16_s32(t3.1),
        ];

        // Horizontal pass.
        let q_a0 = vaddq_s16(t[0], t[2]);
        let q_a1 = vaddq_s16(t[1], t[3]);
        let q_a3 = vsubq_s16(t[0], t[2]);
        let q_a2 = vsubq_s16(t[1], t[3]);
        let h = [
            vabsq_s16(vaddq_s16(q_a0, q_a1)),
            vabsq_s16(vaddq_s16(q_a3, q_a2)),
            vabdq_s16(q_a3, q_a2),
            vabdq_s16(q_a0, q_a1),
        ];

        let q_w07 = vreinterpretq_s16_u16(vld1q_u16(w.as_ptr()));
        let q_w8f = vreinterpretq_s16_u16(vld1q_u16(w.as_ptr().add(8)));
        let wv = [
            vget_low_s16(q_w07),
            vget_high_s16(q_w07),
            vget_low_s16(q_w8f),
            vget_high_s16(q_w8f),
        ];
        let mut sums = [vdupq_n_s32(0); 4];
        for k in 0..4 {
            sums[k] = vmlsl_s16(
                vmull_s16(wv[k], vget_low_s16(h[k])),
                wv[k],
                vget_high_s16(h[k]),
            );
        }
        let sum = vaddq_s32(vaddq_s32(sums[0], sums[1]), vaddq_s32(sums[2], sums[3]));
        vaddvq_s32(sum).abs() >> 5
    }
}

#[inline(always)]
unsafe fn quantize8(input: &mut [i16], mtx: &Matrix, offset: usize) -> int16x8_t {
    let sharp = vld1q_u16(mtx.sharpen.as_ptr().add(offset));
    let q = vld1q_u16(mtx.q.as_ptr().add(offset));
    let iq = vld1q_u16(mtx.iq.as_ptr().add(offset));
    let bias0 = vld1q_u32(mtx.bias.as_ptr().add(offset));
    let bias1 = vld1q_u32(mtx.bias.as_ptr().add(offset + 4));
    let a = vld1q_s16(input.as_ptr().add(offset));
    let b = vreinterpretq_u16_s16(vabsq_s16(a));
    let sign = vshrq_n_s16::<15>(a);
    let c = vaddq_u16(b, sharp);
    let m0 = vmull_u16(vget_low_u16(c), vget_low_u16(iq));
    let m1 = vmull_u16(vget_high_u16(c), vget_high_u16(iq));
    let m2 = vhaddq_u32(m0, bias0);
    let m3 = vhaddq_u32(m1, bias1);
    let c0 = vcombine_u16(vshrn_n_u32::<16>(m2), vshrn_n_u32::<16>(m3));
    let c1 = vminq_u16(c0, vdupq_n_u16(MAX_LEVEL as u16));
    let c2 = veorq_s16(vreinterpretq_s16_u16(c1), sign);
    let c3 = vsubq_s16(c2, sign);
    vst1q_s16(
        input.as_mut_ptr().add(offset),
        vmulq_s16(c3, vreinterpretq_s16_u16(q)),
    );
    c3
}

const SHUFFLES: [[u8; 8]; 4] = [
    [0, 1, 2, 3, 8, 9, 16, 17],
    [10, 11, 4, 5, 6, 7, 12, 13],
    [18, 19, 24, 25, 26, 27, 20, 21],
    [14, 15, 22, 23, 28, 29, 30, 31],
];

#[inline]
pub(crate) fn quantize_block(input: &mut [i16], out: &mut [i16], mtx: &Matrix) -> bool {
    let input = &mut input[..16];
    let out = &mut out[..16];
    unsafe {
        let out0 = quantize8(input, mtx, 0);
        let out1 = quantize8(input, mtx, 8);
        let all = uint8x8x4_t(
            vreinterpret_u8_s16(vget_low_s16(out0)),
            vreinterpret_u8_s16(vget_high_s16(out0)),
            vreinterpret_u8_s16(vget_low_s16(out1)),
            vreinterpret_u8_s16(vget_high_s16(out1)),
        );
        let mut any = vdup_n_u8(0);
        for (k, shuffle) in SHUFFLES.iter().enumerate() {
            let v = vtbl4_u8(all, vld1_u8(shuffle.as_ptr()));
            vst1_u8(out.as_mut_ptr().add(4 * k).cast(), v);
            any = vorr_u8(any, v);
        }
        vget_lane_u64::<0>(vreinterpret_u64_u8(any)) != 0
    }
}

/// The position of the last non-zero coefficient of `coeffs`, or -1: four mask bits per
/// coefficient narrowed into one word.
#[inline]
pub(crate) fn last_non_zero(coeffs: &[i16]) -> i32 {
    let coeffs = &coeffs[..16];
    unsafe {
        let lo = vld1q_s16(coeffs.as_ptr());
        let hi = vld1q_s16(coeffs.as_ptr().add(8));
        let nz = vcombine_u8(vmovn_u16(vtstq_s16(lo, lo)), vmovn_u16(vtstq_s16(hi, hi)));
        let narrowed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(nz));
        let bits = vget_lane_u64::<0>(vreinterpret_u64_u8(narrowed));
        (63 - bits.leading_zeros() as i32) >> 2
    }
}

const LOOKUP1: [u8; 64] = [
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, //
    3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, //
    4, 20, 21, 22, 3, 18, 2, 17, 3, 19, 4, 20, 2, 17, 1, 16, //
    2, 18, 3, 19, 1, 16, 31, 31, 1, 17, 2, 18, 31, 31, 31, 31,
];
const LOOKUP2: [u8; 64] = [
    20, 21, 22, 23, 5, 6, 7, 8, 22, 23, 24, 25, 6, 7, 8, 9, //
    19, 20, 21, 22, 20, 21, 22, 23, 23, 24, 25, 26, 22, 23, 24, 25, //
    18, 19, 20, 21, 19, 5, 6, 7, 24, 25, 26, 27, 7, 8, 9, 26, //
    17, 18, 19, 20, 18, 20, 21, 22, 25, 26, 27, 28, 23, 24, 25, 27,
];
const LOOKUP3: [u8; 64] = [
    30, 30, 30, 30, 0, 0, 0, 0, 21, 22, 23, 24, 19, 19, 19, 19, //
    30, 30, 30, 30, 0, 0, 0, 0, 21, 22, 23, 24, 18, 18, 18, 18, //
    30, 30, 30, 30, 0, 0, 0, 0, 21, 22, 23, 24, 17, 17, 17, 17, //
    30, 30, 30, 30, 0, 0, 0, 0, 21, 22, 23, 24, 16, 16, 16, 16,
];

/// libwebp's `Intra4Preds_NEON`: the ten 4x4 predictions picked from the averages of the
/// edge with table lookups, a whole row of four modes per store. `b[t]` is the first top
/// sample, as for `dsp::pred_luma4`.
#[inline]
pub(crate) fn pred_luma4(dst: &mut [u8], b: &[u8], t: usize) {
    use super::dsp::{I4DC4, I4HD4, I4RD4};
    let edge = &b[t - 5..t + 11];
    let dst = &mut dst[I4DC4..I4DC4 + 7 * BPS + 8];
    unsafe {
        let l1 = vld1q_u8_x4(LOOKUP1.as_ptr());
        let l2 = vld1q_u8_x4(LOOKUP2.as_ptr());
        let l3 = vld1q_u8_x4(LOOKUP3.as_ptr());
        // L K J I X A B C D E F G H
        let preload = vld1q_u8(edge.as_ptr());
        let a = vqtbl1q_u8(preload, l1.0);
        let c = vextq_u8::<2>(a, a);
        let mut avg3_all = vrhaddq_u8(vhaddq_u8(a, c), preload);
        let avg2_all = vrhaddq_u8(a, preload);
        let dc_in = vset_lane_u8::<3>(vgetq_lane_u8::<0>(preload), vget_low_u8(c));
        let dc = ((vaddlv_u8(dc_in) + 4) >> 3) as u8;
        avg3_all = vsetq_lane_u8::<15>(vgetq_lane_u8::<0>(preload), avg3_all);
        avg3_all = vsetq_lane_u8::<14>(dc, avg3_all);
        let combined = uint8x16x2_t(avg2_all, avg3_all);

        // TrueMotion: left + top - top-left, saturated.
        let sub_a = vdup_laneq_u8::<4>(preload);
        let full_b = vqtbl1q_u8(preload, l1.1);
        let abcd = vgetq_lane_u32::<0>(vreinterpretq_u32_u8(vextq_u8::<5>(preload, preload)));
        let sub = vsubl_u8(vreinterpret_u8_u32(vdup_n_u32(abcd)), sub_a);
        let tm_lo = vqmovun_s16(vreinterpretq_s16_u16(vaddw_u8(sub, vget_low_u8(full_b))));
        let tm_hi = vqmovun_s16(vreinterpretq_s16_u16(vaddw_u8(sub, vget_high_u8(full_b))));
        let tm = [
            vget_lane_u32::<0>(vreinterpret_u32_u8(tm_lo)),
            vget_lane_u32::<1>(vreinterpret_u32_u8(tm_lo)),
            vget_lane_u32::<0>(vreinterpret_u32_u8(tm_hi)),
            vget_lane_u32::<1>(vreinterpret_u32_u8(tm_hi)),
        ];

        let p = dst.as_mut_ptr();
        for (y, (tbl3, tbl2)) in [(l3.0, l2.0), (l3.1, l2.1), (l3.2, l2.2), (l3.3, l2.3)]
            .into_iter()
            .enumerate()
        {
            // DC4, TM4, VE4, HE4, then RD4, VR4, LD4, VL4.
            let r = vreinterpretq_u32_u8(vqtbl2q_u8(combined, tbl3));
            let r = vsetq_lane_u32::<1>(tm[y], r);
            vst1q_u8(p.add(y * BPS), vreinterpretq_u8_u32(r));
            vst1q_u8(p.add(I4RD4 - I4DC4 + y * BPS), vqtbl2q_u8(combined, tbl2));
        }
        // HD4, HU4
        let r0 = vqtbl2q_u8(combined, l1.2);
        let r1 = vqtbl2q_u8(combined, l1.3);
        let hd = p.add(I4HD4 - I4DC4);
        vst1_u8(hd, vget_low_u8(r0));
        vst1_u8(hd.add(BPS), vget_high_u8(r0));
        vst1_u8(hd.add(2 * BPS), vget_low_u8(r1));
        vst1_u8(hd.add(3 * BPS), vget_high_u8(r1));
    }
}

/// libwebp's `ConvertRGBToY_NEON`: `(16839 r + 33059 g + 6420 b + 2^15) >> 16` plus 16.
#[inline(always)]
unsafe fn rgb_to_y(r: uint8x8_t, g: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let (r, g, b) = (vmovl_u8(r), vmovl_u8(g), vmovl_u8(b));
    let half = |r: uint16x4_t, g: uint16x4_t, b: uint16x4_t| {
        let y = vmlal_n_u16(vmlal_n_u16(vmull_n_u16(r, 16839), g, 33059), b, 6420);
        vrshrn_n_u32::<16>(y)
    };
    let lo = half(vget_low_u16(r), vget_low_u16(g), vget_low_u16(b));
    let hi = half(vget_high_u16(r), vget_high_u16(g), vget_high_u16(b));
    vqmovn_u16(vaddq_u16(vcombine_u16(lo, hi), vdupq_n_u16(16)))
}

/// The luma of the leading pixels of an RGB (`STEP` 3) or RGBA (`STEP` 4) row, 16 at a time.
/// Returns how many it did.
#[inline]
pub(crate) fn rgb_to_y_row<const STEP: usize>(line: &[u8], out: &mut [u8]) -> usize {
    let n = out.len().min(line.len() / STEP) & !15;
    // SAFETY: the loads read line[..n * STEP] and the stores write out[..n].
    unsafe {
        for i in (0..n).step_by(16) {
            let p = line.as_ptr().add(i * STEP);
            let (r, g, b) = if STEP == 3 {
                let c = vld3q_u8(p);
                (c.0, c.1, c.2)
            } else {
                let c = vld4q_u8(p);
                (c.0, c.1, c.2)
            };
            let lo = rgb_to_y(vget_low_u8(r), vget_low_u8(g), vget_low_u8(b));
            let hi = rgb_to_y(vget_high_u8(r), vget_high_u8(g), vget_high_u8(b));
            vst1q_u8(out.as_mut_ptr().add(i), vcombine_u8(lo, hi));
        }
    }
    n
}
