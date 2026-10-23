//! NEON decoder kernels with the API of `sse2.rs`, ported from libwebp's `dsp/dec_neon.c`,
//! `dsp/upsampling_neon.c` and the SSE2 kernels, bit-exact with the portable ones. NEON is
//! part of aarch64, so no runtime detection is needed. Each wrapper checks the bounds its
//! loads and stores rely on; `p` is the offset of the first pixel past the edge.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.
#![allow(unsafe_code)]

use core::arch::aarch64::*;

#[inline(always)]
fn check(len: usize, p: usize, back: usize, fwd: usize) {
    assert!(p >= back && p.checked_add(fwd).is_some_and(|end| end <= len));
}

#[inline(always)]
unsafe fn flip(x: uint8x16_t) -> int8x16_t {
    vreinterpretq_s8_u8(veorq_u8(x, vdupq_n_u8(0x80)))
}

#[inline(always)]
unsafe fn flip_back(x: int8x16_t) -> uint8x16_t {
    veorq_u8(vreinterpretq_u8_s8(x), vdupq_n_u8(0x80))
}

#[inline(always)]
unsafe fn needs_filter(
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    thresh: u8,
) -> uint8x16_t {
    let a_p0_q0 = vabdq_u8(p0, q0);
    let sum = vqaddq_u8(
        vqaddq_u8(a_p0_q0, a_p0_q0),
        vshrq_n_u8::<1>(vabdq_u8(p1, q1)),
    );
    vcgeq_u8(vdupq_n_u8(thresh), sum)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn needs_filter2(
    p3: uint8x16_t,
    p2: uint8x16_t,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    q2: uint8x16_t,
    q3: uint8x16_t,
    thresh: u8,
    ithresh: u8,
) -> uint8x16_t {
    let m1 = vmaxq_u8(vabdq_u8(p3, p2), vabdq_u8(p2, p1));
    let m2 = vmaxq_u8(vabdq_u8(p1, p0), vabdq_u8(q3, q2));
    let m3 = vmaxq_u8(vabdq_u8(q2, q1), vabdq_u8(q1, q0));
    let interior = vcgeq_u8(vdupq_n_u8(ithresh), vmaxq_u8(vmaxq_u8(m1, m2), m3));
    vandq_u8(interior, needs_filter(p1, p0, q0, q1, thresh))
}

#[inline(always)]
unsafe fn needs_hev(
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    hev: u8,
) -> uint8x16_t {
    let m = vmaxq_u8(vabdq_u8(p1, p0), vabdq_u8(q1, q0));
    vcgtq_u8(m, vdupq_n_u8(hev))
}

#[inline(always)]
unsafe fn base_delta(p1: int8x16_t, p0: int8x16_t, q0: int8x16_t, q1: int8x16_t) -> int8x16_t {
    let q0_p0 = vqsubq_s8(q0, p0);
    let s1 = vqaddq_s8(vqsubq_s8(p1, q1), q0_p0);
    let s2 = vqaddq_s8(q0_p0, s1);
    vqaddq_s8(q0_p0, s2)
}

#[inline(always)]
unsafe fn base_delta0(p0: int8x16_t, q0: int8x16_t) -> int8x16_t {
    let q0_p0 = vqsubq_s8(q0, p0);
    vqaddq_s8(q0_p0, vqaddq_s8(q0_p0, q0_p0))
}

/// libwebp's `ApplyFilter2NoFlip`: `p0 + (delta + 3) >> 3` and `q0 - (delta + 4) >> 3`.
#[inline(always)]
unsafe fn apply_filter2(p0: int8x16_t, q0: int8x16_t, delta: int8x16_t) -> (int8x16_t, int8x16_t) {
    let d3 = vshrq_n_s8::<3>(vqaddq_s8(delta, vdupq_n_s8(3)));
    let d4 = vshrq_n_s8::<3>(vqaddq_s8(delta, vdupq_n_s8(4)));
    (vqaddq_s8(p0, d3), vqsubq_s8(q0, d4))
}

#[inline(always)]
unsafe fn do_filter2(
    p1: uint8x16_t,
    p0: &mut uint8x16_t,
    q0: &mut uint8x16_t,
    q1: uint8x16_t,
    thresh: u8,
) {
    let mask = needs_filter(p1, *p0, *q0, q1, thresh);
    let (p0s, q0s) = (flip(*p0), flip(*q0));
    let delta = vandq_s8(
        base_delta(flip(p1), p0s, q0s, flip(q1)),
        vreinterpretq_s8_u8(mask),
    );
    let (p0s, q0s) = apply_filter2(p0s, q0s, delta);
    *p0 = flip_back(p0s);
    *q0 = flip_back(q0s);
}

#[inline(always)]
unsafe fn do_filter4(
    p1: &mut uint8x16_t,
    p0: &mut uint8x16_t,
    q0: &mut uint8x16_t,
    q1: &mut uint8x16_t,
    mask: uint8x16_t,
    hev_mask: uint8x16_t,
) {
    let (p1s, q1s) = (flip(*p1), flip(*q1));
    let simple = vandq_u8(mask, hev_mask);
    let delta = base_delta(p1s, flip(*p0), flip(*q0), q1s);
    let (p0s, q0s) = apply_filter2(
        flip(*p0),
        flip(*q0),
        vandq_s8(delta, vreinterpretq_s8_u8(simple)),
    );
    let complex = veorq_u8(simple, mask);
    let delta0 = vandq_s8(base_delta0(p0s, q0s), vreinterpretq_s8_u8(complex));
    let a1 = vshrq_n_s8::<3>(vqaddq_s8(delta0, vdupq_n_s8(4)));
    let a2 = vshrq_n_s8::<3>(vqaddq_s8(delta0, vdupq_n_s8(3)));
    let a3 = vrshrq_n_s8::<1>(a1);
    *p0 = flip_back(vqaddq_s8(p0s, a2));
    *q0 = flip_back(vqsubq_s8(q0s, a1));
    *p1 = flip_back(vqaddq_s8(p1s, a3));
    *q1 = flip_back(vqsubq_s8(q1s, a3));
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn do_filter6(
    p2: &mut uint8x16_t,
    p1: &mut uint8x16_t,
    p0: &mut uint8x16_t,
    q0: &mut uint8x16_t,
    q1: &mut uint8x16_t,
    q2: &mut uint8x16_t,
    mask: uint8x16_t,
    hev_mask: uint8x16_t,
) {
    let (p2s, p1s, q1s, q2s) = (flip(*p2), flip(*p1), flip(*q1), flip(*q2));
    let simple = vandq_u8(mask, hev_mask);
    let delta0 = base_delta(p1s, flip(*p0), flip(*q0), q1s);
    let (p0s, q0s) = apply_filter2(
        flip(*p0),
        flip(*q0),
        vandq_s8(delta0, vreinterpretq_s8_u8(simple)),
    );
    let delta = vandq_s8(delta0, vreinterpretq_s8_u8(veorq_u8(simple, mask)));
    // With S = 9 * a - 1: (9 * a + 63) >> 7 = (S + 64) >> 7, (18 * a + 63) >> 7 =
    // (S + 32) >> 6 and (27 * a + 63) >> 7 = (S + 18 * a + 64) >> 7.
    let (lo, hi) = (vget_low_s8(delta), vget_high_s8(delta));
    let m1 = vdupq_n_s16(-1);
    let s_lo = vmlal_s8(m1, vdup_n_s8(9), lo);
    let s_hi = vmlal_s8(m1, vdup_n_s8(9), hi);
    let z_lo = vmlal_s8(s_lo, vdup_n_s8(18), lo);
    let z_hi = vmlal_s8(s_hi, vdup_n_s8(18), hi);
    let a3 = vcombine_s8(vqrshrn_n_s16::<7>(s_lo), vqrshrn_n_s16::<7>(s_hi));
    let a2 = vcombine_s8(vqrshrn_n_s16::<6>(s_lo), vqrshrn_n_s16::<6>(s_hi));
    let a1 = vcombine_s8(vqrshrn_n_s16::<7>(z_lo), vqrshrn_n_s16::<7>(z_hi));
    *p0 = flip_back(vqaddq_s8(p0s, a1));
    *q0 = flip_back(vqsubq_s8(q0s, a1));
    *p1 = flip_back(vqaddq_s8(p1s, a2));
    *q1 = flip_back(vqsubq_s8(q1s, a2));
    *p2 = flip_back(vqaddq_s8(p2s, a3));
    *q2 = flip_back(vqsubq_s8(q2s, a3));
}

/// Two side-by-side 8x8 byte transposes, one in each half of the registers.
#[inline(always)]
unsafe fn transpose8x8x2(r: [uint8x16_t; 8]) -> [uint8x16_t; 8] {
    let r01 = vtrnq_u8(r[0], r[1]);
    let r23 = vtrnq_u8(r[2], r[3]);
    let r45 = vtrnq_u8(r[4], r[5]);
    let r67 = vtrnq_u8(r[6], r[7]);
    let u16 = vreinterpretq_u16_u8;
    let r02 = vtrnq_u16(u16(r01.0), u16(r23.0));
    let r13 = vtrnq_u16(u16(r01.1), u16(r23.1));
    let r46 = vtrnq_u16(u16(r45.0), u16(r67.0));
    let r57 = vtrnq_u16(u16(r45.1), u16(r67.1));
    let u32 = vreinterpretq_u32_u16;
    let r04 = vtrnq_u32(u32(r02.0), u32(r46.0));
    let r26 = vtrnq_u32(u32(r02.1), u32(r46.1));
    let r15 = vtrnq_u32(u32(r13.0), u32(r57.0));
    let r37 = vtrnq_u32(u32(r13.1), u32(r57.1));
    let u8 = vreinterpretq_u8_u32;
    [
        u8(r04.0),
        u8(r15.0),
        u8(r26.0),
        u8(r37.0),
        u8(r04.1),
        u8(r15.1),
        u8(r26.1),
        u8(r37.1),
    ]
}

/// 8 columns of 8 rows from `a` (low halves) and 8 rows from `b` (high halves), one
/// register per column.
#[inline(always)]
unsafe fn load8x16t(a: *const u8, b: *const u8, stride: usize) -> [uint8x16_t; 8] {
    let row = |k: usize| vcombine_u8(vld1_u8(a.add(k * stride)), vld1_u8(b.add(k * stride)));
    transpose8x8x2([
        row(0),
        row(1),
        row(2),
        row(3),
        row(4),
        row(5),
        row(6),
        row(7),
    ])
}

#[inline(always)]
unsafe fn store8x16t(c: [uint8x16_t; 8], a: *mut u8, b: *mut u8, stride: usize) {
    for (k, r) in transpose8x8x2(c).into_iter().enumerate() {
        vst1_u8(a.add(k * stride), vget_low_u8(r));
        vst1_u8(b.add(k * stride), vget_high_u8(r));
    }
}

#[inline(always)]
unsafe fn load16(b: *const u8) -> uint8x16_t {
    vld1q_u8(b)
}

#[inline(always)]
unsafe fn store16(b: *mut u8, x: uint8x16_t) {
    vst1q_u8(b, x)
}

#[inline(always)]
unsafe fn load_uv(u: *const u8, v: *const u8) -> uint8x16_t {
    vcombine_u8(vld1_u8(u), vld1_u8(v))
}

#[inline(always)]
unsafe fn store_uv(x: uint8x16_t, u: *mut u8, v: *mut u8) {
    vst1_u8(u, vget_low_u8(x));
    vst1_u8(v, vget_high_u8(x));
}

pub(crate) fn simple_v16(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    check(buf.len(), p, 2 * stride, stride + 16);
    unsafe {
        let b = buf.as_mut_ptr().add(p);
        let p1 = load16(b.sub(2 * stride));
        let mut p0 = load16(b.sub(stride));
        let mut q0 = load16(b);
        let q1 = load16(b.add(stride));
        do_filter2(p1, &mut p0, &mut q0, q1, thresh);
        store16(b.sub(stride), p0);
        store16(b, q0);
    }
}

pub(crate) fn simple_h16(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    check(buf.len(), p, 4, 15 * stride + 4);
    unsafe {
        let b = buf.as_mut_ptr().add(p - 4);
        let mut c = load8x16t(b, b.add(8 * stride), stride);
        let [_, _, p1, mut p0, mut q0, q1, _, _] = c;
        do_filter2(p1, &mut p0, &mut q0, q1, thresh);
        (c[3], c[4]) = (p0, q0);
        store8x16t(c, b, b.add(8 * stride), stride);
    }
}

pub(crate) fn simple_v16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    for k in 1..4 {
        simple_v16(buf, p + 4 * k * stride, stride, thresh);
    }
}

pub(crate) fn simple_h16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    for k in 1..4 {
        simple_h16(buf, p + 4 * k, stride, thresh);
    }
}

pub(crate) fn v16(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    check(buf.len(), p, 4 * stride, 3 * stride + 16);
    unsafe {
        let b = buf.as_mut_ptr().add(p);
        let row = |k: isize| load16(b.offset(k * stride as isize));
        let (p3, mut p2, mut p1, mut p0) = (row(-4), row(-3), row(-2), row(-1));
        let (mut q0, mut q1, mut q2, q3) = (row(0), row(1), row(2), row(3));
        let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
        let hev_mask = needs_hev(p1, p0, q0, q1, hev);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
        );
        for (k, x) in [(-3isize, p2), (-2, p1), (-1, p0), (0, q0), (1, q1), (2, q2)] {
            store16(b.offset(k * stride as isize), x);
        }
    }
}

pub(crate) fn h16(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    check(buf.len(), p, 4, 15 * stride + 4);
    unsafe {
        let b = buf.as_mut_ptr().add(p - 4);
        let [p3, mut p2, mut p1, mut p0, mut q0, mut q1, mut q2, q3] =
            load8x16t(b, b.add(8 * stride), stride);
        let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
        let hev_mask = needs_hev(p1, p0, q0, q1, hev);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
        );
        store8x16t(
            [p3, p2, p1, p0, q0, q1, q2, q3],
            b,
            b.add(8 * stride),
            stride,
        );
    }
}

pub(crate) fn v16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    check(buf.len(), p, 0, 15 * stride + 16);
    unsafe {
        let mut b = buf.as_mut_ptr().add(p);
        let (mut p3, mut p2, mut p1, mut p0) = (
            load16(b),
            load16(b.add(stride)),
            load16(b.add(2 * stride)),
            load16(b.add(3 * stride)),
        );
        for _ in 0..3 {
            let d = b.add(2 * stride);
            b = b.add(4 * stride);
            let (mut q0, mut q1, q2, q3) = (
                load16(b),
                load16(b.add(stride)),
                load16(b.add(2 * stride)),
                load16(b.add(3 * stride)),
            );
            let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
            let hev_mask = needs_hev(p1, p0, q0, q1, hev);
            do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);
            store16(d, p1);
            store16(d.add(stride), p0);
            store16(d.add(2 * stride), q0);
            store16(d.add(3 * stride), q1);
            (p3, p2, p1, p0) = (q0, q1, q2, q3);
        }
    }
}

pub(crate) fn h16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    check(buf.len(), p, 0, 15 * stride + 16);
    unsafe {
        let b = buf.as_mut_ptr().add(p);
        let (b0, b8) = (b, b.add(8 * stride));
        // All 16 columns: the left 8 and the right 8, each filtered edge in registers.
        let [l0, l1, l2, l3, l4, l5, l6, l7] = load8x16t(b0, b8, stride);
        let [r0, r1, r2, r3, r4, r5, r6, r7] = load8x16t(b0.add(8), b8.add(8), stride);
        let mut c = [
            l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7,
        ];
        for e in [4, 8, 12] {
            let mask = needs_filter2(
                c[e - 4],
                c[e - 3],
                c[e - 2],
                c[e - 1],
                c[e],
                c[e + 1],
                c[e + 2],
                c[e + 3],
                thresh,
                ithresh,
            );
            let hev_mask = needs_hev(c[e - 2], c[e - 1], c[e], c[e + 1], hev);
            let [mut p1, mut p0, mut q0, mut q1] = [c[e - 2], c[e - 1], c[e], c[e + 1]];
            do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);
            (c[e - 2], c[e - 1], c[e], c[e + 1]) = (p1, p0, q0, q1);
        }
        store8x16t(
            [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]],
            b0,
            b8,
            stride,
        );
        store8x16t(
            [c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]],
            b0.add(8),
            b8.add(8),
            stride,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn v8(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    check(u.len(), p, 4 * stride, 3 * stride + 8);
    check(v.len(), p, 4 * stride, 3 * stride + 8);
    unsafe {
        let (u, v) = (u.as_mut_ptr().add(p), v.as_mut_ptr().add(p));
        let row = |k: isize| load_uv(u.offset(k * stride as isize), v.offset(k * stride as isize));
        let (p3, mut p2, mut p1, mut p0) = (row(-4), row(-3), row(-2), row(-1));
        let (mut q0, mut q1, mut q2, q3) = (row(0), row(1), row(2), row(3));
        let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
        let hev_mask = needs_hev(p1, p0, q0, q1, hev);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
        );
        for (k, x) in [(-3isize, p2), (-2, p1), (-1, p0), (0, q0), (1, q1), (2, q2)] {
            store_uv(
                x,
                u.offset(k * stride as isize),
                v.offset(k * stride as isize),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn h8(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    check(u.len(), p, 4, 7 * stride + 4);
    check(v.len(), p, 4, 7 * stride + 4);
    unsafe {
        let (u, v) = (u.as_mut_ptr().add(p - 4), v.as_mut_ptr().add(p - 4));
        let [p3, mut p2, mut p1, mut p0, mut q0, mut q1, mut q2, q3] = load8x16t(u, v, stride);
        let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
        let hev_mask = needs_hev(p1, p0, q0, q1, hev);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
        );
        store8x16t([p3, p2, p1, p0, q0, q1, q2, q3], u, v, stride);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn v8i(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    check(u.len(), p, 0, 7 * stride + 8);
    check(v.len(), p, 0, 7 * stride + 8);
    unsafe {
        let (u, v) = (u.as_mut_ptr().add(p), v.as_mut_ptr().add(p));
        let row = |k: usize| load_uv(u.add(k * stride), v.add(k * stride));
        let (p3, p2, mut p1, mut p0) = (row(0), row(1), row(2), row(3));
        let (mut q0, mut q1, q2, q3) = (row(4), row(5), row(6), row(7));
        let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
        let hev_mask = needs_hev(p1, p0, q0, q1, hev);
        do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);
        for (k, x) in [(2usize, p1), (3, p0), (4, q0), (5, q1)] {
            store_uv(x, u.add(k * stride), v.add(k * stride));
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn h8i(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    check(u.len(), p, 0, 7 * stride + 8);
    check(v.len(), p, 0, 7 * stride + 8);
    unsafe {
        let (u, v) = (u.as_mut_ptr().add(p), v.as_mut_ptr().add(p));
        let [p3, p2, mut p1, mut p0, mut q0, mut q1, q2, q3] = load8x16t(u, v, stride);
        let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
        let hev_mask = needs_hev(p1, p0, q0, q1, hev);
        do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);
        store8x16t([p3, p2, p1, p0, q0, q1, q2, q3], u, v, stride);
    }
}

/// libwebp's `CONVERT8` of `upsampling_neon.c`: 8 pixels to R, G, B. The samples are
/// shifted up by 7 so that `vqdmulh` (`2 * a * b >> 16`) gives libwebp's `MultHi`, and
/// `u * 33050 >> 8` is `u << 7` plus `u * 282 >> 8`.
#[inline(always)]
unsafe fn yuv444_to_rgb(y: *const u8, u: *const u8, v: *const u8) -> [uint8x8_t; 3] {
    let y0 = vreinterpretq_s16_u16(vshll_n_u8::<7>(vld1_u8(y)));
    let u0 = vreinterpretq_s16_u16(vshll_n_u8::<7>(vld1_u8(u)));
    let v0 = vreinterpretq_s16_u16(vshll_n_u8::<7>(vld1_u8(v)));
    let y1 = vqdmulhq_n_s16(y0, 19077);
    let r0 = vqdmulhq_n_s16(v0, 26149);
    let g0 = vqdmulhq_n_s16(u0, 6419);
    let g1 = vqdmulhq_n_s16(v0, 13320);
    let b0 = vqdmulhq_n_s16(u0, 282);
    let r = vqaddq_s16(r0, vqaddq_s16(y1, vdupq_n_s16(-14234)));
    let g = vqsubq_s16(vqaddq_s16(y1, vdupq_n_s16(8708)), vqaddq_s16(g0, g1));
    let b = vqaddq_s16(vqaddq_s16(b0, vqaddq_s16(y1, vdupq_n_s16(-17685))), u0);
    [
        vqshrun_n_s16::<6>(r),
        vqshrun_n_s16::<6>(g),
        vqshrun_n_s16::<6>(b),
    ]
}

/// RGB (`BPP` 3) or RGBA with alpha 255 (`BPP` 4) of the leading pixels of a row with
/// full-width chroma, 16 or 8 pixels at a time. Returns how many pixels it converted.
pub(crate) fn yuv_to_rgb_row<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
) -> usize {
    let n = y.len().min(u.len()).min(v.len()).min(out.len() / BPP);
    let mut i = 0;
    unsafe {
        let (y, u, v, out) = (y.as_ptr(), u.as_ptr(), v.as_ptr(), out.as_mut_ptr());
        while i + 16 <= n {
            let [r0, g0, b0] = yuv444_to_rgb(y.add(i), u.add(i), v.add(i));
            let [r1, g1, b1] = yuv444_to_rgb(y.add(i + 8), u.add(i + 8), v.add(i + 8));
            let (r, g, b) = (
                vcombine_u8(r0, r1),
                vcombine_u8(g0, g1),
                vcombine_u8(b0, b1),
            );
            if BPP == 3 {
                vst3q_u8(out.add(3 * i), uint8x16x3_t(r, g, b));
            } else {
                vst4q_u8(out.add(4 * i), uint8x16x4_t(r, g, b, vdupq_n_u8(255)));
            }
            i += 16;
        }
        while i + 8 <= n {
            let [r, g, b] = yuv444_to_rgb(y.add(i), u.add(i), v.add(i));
            if BPP == 3 {
                vst3_u8(out.add(3 * i), uint8x8x3_t(r, g, b));
            } else {
                vst4_u8(out.add(4 * i), uint8x8x4_t(r, g, b, vdup_n_u8(255)));
            }
            i += 8;
        }
    }
    i
}

/// A 4x4 transpose of 16-bit lanes.
#[inline(always)]
unsafe fn transpose4x4(r: [int16x4_t; 4]) -> [int16x4_t; 4] {
    let a = vtrn_s16(r[0], r[1]);
    let b = vtrn_s16(r[2], r[3]);
    let c = vtrn_s32(vreinterpret_s32_s16(a.0), vreinterpret_s32_s16(b.0));
    let d = vtrn_s32(vreinterpret_s32_s16(a.1), vreinterpret_s32_s16(b.1));
    [
        vreinterpret_s16_s32(c.0),
        vreinterpret_s16_s32(d.0),
        vreinterpret_s16_s32(c.1),
        vreinterpret_s16_s32(d.1),
    ]
}

/// One pass of libwebp's `TransformPass_NEON`, saturating: `x * 35468 >> 16` is `vqdmulh` by
/// 17734, and `x + (x * 20091 >> 16)` adds half of `vqdmulh` by 20091.
#[inline(always)]
unsafe fn transform_pass(i: [int16x4_t; 4]) -> [int16x4_t; 4] {
    let mul1 = |x: int16x4_t| vsra_n_s16::<1>(x, vqdmulh_n_s16(x, 20091));
    let mul2 = |x: int16x4_t| vqdmulh_n_s16(x, 17734);
    let a = vqadd_s16(i[0], i[2]);
    let b = vqsub_s16(i[0], i[2]);
    let c = vqsub_s16(mul2(i[1]), mul1(i[3]));
    let d = vqadd_s16(mul1(i[1]), mul2(i[3]));
    [
        vqadd_s16(a, d),
        vqadd_s16(b, c),
        vqsub_s16(b, c),
        vqsub_s16(a, d),
    ]
}

#[inline(always)]
unsafe fn load4(p: *const u8) -> uint8x8_t {
    vreinterpret_u8_u32(vdup_n_u32(p.cast::<u32>().read_unaligned()))
}

#[inline(always)]
unsafe fn store4(p: *mut u8, x: uint8x8_t) {
    p.cast::<u32>()
        .write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(x)));
}

/// `t` added to the 4 pixels at `p`, saturated to 8 bits.
#[inline(always)]
unsafe fn add_row(p: *mut u8, t: int16x4_t) {
    let d = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(load4(p))));
    let s = vadd_s16(d, t);
    store4(p, vqmovun_s16(vcombine_s16(s, s)));
}

/// libwebp's `TransformOne_NEON`: the inverse transform of `block` (16 coefficients that fit 16
/// bits) in saturating 16 bits, added to the 4x4 pixels at `pos` of `dst`.
pub(crate) fn transform_add(block: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    let block = &block[..16];
    check(dst.len(), pos, 0, 3 * stride + 4);
    unsafe {
        let row = |r: usize| vld1_s16(block.as_ptr().add(4 * r));
        let t = transpose4x4(transform_pass([row(0), row(1), row(2), row(3)]));
        let t = transpose4x4(transform_pass(t));
        let p = dst.as_mut_ptr().add(pos);
        for (k, t) in t.into_iter().enumerate() {
            add_row(p.add(k * stride), vrshr_n_s16::<3>(t));
        }
    }
}

/// libwebp's `TransformAC3_NEON`: the inverse transform of `block`, whose only coefficients are
/// 0, 1 and 4, in saturating 16 bits, added to the 4x4 pixels at `pos` of `dst`.
pub(crate) fn transform_ac3_add(block: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    let block = &block[..16];
    check(dst.len(), pos, 0, 3 * stride + 4);
    let mul1 = |a: i16| (((i32::from(a) * 20091) >> 16) + i32::from(a)) as i16;
    let mul2 = |a: i16| ((i32::from(a) * 35468) >> 16) as i16;
    let (c1, d1) = (mul2(block[1]), mul1(block[1]));
    let cd = [d1, c1, c1.wrapping_neg(), d1.wrapping_neg()];
    unsafe {
        let b = vqadd_s16(vdup_n_s16(block[0]), vld1_s16(cd.as_ptr()));
        let (c4, d4) = (vdup_n_s16(mul2(block[4])), vdup_n_s16(mul1(block[4])));
        let rows = [
            vqadd_s16(b, d4),
            vqadd_s16(b, c4),
            vqsub_s16(b, c4),
            vqsub_s16(b, d4),
        ];
        let p = dst.as_mut_ptr().add(pos);
        for (k, t) in rows.into_iter().enumerate() {
            add_row(p.add(k * stride), vrshr_n_s16::<3>(t));
        }
    }
}

/// The interpolated pixels `out[2 * j + 1]` and `out[2 * j + 2]` of the leading chroma pairs
/// `j` of `upsample_chroma_row`, 16 or 8 pairs at a time in 8 bits as libwebp's
/// `UPSAMPLE_16PIXELS`. Returns how many pairs it did.
pub(crate) fn upsample_pairs(near: &[u8], far: &[u8], out: &mut [u8]) -> usize {
    let n = near.len().min(far.len());
    let pairs = (out.len().saturating_sub(1) / 2).min(n.saturating_sub(1));
    let mut j = 0;
    unsafe {
        let one = vdupq_n_u8(1);
        let pairs16 = |a: uint8x16_t, b: uint8x16_t, c: uint8x16_t, d: uint8x16_t| {
            let s = vrhaddq_u8(a, d);
            let t = vrhaddq_u8(b, c);
            let st = veorq_u8(s, t);
            let ad = veorq_u8(a, d);
            let bc = veorq_u8(b, c);
            let lsb = vandq_u8(vorrq_u8(vorrq_u8(ad, bc), st), one);
            let k = vsubq_u8(vrhaddq_u8(s, t), lsb);
            let get_m = |ij: uint8x16_t, x: uint8x16_t| {
                let lsb = vandq_u8(vorrq_u8(vandq_u8(ij, st), veorq_u8(k, x)), one);
                vsubq_u8(vrhaddq_u8(k, x), lsb)
            };
            uint8x16x2_t(vrhaddq_u8(a, get_m(bc, t)), vrhaddq_u8(b, get_m(ad, s)))
        };
        let (near, far, out) = (near.as_ptr(), far.as_ptr(), out.as_mut_ptr());
        while j + 16 <= pairs && j + 17 <= n {
            let eo = pairs16(
                load16(near.add(j)),
                load16(near.add(j + 1)),
                load16(far.add(j)),
                load16(far.add(j + 1)),
            );
            vst2q_u8(out.add(2 * j + 1), eo);
            j += 16;
        }
        if j + 8 <= pairs && j + 9 <= n {
            let half = |p: *const u8| vcombine_u8(vld1_u8(p), vdup_n_u8(0));
            let eo = pairs16(
                half(near.add(j)),
                half(near.add(j + 1)),
                half(far.add(j)),
                half(far.add(j + 1)),
            );
            vst2_u8(
                out.add(2 * j + 1),
                uint8x8x2_t(vget_low_u8(eo.0), vget_low_u8(eo.1)),
            );
            j += 8;
        }
    }
    j
}

/// libwebp's `DC16`, `DC8uv` and their `NoTop`, `NoLeft` and `NoTopLeft` variants: a
/// `SIZE`x`SIZE` block (8 or 16) at `p` set to the rounded mean of the row above and the
/// column to the left, as far as they are given.
pub(crate) fn dc<const SIZE: usize>(
    buf: &mut [u8],
    p: usize,
    stride: usize,
    top: bool,
    left: bool,
) {
    check(buf.len(), p, stride + 1, (SIZE - 1) * stride + SIZE);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let shift = SIZE.trailing_zeros() - 1;
        let mut sum = 0;
        if top {
            sum += if SIZE == 8 {
                u32::from(vaddlv_u8(vld1_u8(dst.sub(stride))))
            } else {
                u32::from(vaddlvq_u8(load16(dst.sub(stride))))
            };
        }
        if left {
            for y in 0..SIZE {
                sum += u32::from(*dst.add(y * stride).sub(1));
            }
        }
        let dc = match (top, left) {
            (true, true) => (sum + SIZE as u32) >> (shift + 2),
            (false, false) => 128,
            _ => (sum + SIZE as u32 / 2) >> (shift + 1),
        };
        let v = vdupq_n_u8(dc as u8);
        for y in 0..SIZE {
            if SIZE == 8 {
                vst1_u8(dst.add(y * stride), vget_low_u8(v));
            } else {
                store16(dst.add(y * stride), v);
            }
        }
    }
}

/// libwebp's `TransformDC`: `dc` (the DC coefficient plus rounding, shifted) added to the
/// 4x4 pixels at `pos` of `dst`.
pub(crate) fn add_dc(dc: i32, dst: &mut [u8], pos: usize, stride: usize) {
    check(dst.len(), pos, 0, 3 * stride + 4);
    unsafe {
        let dc = vdup_n_s16(dc as i16);
        let p = dst.as_mut_ptr().add(pos);
        for k in 0..4 {
            add_row(p.add(k * stride), dc);
        }
    }
}

/// `(a + 2 * b + c + 2) >> 2` per byte, as libwebp's predictors compute it.
#[inline(always)]
unsafe fn avg3(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    vrhadd_u8(vhadd_u8(a, c), b)
}

/// The bytes of `x` from `N` on, zeros shifted in.
#[inline(always)]
unsafe fn shr<const N: i32>(x: uint8x8_t) -> uint8x8_t {
    vext_u8::<N>(x, vdup_n_u8(0))
}

/// Checks the bounds of a 4x4 prediction at `p` that reads the row above from one pixel to
/// its left on and 8 bytes wide.
#[inline(always)]
fn check4(len: usize, p: usize, stride: usize) {
    check(len, p, stride + 1, 3 * stride + 4);
}

/// libwebp's `VE4`.
pub(crate) fn ve4(buf: &mut [u8], p: usize, stride: usize) {
    check4(buf.len(), p, stride);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let t = vld1_u8(dst.sub(stride + 1));
        let avg = avg3(t, shr::<1>(t), shr::<2>(t));
        for y in 0..4 {
            store4(dst.add(y * stride), avg);
        }
    }
}

/// libwebp's `LD4`.
pub(crate) fn ld4(buf: &mut [u8], p: usize, stride: usize) {
    check4(buf.len(), p, stride);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let top = dst.sub(stride);
        let t = vld1_u8(top);
        let c = vext_u8::<2>(t, vdup_n_u8(*top.add(7)));
        let avg = avg3(t, shr::<1>(t), c);
        store4(dst, avg);
        store4(dst.add(stride), shr::<1>(avg));
        store4(dst.add(2 * stride), shr::<2>(avg));
        store4(dst.add(3 * stride), shr::<3>(avg));
    }
}

/// libwebp's `RD4`.
pub(crate) fn rd4(buf: &mut [u8], p: usize, stride: usize) {
    check4(buf.len(), p, stride);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let left = |y: usize| u32::from(*dst.add(y * stride).sub(1));
        let lkji = left(3) | (left(2) << 8) | (left(1) << 16) | (left(0) << 24);
        // L K J I X A B C D ...
        let xabcd = vld1_u8(dst.sub(stride + 1));
        let e = vcombine_u8(vreinterpret_u8_u32(vdup_n_u32(lkji)), xabcd);
        let e = vextq_u8::<4>(e, e);
        let a = vrhaddq_u8(vhaddq_u8(e, vextq_u8::<2>(e, e)), vextq_u8::<1>(e, e));
        let lane = |x: uint8x16_t| vget_low_u8(x);
        store4(dst.add(3 * stride), lane(a));
        store4(dst.add(2 * stride), lane(vextq_u8::<1>(a, a)));
        store4(dst.add(stride), lane(vextq_u8::<2>(a, a)));
        store4(dst, lane(vextq_u8::<3>(a, a)));
    }
}

/// libwebp's `VR4`.
pub(crate) fn vr4(buf: &mut [u8], p: usize, stride: usize) {
    check4(buf.len(), p, stride);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let (i, j, k) = (
            *dst.sub(1),
            *dst.add(stride).sub(1),
            *dst.add(2 * stride).sub(1),
        );
        let x = *dst.sub(stride + 1);
        let xabcd = vld1_u8(dst.sub(stride + 1));
        let abcd0 = shr::<1>(xabcd);
        let abcd = vrhadd_u8(xabcd, abcd0);
        let ixabcd = vext_u8::<7>(vdup_n_u8(i), xabcd);
        let efgh = avg3(ixabcd, xabcd, abcd0);
        let zero = vdup_n_u8(0);
        store4(dst, abcd);
        store4(dst.add(stride), efgh);
        store4(dst.add(2 * stride), vext_u8::<7>(zero, abcd));
        store4(dst.add(3 * stride), vext_u8::<7>(zero, efgh));
        let avg3_1 =
            |a: u8, b: u8, c: u8| ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8;
        *dst.add(2 * stride) = avg3_1(j, i, x);
        *dst.add(3 * stride) = avg3_1(k, j, i);
    }
}

/// libwebp's `VL4`.
pub(crate) fn vl4(buf: &mut [u8], p: usize, stride: usize) {
    check4(buf.len(), p, stride);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let one = vdup_n_u8(1);
        let a = vld1_u8(dst.sub(stride));
        let b = shr::<1>(a);
        let c = shr::<2>(a);
        let avg1 = vrhadd_u8(a, b);
        let avg2 = vrhadd_u8(c, b);
        let avg3 = vrhadd_u8(avg1, avg2);
        let lsb1 = vand_u8(veor_u8(avg1, avg2), one);
        let abbc = vorr_u8(veor_u8(a, b), veor_u8(c, b));
        let avg4 = vqsub_u8(avg3, vand_u8(abbc, lsb1));
        store4(dst, avg1);
        store4(dst.add(stride), avg4);
        store4(dst.add(2 * stride), shr::<1>(avg1));
        store4(dst.add(3 * stride), shr::<1>(avg4));
        *dst.add(2 * stride + 3) = vget_lane_u8::<4>(avg4);
        *dst.add(3 * stride + 3) = vget_lane_u8::<5>(avg4);
    }
}

/// libwebp's `TrueMotion` of a `SIZE`x`SIZE` block (4, 8 or 16) at `p`.
pub(crate) fn tm<const SIZE: usize>(buf: &mut [u8], p: usize, stride: usize) {
    check(buf.len(), p, stride + 1, (SIZE - 1) * stride + SIZE);
    unsafe {
        let mut dst = buf.as_mut_ptr().add(p);
        let top = dst.sub(stride);
        let corner = i16::from(*top.sub(1));
        let t = match SIZE {
            4 => vcombine_u8(load4(top), vdup_n_u8(0)),
            8 => vcombine_u8(vld1_u8(top), vdup_n_u8(0)),
            _ => load16(top),
        };
        let t0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t)));
        let t1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t)));
        for _ in 0..SIZE {
            let base = vdupq_n_s16(i16::from(*dst.sub(1)) - corner);
            let lo = vqmovun_s16(vaddq_s16(base, t0));
            match SIZE {
                4 => store4(dst, lo),
                8 => vst1_u8(dst, lo),
                _ => store16(dst, vcombine_u8(lo, vqmovun_s16(vaddq_s16(base, t1)))),
            }
            dst = dst.add(stride);
        }
    }
}

/// The gradient alpha unfiltering of `row[x..]` (`row[x - 1]` done), 16 pixels at a time as
/// a running sum of the residuals and the top row's steps, for as long as no prediction
/// needs clamping. Returns where it stopped: at a block that clamps or with fewer than 16
/// pixels left.
pub(crate) fn unfilter_gradient(prev: &[u8], row: &mut [u8], mut x: usize) -> usize {
    let n = row.len().min(prev.len());
    assert!(x >= 1);
    // SAFETY: the loads read prev[x - 1..x + 16] and row[x - 1..x + 16] and the store writes
    // row[x..x + 16], in bounds for x + 16 <= n.
    unsafe {
        let zero = vdupq_n_u8(0);
        let max = vdupq_n_u16(255);
        let mut left = vld1q_dup_u8(row.as_ptr().add(x - 1));
        while x + 16 <= n {
            let top = vld1q_u8(prev.as_ptr().add(x));
            let top_left = vld1q_u8(prev.as_ptr().add(x - 1));
            let mut s = vaddq_u8(vld1q_u8(row.as_ptr().add(x)), vsubq_u8(top, top_left));
            s = vaddq_u8(s, vextq_u8::<15>(zero, s));
            s = vaddq_u8(s, vextq_u8::<14>(zero, s));
            s = vaddq_u8(s, vextq_u8::<12>(zero, s));
            s = vaddq_u8(s, vextq_u8::<8>(zero, s));
            let out = vaddq_u8(s, left);
            // The unclamped predictions, from the left pixels this block would produce.
            let before = vextq_u8::<15>(left, out);
            let pred = |b: uint8x8_t, t: uint8x8_t, tl: uint8x8_t| vsubw_u8(vaddl_u8(b, t), tl);
            let lo = pred(vget_low_u8(before), vget_low_u8(top), vget_low_u8(top_left));
            let hi = pred(
                vget_high_u8(before),
                vget_high_u8(top),
                vget_high_u8(top_left),
            );
            if vmaxvq_u16(vorrq_u16(vcgtq_u16(lo, max), vcgtq_u16(hi, max))) != 0 {
                break;
            }
            vst1q_u8(row.as_mut_ptr().add(x), out);
            left = vdupq_laneq_u8::<15>(out);
            x += 16;
        }
    }
    x
}

/// The horizontal alpha unfiltering of `row[x..]` (`row[x - 1]` done), a running sum 16
/// pixels at a time. Returns where it stopped, with fewer than 16 pixels left.
pub(crate) fn unfilter_horizontal(row: &mut [u8], mut x: usize) -> usize {
    assert!(x >= 1);
    // SAFETY: the loads read row[x - 1..x + 16] and the store writes row[x..x + 16], in bounds
    // for x + 16 <= row.len().
    unsafe {
        let zero = vdupq_n_u8(0);
        let mut left = vld1q_dup_u8(row.as_ptr().add(x - 1));
        while x + 16 <= row.len() {
            let mut s = vld1q_u8(row.as_ptr().add(x));
            s = vaddq_u8(s, vextq_u8::<15>(zero, s));
            s = vaddq_u8(s, vextq_u8::<14>(zero, s));
            s = vaddq_u8(s, vextq_u8::<12>(zero, s));
            s = vaddq_u8(s, vextq_u8::<8>(zero, s));
            let out = vaddq_u8(s, left);
            vst1q_u8(row.as_mut_ptr().add(x), out);
            left = vdupq_laneq_u8::<15>(out);
            x += 16;
        }
    }
    x
}
