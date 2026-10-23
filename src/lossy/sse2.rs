//! SSE2 decoder kernels ported from libwebp's `dsp/dec_sse2.c`, bit-exact with the portable
//! ones. SSE2 is part of x86_64, so no runtime detection is needed. Each wrapper checks the
//! bounds its loads and stores rely on; `p` is the offset of the first pixel past the edge.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

#[inline(always)]
fn check(len: usize, p: usize, back: usize, fwd: usize) {
    assert!(p >= back && p.checked_add(fwd).is_some_and(|end| end <= len));
}

#[inline(always)]
unsafe fn abs_diff(p: __m128i, q: __m128i) -> __m128i {
    _mm_or_si128(_mm_subs_epu8(q, p), _mm_subs_epu8(p, q))
}

#[inline(always)]
unsafe fn flip(x: __m128i) -> __m128i {
    _mm_xor_si128(x, _mm_set1_epi8(-128))
}

#[inline(always)]
unsafe fn signed_shift8b(x: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let lo = _mm_srai_epi16::<11>(_mm_unpacklo_epi8(zero, x));
    let hi = _mm_srai_epi16::<11>(_mm_unpackhi_epi8(zero, x));
    _mm_packs_epi16(lo, hi)
}

#[inline(always)]
unsafe fn not_hev(p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i, hev_thresh: u8) -> __m128i {
    let t_max = _mm_max_epu8(abs_diff(p1, p0), abs_diff(q1, q0));
    let t_max_h = _mm_subs_epu8(t_max, _mm_set1_epi8(hev_thresh as i8));
    _mm_cmpeq_epi8(t_max_h, _mm_setzero_si128())
}

#[inline(always)]
unsafe fn base_delta(p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i) -> __m128i {
    let p1_q1 = _mm_subs_epi8(p1, q1);
    let q0_p0 = _mm_subs_epi8(q0, p0);
    let s1 = _mm_adds_epi8(p1_q1, q0_p0);
    let s2 = _mm_adds_epi8(q0_p0, s1);
    _mm_adds_epi8(q0_p0, s2)
}

#[inline(always)]
unsafe fn do_simple_filter(p0: &mut __m128i, q0: &mut __m128i, fl: __m128i) {
    let v3 = signed_shift8b(_mm_adds_epi8(fl, _mm_set1_epi8(3)));
    let v4 = signed_shift8b(_mm_adds_epi8(fl, _mm_set1_epi8(4)));
    *q0 = _mm_subs_epi8(*q0, v4);
    *p0 = _mm_adds_epi8(*p0, v3);
}

#[inline(always)]
unsafe fn update_2_pixels(pi: &mut __m128i, qi: &mut __m128i, a0_lo: __m128i, a0_hi: __m128i) {
    let delta = _mm_packs_epi16(_mm_srai_epi16::<7>(a0_lo), _mm_srai_epi16::<7>(a0_hi));
    *pi = flip(_mm_adds_epi8(*pi, delta));
    *qi = flip(_mm_subs_epi8(*qi, delta));
}

#[inline(always)]
unsafe fn needs_filter(p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i, thresh: u8) -> __m128i {
    let t3 = _mm_srli_epi16::<1>(_mm_and_si128(abs_diff(p1, q1), _mm_set1_epi8(-2)));
    let t4 = abs_diff(p0, q0);
    let t6 = _mm_adds_epu8(_mm_adds_epu8(t4, t4), t3);
    let t7 = _mm_subs_epu8(t6, _mm_set1_epi8(thresh as i8));
    _mm_cmpeq_epi8(t7, _mm_setzero_si128())
}

#[inline(always)]
unsafe fn do_filter2(p1: __m128i, p0: &mut __m128i, q0: &mut __m128i, q1: __m128i, thresh: u8) {
    let mask = needs_filter(p1, *p0, *q0, q1, thresh);
    let (mut sp0, mut sq0) = (flip(*p0), flip(*q0));
    let a = _mm_and_si128(base_delta(flip(p1), sp0, sq0, flip(q1)), mask);
    do_simple_filter(&mut sp0, &mut sq0, a);
    *p0 = flip(sp0);
    *q0 = flip(sq0);
}

#[inline(always)]
unsafe fn do_filter4(
    p1: &mut __m128i,
    p0: &mut __m128i,
    q0: &mut __m128i,
    q1: &mut __m128i,
    mask: __m128i,
    hev_thresh: u8,
) {
    let not_hev = not_hev(*p1, *p0, *q0, *q1, hev_thresh);
    let (sp1, sp0, sq0, sq1) = (flip(*p1), flip(*p0), flip(*q0), flip(*q1));
    let mut t1 = _mm_andnot_si128(not_hev, _mm_subs_epi8(sp1, sq1));
    let t2 = _mm_subs_epi8(sq0, sp0);
    t1 = _mm_adds_epi8(t1, t2);
    t1 = _mm_adds_epi8(t1, t2);
    t1 = _mm_adds_epi8(t1, t2);
    t1 = _mm_and_si128(t1, mask);
    let t2 = signed_shift8b(_mm_adds_epi8(t1, _mm_set1_epi8(3)));
    let t3 = signed_shift8b(_mm_adds_epi8(t1, _mm_set1_epi8(4)));
    *p0 = flip(_mm_adds_epi8(sp0, t2));
    *q0 = flip(_mm_subs_epi8(sq0, t3));
    // signed (t3 + 1) >> 1
    let t3 = _mm_sub_epi8(
        _mm_avg_epu8(_mm_add_epi8(t3, _mm_set1_epi8(-128)), _mm_setzero_si128()),
        _mm_set1_epi8(64),
    );
    let t3 = _mm_and_si128(not_hev, t3);
    *q1 = flip(_mm_subs_epi8(sq1, t3));
    *p1 = flip(_mm_adds_epi8(sp1, t3));
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn do_filter6(
    p2: &mut __m128i,
    p1: &mut __m128i,
    p0: &mut __m128i,
    q0: &mut __m128i,
    q1: &mut __m128i,
    q2: &mut __m128i,
    mask: __m128i,
    hev_thresh: u8,
) {
    let zero = _mm_setzero_si128();
    let not_hev = not_hev(*p1, *p0, *q0, *q1, hev_thresh);
    let (sp2, sp1, mut sp0, mut sq0, sq1, sq2) = (
        flip(*p2),
        flip(*p1),
        flip(*p0),
        flip(*q0),
        flip(*q1),
        flip(*q2),
    );
    let a = base_delta(sp1, sp0, sq0, sq1);
    do_simple_filter(
        &mut sp0,
        &mut sq0,
        _mm_and_si128(a, _mm_andnot_si128(not_hev, mask)),
    );

    let k9 = _mm_set1_epi16(0x0900);
    let k63 = _mm_set1_epi16(63);
    let f = _mm_and_si128(a, _mm_and_si128(not_hev, mask));
    let f9_lo = _mm_mulhi_epi16(_mm_unpacklo_epi8(zero, f), k9);
    let f9_hi = _mm_mulhi_epi16(_mm_unpackhi_epi8(zero, f), k9);
    let a2_lo = _mm_add_epi16(f9_lo, k63);
    let a2_hi = _mm_add_epi16(f9_hi, k63);
    let a1_lo = _mm_add_epi16(a2_lo, f9_lo);
    let a1_hi = _mm_add_epi16(a2_hi, f9_hi);
    let a0_lo = _mm_add_epi16(a1_lo, f9_lo);
    let a0_hi = _mm_add_epi16(a1_hi, f9_hi);
    (*p2, *q2, *p1, *q1, *p0, *q0) = (sp2, sq2, sp1, sq1, sp0, sq0);
    update_2_pixels(p2, q2, a2_lo, a2_hi);
    update_2_pixels(p1, q1, a1_lo, a1_hi);
    update_2_pixels(p0, q0, a0_lo, a0_hi);
}

#[inline(always)]
pub(super) unsafe fn rd32(b: *const u8) -> i32 {
    b.cast::<i32>().read_unaligned()
}

/// 8 rows of 4 pixels from `b`, transposed: columns 0 and 1 in `p`, 2 and 3 in `q`.
#[inline(always)]
unsafe fn load8x4(b: *const u8, stride: usize) -> (__m128i, __m128i) {
    let a0 = _mm_set_epi32(
        rd32(b.add(6 * stride)),
        rd32(b.add(2 * stride)),
        rd32(b.add(4 * stride)),
        rd32(b),
    );
    let a1 = _mm_set_epi32(
        rd32(b.add(7 * stride)),
        rd32(b.add(3 * stride)),
        rd32(b.add(5 * stride)),
        rd32(b.add(stride)),
    );
    let b0 = _mm_unpacklo_epi8(a0, a1);
    let b1 = _mm_unpackhi_epi8(a0, a1);
    let c0 = _mm_unpacklo_epi16(b0, b1);
    let c1 = _mm_unpackhi_epi16(b0, b1);
    (_mm_unpacklo_epi32(c0, c1), _mm_unpackhi_epi32(c0, c1))
}

/// Columns 0..4 of 8 rows from `r0` and 8 rows from `r8`, one register per column.
#[inline(always)]
pub(super) unsafe fn load16x4(r0: *const u8, r8: *const u8, stride: usize) -> [__m128i; 4] {
    let (t1, t2) = load8x4(r0, stride);
    let (p0, q1) = load8x4(r8, stride);
    [
        _mm_unpacklo_epi64(t1, p0),
        _mm_unpackhi_epi64(t1, p0),
        _mm_unpacklo_epi64(t2, q1),
        _mm_unpackhi_epi64(t2, q1),
    ]
}

#[inline(always)]
pub(super) unsafe fn store4x4(mut x: __m128i, dst: *mut u8, stride: usize) {
    for i in 0..4 {
        dst.add(i * stride)
            .cast::<i32>()
            .write_unaligned(_mm_cvtsi128_si32(x));
        x = _mm_srli_si128::<4>(x);
    }
}

#[inline(always)]
pub(super) unsafe fn store16x4(c: [__m128i; 4], r0: *mut u8, r8: *mut u8, stride: usize) {
    let [p1, p0, q0, q1] = c;
    let p0_s = _mm_unpacklo_epi8(p1, p0);
    let p1_s = _mm_unpackhi_epi8(p1, p0);
    let q0_s = _mm_unpacklo_epi8(q0, q1);
    let q1_s = _mm_unpackhi_epi8(q0, q1);
    store4x4(_mm_unpacklo_epi16(p0_s, q0_s), r0, stride);
    store4x4(_mm_unpackhi_epi16(p0_s, q0_s), r0.add(4 * stride), stride);
    store4x4(_mm_unpacklo_epi16(p1_s, q1_s), r8, stride);
    store4x4(_mm_unpackhi_epi16(p1_s, q1_s), r8.add(4 * stride), stride);
}

#[inline(always)]
pub(super) unsafe fn load16(b: *const u8) -> __m128i {
    _mm_loadu_si128(b.cast())
}

#[inline(always)]
pub(super) unsafe fn store16(b: *mut u8, x: __m128i) {
    _mm_storeu_si128(b.cast(), x)
}

#[inline(always)]
pub(super) unsafe fn load_uv(u: *const u8, v: *const u8) -> __m128i {
    _mm_unpacklo_epi64(_mm_loadl_epi64(u.cast()), _mm_loadl_epi64(v.cast()))
}

#[inline(always)]
pub(super) unsafe fn store_uv(x: __m128i, u: *mut u8, v: *mut u8) {
    _mm_storel_epi64(u.cast(), x);
    _mm_storel_epi64(v.cast(), _mm_srli_si128::<8>(x));
}

#[inline(always)]
unsafe fn max_diff(m: __m128i, p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i) -> __m128i {
    let m = _mm_max_epu8(m, abs_diff(p1, p0));
    let m = _mm_max_epu8(m, abs_diff(p3, p2));
    _mm_max_epu8(m, abs_diff(p2, p1))
}

#[inline(always)]
unsafe fn complex_mask(
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    thresh: u8,
    ithresh: u8,
    mask: __m128i,
) -> __m128i {
    let diff = _mm_subs_epu8(mask, _mm_set1_epi8(ithresh as i8));
    let thresh_mask = _mm_cmpeq_epi8(diff, _mm_setzero_si128());
    _mm_and_si128(thresh_mask, needs_filter(p1, p0, q0, q1, thresh))
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
    check(buf.len(), p, 2, 15 * stride + 2);
    unsafe {
        let b = buf.as_mut_ptr().add(p - 2);
        let [p1, mut p0, mut q0, q1] = load16x4(b, b.add(8 * stride), stride);
        do_filter2(p1, &mut p0, &mut q0, q1, thresh);
        store16x4([p1, p0, q0, q1], b, b.add(8 * stride), stride);
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
        let (t1, mut p2, mut p1, mut p0) = (
            load16(b.sub(4 * stride)),
            load16(b.sub(3 * stride)),
            load16(b.sub(2 * stride)),
            load16(b.sub(stride)),
        );
        let mask = max_diff(abs_diff(p1, p0), t1, p2, p1, p0);
        let (mut q0, mut q1, mut q2, t1) = (
            load16(b),
            load16(b.add(stride)),
            load16(b.add(2 * stride)),
            load16(b.add(3 * stride)),
        );
        let mask = max_diff(mask, t1, q2, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
        );
        store16(b.sub(3 * stride), p2);
        store16(b.sub(2 * stride), p1);
        store16(b.sub(stride), p0);
        store16(b, q0);
        store16(b.add(stride), q1);
        store16(b.add(2 * stride), q2);
    }
}

pub(crate) fn h16(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    check(buf.len(), p, 4, 15 * stride + 4);
    unsafe {
        let q = buf.as_mut_ptr().add(p);
        let b = q.sub(4);
        let [p3, mut p2, mut p1, mut p0] = load16x4(b, b.add(8 * stride), stride);
        let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
        let [mut q0, mut q1, mut q2, q3] = load16x4(q, q.add(8 * stride), stride);
        let mask = max_diff(mask, q3, q2, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
        );
        store16x4([p3, p2, p1, p0], b, b.add(8 * stride), stride);
        store16x4([q0, q1, q2, q3], q, q.add(8 * stride), stride);
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
            let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
            p3 = load16(b);
            p2 = load16(b.add(stride));
            let (tmp1, tmp2) = (load16(b.add(2 * stride)), load16(b.add(3 * stride)));
            let mask = max_diff(mask, p3, p2, tmp1, tmp2);
            let mask = complex_mask(p1, p0, p3, p2, thresh, ithresh, mask);
            do_filter4(&mut p1, &mut p0, &mut p3, &mut p2, mask, hev);
            store16(d, p1);
            store16(d.add(stride), p0);
            store16(d.add(2 * stride), p3);
            store16(d.add(3 * stride), p2);
            p1 = tmp1;
            p0 = tmp2;
        }
    }
}

pub(crate) fn h16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    check(buf.len(), p, 0, 15 * stride + 16);
    unsafe {
        let mut b = buf.as_mut_ptr().add(p);
        let [mut p3, mut p2, mut p1, mut p0] = load16x4(b, b.add(8 * stride), stride);
        for _ in 0..3 {
            let d = b.add(2);
            b = b.add(4);
            let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
            let [n3, n2, tmp1, tmp2] = load16x4(b, b.add(8 * stride), stride);
            (p3, p2) = (n3, n2);
            let mask = max_diff(mask, p3, p2, tmp1, tmp2);
            let mask = complex_mask(p1, p0, p3, p2, thresh, ithresh, mask);
            do_filter4(&mut p1, &mut p0, &mut p3, &mut p2, mask, hev);
            store16x4([p1, p0, p3, p2], d, d.add(8 * stride), stride);
            p1 = tmp1;
            p0 = tmp2;
        }
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
        let (t1, mut p2, mut p1, mut p0) = (row(-4), row(-3), row(-2), row(-1));
        let mask = max_diff(abs_diff(p1, p0), t1, p2, p1, p0);
        let (mut q0, mut q1, mut q2, t1) = (row(0), row(1), row(2), row(3));
        let mask = max_diff(mask, t1, q2, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
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
        let (u, v) = (u.as_mut_ptr().add(p), v.as_mut_ptr().add(p));
        let (tu, tv) = (u.sub(4), v.sub(4));
        let [p3, mut p2, mut p1, mut p0] = load16x4(tu, tv, stride);
        let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
        let [mut q0, mut q1, mut q2, q3] = load16x4(u, v, stride);
        let mask = max_diff(mask, q3, q2, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
        );
        store16x4([p3, p2, p1, p0], tu, tv, stride);
        store16x4([q0, q1, q2, q3], u, v, stride);
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
        let (t2, t1, mut p1, mut p0) = (row(0), row(1), row(2), row(3));
        let mask = max_diff(abs_diff(p1, p0), t2, t1, p1, p0);
        let (mut q0, mut q1, t1, t2) = (row(4), row(5), row(6), row(7));
        let mask = max_diff(mask, t2, t1, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
        do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev);
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
        let [t2, t1, mut p1, mut p0] = load16x4(u, v, stride);
        let mask = max_diff(abs_diff(p1, p0), t2, t1, p1, p0);
        let [mut q0, mut q1, t1, t2] = load16x4(u.add(4), v.add(4), stride);
        let mask = max_diff(mask, t2, t1, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
        do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev);
        store16x4([p1, p0, q0, q1], u.add(2), v.add(2), stride);
    }
}

#[inline(always)]
unsafe fn load_hi16(p: *const u8) -> __m128i {
    _mm_unpacklo_epi8(_mm_setzero_si128(), _mm_loadl_epi64(p.cast()))
}

/// libwebp's `YUV444ToRGB_SSE2`: 8 pixels to 16-bit R, G, B (to be saturated to 8 bits).
#[inline(always)]
unsafe fn yuv444_to_rgb(y: *const u8, u: *const u8, v: *const u8) -> [__m128i; 3] {
    let (y0, u0, v0) = (load_hi16(y), load_hi16(u), load_hi16(v));
    let y1 = _mm_mulhi_epu16(y0, _mm_set1_epi16(19077));
    let r0 = _mm_mulhi_epu16(v0, _mm_set1_epi16(26149));
    let r2 = _mm_add_epi16(_mm_sub_epi16(y1, _mm_set1_epi16(14234)), r0);
    let g0 = _mm_mulhi_epu16(u0, _mm_set1_epi16(6419));
    let g1 = _mm_mulhi_epu16(v0, _mm_set1_epi16(13320));
    let g4 = _mm_sub_epi16(
        _mm_add_epi16(y1, _mm_set1_epi16(8708)),
        _mm_add_epi16(g0, g1),
    );
    let b0 = _mm_mulhi_epu16(u0, _mm_set1_epi16(33050u16 as i16));
    let b2 = _mm_subs_epu16(_mm_adds_epu16(b0, y1), _mm_set1_epi16(17685));
    [
        _mm_srai_epi16::<6>(r2),
        _mm_srai_epi16::<6>(g4),
        _mm_srli_epi16::<6>(b2),
    ]
}

#[inline(always)]
unsafe fn planar_to_24b_step(i: [__m128i; 6]) -> [__m128i; 6] {
    let m = _mm_set1_epi16(0x00ff);
    let even = |a, b| _mm_packus_epi16(_mm_and_si128(a, m), _mm_and_si128(b, m));
    let odd = |a, b| _mm_packus_epi16(_mm_srli_epi16::<8>(a), _mm_srli_epi16::<8>(b));
    [
        even(i[0], i[1]),
        even(i[2], i[3]),
        even(i[4], i[5]),
        odd(i[0], i[1]),
        odd(i[2], i[3]),
        odd(i[4], i[5]),
    ]
}

/// RGB (`BPP` 3) or RGBA with alpha 255 (`BPP` 4) of the leading pixels of a row with
/// full-width chroma, 32 or 8 pixels at a time as libwebp's `VP8YuvToRgb32_SSE2` and
/// `VP8YuvToRgba32_SSE2`. Returns how many pixels it converted.
pub(crate) fn yuv_to_rgb_row<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
) -> usize {
    let mut i = 0;
    if crate::wide::has_avx2() {
        // SAFETY: the CPU has AVX2.
        i = unsafe { super::avx2::yuv_to_rgb_row::<BPP>(y, u, v, out) };
    }
    yuv_to_rgb_row_from::<BPP>(y, u, v, out, i)
}

/// `yuv_to_rgb_row` with the pixels before `i` already converted.
pub(crate) fn yuv_to_rgb_row_from<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
    mut i: usize,
) -> usize {
    let n = y.len().min(u.len()).min(v.len()).min(out.len() / BPP);
    unsafe {
        let (y, u, v, out) = (y.as_ptr(), u.as_ptr(), v.as_ptr(), out.as_mut_ptr());
        if BPP == 3 {
            while i + 32 <= n {
                let [r0, g0, b0] = yuv444_to_rgb(y.add(i), u.add(i), v.add(i));
                let [r1, g1, b1] = yuv444_to_rgb(y.add(i + 8), u.add(i + 8), v.add(i + 8));
                let [r2, g2, b2] = yuv444_to_rgb(y.add(i + 16), u.add(i + 16), v.add(i + 16));
                let [r3, g3, b3] = yuv444_to_rgb(y.add(i + 24), u.add(i + 24), v.add(i + 24));
                let mut c = [
                    _mm_packus_epi16(r0, r1),
                    _mm_packus_epi16(r2, r3),
                    _mm_packus_epi16(g0, g1),
                    _mm_packus_epi16(g2, g3),
                    _mm_packus_epi16(b0, b1),
                    _mm_packus_epi16(b2, b3),
                ];
                for _ in 0..5 {
                    c = planar_to_24b_step(c);
                }
                for (k, x) in c.into_iter().enumerate() {
                    store16(out.add(3 * i + 16 * k), x);
                }
                i += 32;
            }
        } else {
            let alpha = _mm_set1_epi16(255);
            while i + 8 <= n {
                let [r, g, b] = yuv444_to_rgb(y.add(i), u.add(i), v.add(i));
                let rb = _mm_packus_epi16(r, b);
                let ga = _mm_packus_epi16(g, alpha);
                let rg = _mm_unpacklo_epi8(rb, ga);
                let ba = _mm_unpackhi_epi8(rb, ga);
                store16(out.add(4 * i), _mm_unpacklo_epi16(rg, ba));
                store16(out.add(4 * i + 16), _mm_unpackhi_epi16(rg, ba));
                i += 8;
            }
        }
    }
    i
}

#[inline(always)]
unsafe fn transpose_2_4x4(i0: __m128i, i1: __m128i, i2: __m128i, i3: __m128i) -> [__m128i; 4] {
    let t0_0 = _mm_unpacklo_epi16(i0, i1);
    let t0_1 = _mm_unpacklo_epi16(i2, i3);
    let t0_2 = _mm_unpackhi_epi16(i0, i1);
    let t0_3 = _mm_unpackhi_epi16(i2, i3);
    let t1_0 = _mm_unpacklo_epi32(t0_0, t0_1);
    let t1_1 = _mm_unpacklo_epi32(t0_2, t0_3);
    let t1_2 = _mm_unpackhi_epi32(t0_0, t0_1);
    let t1_3 = _mm_unpackhi_epi32(t0_2, t0_3);
    [
        _mm_unpacklo_epi64(t1_0, t1_1),
        _mm_unpackhi_epi64(t1_0, t1_1),
        _mm_unpacklo_epi64(t1_2, t1_3),
        _mm_unpackhi_epi64(t1_2, t1_3),
    ]
}

/// The two passes of libwebp's `Transform_SSE2` on the coefficient rows of one or two blocks,
/// side by side: the rows of pixel differences.
#[inline(always)]
unsafe fn transform_rows(rows: [__m128i; 4]) -> [__m128i; 4] {
    let k1 = _mm_set1_epi16(20091);
    let k2 = _mm_set1_epi16(-30068);
    let pass = |i0: __m128i, i1: __m128i, i2: __m128i, i3: __m128i| {
        let a = _mm_add_epi16(i0, i2);
        let b = _mm_sub_epi16(i0, i2);
        let c = _mm_add_epi16(
            _mm_sub_epi16(i1, i3),
            _mm_sub_epi16(_mm_mulhi_epi16(i1, k2), _mm_mulhi_epi16(i3, k1)),
        );
        let d = _mm_add_epi16(
            _mm_add_epi16(i1, i3),
            _mm_add_epi16(_mm_mulhi_epi16(i1, k1), _mm_mulhi_epi16(i3, k2)),
        );
        [
            _mm_add_epi16(a, d),
            _mm_add_epi16(b, c),
            _mm_sub_epi16(b, c),
            _mm_sub_epi16(a, d),
        ]
    };
    let [r0, r1, r2, r3] = rows;
    let [t0, t1, t2, t3] = pass(r0, r1, r2, r3);
    let [t0, t1, t2, t3] = transpose_2_4x4(t0, t1, t2, t3);
    let [s0, s1, s2, s3] = pass(_mm_add_epi16(t0, _mm_set1_epi16(4)), t1, t2, t3);
    transpose_2_4x4(
        _mm_srai_epi16::<3>(s0),
        _mm_srai_epi16::<3>(s1),
        _mm_srai_epi16::<3>(s2),
        _mm_srai_epi16::<3>(s3),
    )
}

/// libwebp's `Transform_SSE2` for one block: the inverse transform of `block` (16
/// coefficients that fit 16 bits) added to the 4x4 pixels at `pos` of `dst`.
#[inline(always)]
pub(crate) fn transform_add(block: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    let block = &block[..16];
    check(dst.len(), pos, 0, 3 * stride + 4);
    unsafe {
        let row = |r: usize| _mm_loadl_epi64(block.as_ptr().add(4 * r).cast());
        let [t0, t1, t2, t3] = transform_rows([row(0), row(1), row(2), row(3)]);
        let zero = _mm_setzero_si128();
        let p = dst.as_mut_ptr().add(pos);
        let ld = |k: usize| _mm_cvtsi32_si128(p.add(k * stride).cast::<i32>().read_unaligned());
        let d01 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(ld(0), ld(1)), zero);
        let d23 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(ld(2), ld(3)), zero);
        // Two rows a pack: LLVM narrows a pack of which it keeps one half with extra clamps.
        let r = _mm_packus_epi16(
            _mm_add_epi16(d01, _mm_unpacklo_epi64(t0, t1)),
            _mm_add_epi16(d23, _mm_unpacklo_epi64(t2, t3)),
        );
        store4(p, r);
        store4(p.add(stride), _mm_srli_si128::<4>(r));
        store4(p.add(2 * stride), _mm_srli_si128::<8>(r));
        store4(p.add(3 * stride), _mm_srli_si128::<12>(r));
    }
}

/// libwebp's `Transform_SSE2` for two blocks side by side (`do_two`): the inverse transforms of
/// `blocks` (32 coefficients that fit 16 bits) added to the 8x4 pixels at `pos` of `dst`.
pub(crate) fn transform_add_two(blocks: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    let blocks = &blocks[..32];
    check(dst.len(), pos, 0, 3 * stride + 8);
    unsafe {
        let row = |r: usize| {
            let a = _mm_loadl_epi64(blocks.as_ptr().add(4 * r).cast());
            let b = _mm_loadl_epi64(blocks.as_ptr().add(16 + 4 * r).cast());
            _mm_unpacklo_epi64(a, b)
        };
        let t = transform_rows([row(0), row(1), row(2), row(3)]);
        let zero = _mm_setzero_si128();
        let mut p = dst.as_mut_ptr().add(pos);
        for t in t {
            let d = _mm_unpacklo_epi8(_mm_loadl_epi64(p.cast()), zero);
            let d = _mm_add_epi16(d, t);
            _mm_storel_epi64(p.cast(), _mm_packus_epi16(d, d));
            p = p.add(stride);
        }
    }
}

/// The interpolated pixels `out[2 * j + 1]` and `out[2 * j + 2]` of the leading chroma pairs
/// `j` of `upsample_chroma_row`, 8 pairs at a time. Returns how many pairs it did.
pub(crate) fn upsample_pairs(near: &[u8], far: &[u8], out: &mut [u8]) -> usize {
    let mut j = 0;
    if crate::wide::has_avx2() {
        // SAFETY: the CPU has AVX2.
        j = unsafe { super::avx2::upsample_pairs(near, far, out) };
    }
    upsample_pairs_from(near, far, out, j)
}

/// `upsample_pairs` with the pairs before `j` already done.
pub(crate) fn upsample_pairs_from(near: &[u8], far: &[u8], out: &mut [u8], mut j: usize) -> usize {
    let n = near.len().min(far.len());
    let pairs = (out.len().saturating_sub(1) / 2).min(n.saturating_sub(1));
    unsafe {
        let zero = _mm_setzero_si128();
        let k8 = _mm_set1_epi16(8);
        let mix = |k: usize| {
            let a = _mm_unpacklo_epi8(_mm_loadl_epi64(near.as_ptr().add(k).cast()), zero);
            let c = _mm_unpacklo_epi8(_mm_loadl_epi64(far.as_ptr().add(k).cast()), zero);
            _mm_add_epi16(_mm_add_epi16(a, _mm_add_epi16(a, a)), c)
        };
        // libwebp's `UPSAMPLE_32PIXELS` for the near row: 16 pairs in 8 bits.
        let one = _mm_set1_epi8(1);
        while j + 16 <= pairs && j + 17 <= n {
            let a = load16(near.as_ptr().add(j));
            let b = load16(near.as_ptr().add(j + 1));
            let c = load16(far.as_ptr().add(j));
            let d = load16(far.as_ptr().add(j + 1));
            let s = _mm_avg_epu8(a, d);
            let t = _mm_avg_epu8(b, c);
            let st = _mm_xor_si128(s, t);
            let ad = _mm_xor_si128(a, d);
            let bc = _mm_xor_si128(b, c);
            let lsb = _mm_and_si128(_mm_or_si128(_mm_or_si128(ad, bc), st), one);
            let k = _mm_sub_epi8(_mm_avg_epu8(s, t), lsb);
            let get_m = |ij: __m128i, x: __m128i| {
                let lsb = _mm_and_si128(
                    _mm_or_si128(_mm_and_si128(ij, st), _mm_xor_si128(k, x)),
                    one,
                );
                _mm_sub_epi8(_mm_avg_epu8(k, x), lsb)
            };
            let even = _mm_avg_epu8(a, get_m(bc, t));
            let odd = _mm_avg_epu8(b, get_m(ad, s));
            store16(
                out.as_mut_ptr().add(2 * j + 1),
                _mm_unpacklo_epi8(even, odd),
            );
            store16(
                out.as_mut_ptr().add(2 * j + 17),
                _mm_unpackhi_epi8(even, odd),
            );
            j += 16;
        }
        while j + 8 <= pairs && j + 9 <= n {
            let a = mix(j);
            let b = mix(j + 1);
            let a3 = _mm_add_epi16(a, _mm_add_epi16(a, a));
            let b3 = _mm_add_epi16(b, _mm_add_epi16(b, b));
            let e = _mm_srli_epi16::<4>(_mm_add_epi16(_mm_add_epi16(a3, b), k8));
            let o = _mm_srli_epi16::<4>(_mm_add_epi16(_mm_add_epi16(b3, a), k8));
            let eo = _mm_unpacklo_epi8(_mm_packus_epi16(e, e), _mm_packus_epi16(o, o));
            store16(out.as_mut_ptr().add(2 * j + 1), eo);
            j += 8;
        }
    }
    j
}

/// libwebp's `DC16_SSE2`, `DC8uv_SSE2` and their `NoTop`, `NoLeft` and `NoTopLeft` variants:
/// a `SIZE`x`SIZE` block (8 or 16) at `p` set to the rounded mean of the row above and the
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
            let t = if SIZE == 8 {
                _mm_loadl_epi64(dst.sub(stride).cast())
            } else {
                load16(dst.sub(stride))
            };
            let sad = _mm_sad_epu8(t, _mm_setzero_si128());
            sum += (_mm_cvtsi128_si32(sad) + _mm_extract_epi16::<4>(sad)) as u32;
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
        let v = _mm_set1_epi8(dc as i8);
        for y in 0..SIZE {
            if SIZE == 8 {
                _mm_storel_epi64(dst.add(y * stride).cast(), v);
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
        let dc = _mm_set1_epi16(dc as i16);
        let up = _mm_packus_epi16(dc, dc);
        let down = _mm_packus_epi16(_mm_sub_epi16(_mm_setzero_si128(), dc), dc);
        let mut p = dst.as_mut_ptr().add(pos);
        for _ in 0..4 {
            let d = _mm_cvtsi32_si128(p.cast::<i32>().read_unaligned());
            store4(p, _mm_subs_epu8(_mm_adds_epu8(d, up), down));
            p = p.add(stride);
        }
    }
}

/// `(a + 2 * b + c + 2) >> 2` per byte, as libwebp's predictors compute it.
#[inline(always)]
unsafe fn avg3(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    let lsb = _mm_and_si128(_mm_xor_si128(a, c), _mm_set1_epi8(1));
    _mm_avg_epu8(_mm_subs_epu8(_mm_avg_epu8(a, c), lsb), b)
}

#[inline(always)]
unsafe fn store4(dst: *mut u8, x: __m128i) {
    dst.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(x));
}

/// Checks the bounds of a 4x4 prediction at `p` that reads the row above from one pixel to
/// its left on and 8 bytes wide.
#[inline(always)]
fn check4(len: usize, p: usize, stride: usize) {
    check(len, p, stride + 1, 3 * stride + 4);
}

/// libwebp's `VE4_SSE2`.
pub(crate) fn ve4(buf: &mut [u8], p: usize, stride: usize) {
    check4(buf.len(), p, stride);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let t = _mm_loadl_epi64(dst.sub(stride + 1).cast());
        let avg = avg3(t, _mm_srli_si128::<1>(t), _mm_srli_si128::<2>(t));
        for y in 0..4 {
            store4(dst.add(y * stride), avg);
        }
    }
}

/// libwebp's `LD4_SSE2`.
pub(crate) fn ld4(buf: &mut [u8], p: usize, stride: usize) {
    check4(buf.len(), p, stride);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let top = dst.sub(stride);
        let t = _mm_loadl_epi64(top.cast());
        let c = _mm_insert_epi16::<3>(_mm_srli_si128::<2>(t), i32::from(*top.add(7)));
        let avg = avg3(t, _mm_srli_si128::<1>(t), c);
        store4(dst, avg);
        store4(dst.add(stride), _mm_srli_si128::<1>(avg));
        store4(dst.add(2 * stride), _mm_srli_si128::<2>(avg));
        store4(dst.add(3 * stride), _mm_srli_si128::<3>(avg));
    }
}

/// libwebp's `RD4_SSE2`.
pub(crate) fn rd4(buf: &mut [u8], p: usize, stride: usize) {
    check4(buf.len(), p, stride);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let left = |y: usize| u32::from(*dst.add(y * stride).sub(1));
        let xabcd = _mm_slli_si128::<4>(_mm_loadl_epi64(dst.sub(stride + 1).cast()));
        let lkji = _mm_cvtsi32_si128(
            (left(3) | (left(2) << 8) | (left(1) << 16) | (left(0) << 24)) as i32,
        );
        let e = _mm_or_si128(lkji, xabcd);
        let avg = avg3(e, _mm_srli_si128::<1>(e), _mm_srli_si128::<2>(e));
        store4(dst.add(3 * stride), avg);
        store4(dst.add(2 * stride), _mm_srli_si128::<1>(avg));
        store4(dst.add(stride), _mm_srli_si128::<2>(avg));
        store4(dst, _mm_srli_si128::<3>(avg));
    }
}

/// libwebp's `VR4_SSE2`.
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
        let xabcd = _mm_loadl_epi64(dst.sub(stride + 1).cast());
        let abcd0 = _mm_srli_si128::<1>(xabcd);
        let abcd = _mm_avg_epu8(xabcd, abcd0);
        let ixabcd = _mm_insert_epi16::<0>(
            _mm_slli_si128::<1>(xabcd),
            i32::from(i) | (i32::from(x) << 8),
        );
        let efgh = avg3(ixabcd, xabcd, abcd0);
        store4(dst, abcd);
        store4(dst.add(stride), efgh);
        store4(dst.add(2 * stride), _mm_slli_si128::<1>(abcd));
        store4(dst.add(3 * stride), _mm_slli_si128::<1>(efgh));
        let avg3_1 =
            |a: u8, b: u8, c: u8| ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8;
        *dst.add(2 * stride) = avg3_1(j, i, x);
        *dst.add(3 * stride) = avg3_1(k, j, i);
    }
}

/// libwebp's `VL4_SSE2`.
pub(crate) fn vl4(buf: &mut [u8], p: usize, stride: usize) {
    check4(buf.len(), p, stride);
    unsafe {
        let dst = buf.as_mut_ptr().add(p);
        let one = _mm_set1_epi8(1);
        let a = _mm_loadl_epi64(dst.sub(stride).cast());
        let b = _mm_srli_si128::<1>(a);
        let c = _mm_srli_si128::<2>(a);
        let avg1 = _mm_avg_epu8(a, b);
        let avg2 = _mm_avg_epu8(c, b);
        let avg3 = _mm_avg_epu8(avg1, avg2);
        let lsb1 = _mm_and_si128(_mm_xor_si128(avg1, avg2), one);
        let abbc = _mm_or_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, b));
        let avg4 = _mm_subs_epu8(avg3, _mm_and_si128(abbc, lsb1));
        let extra = _mm_cvtsi128_si32(_mm_srli_si128::<4>(avg4)) as u32;
        store4(dst, avg1);
        store4(dst.add(stride), avg4);
        store4(dst.add(2 * stride), _mm_srli_si128::<1>(avg1));
        store4(dst.add(3 * stride), _mm_srli_si128::<1>(avg4));
        *dst.add(2 * stride + 3) = extra as u8;
        *dst.add(3 * stride + 3) = (extra >> 8) as u8;
    }
}

/// libwebp's `TrueMotion_SSE2` of a `SIZE`x`SIZE` block (4, 8 or 16) at `p`.
pub(crate) fn tm<const SIZE: usize>(buf: &mut [u8], p: usize, stride: usize) {
    check(buf.len(), p, stride + 1, (SIZE - 1) * stride + SIZE);
    unsafe {
        let zero = _mm_setzero_si128();
        let mut dst = buf.as_mut_ptr().add(p);
        let top = dst.sub(stride);
        let corner = i16::from(*top.sub(1));
        let t = match SIZE {
            4 => _mm_cvtsi32_si128(rd32(top)),
            8 => _mm_loadl_epi64(top.cast()),
            _ => load16(top),
        };
        let (t0, t1) = (_mm_unpacklo_epi8(t, zero), _mm_unpackhi_epi8(t, zero));
        for _ in 0..SIZE {
            let base = _mm_set1_epi16(i16::from(*dst.sub(1)) - corner);
            let out = _mm_packus_epi16(_mm_add_epi16(base, t0), _mm_add_epi16(base, t1));
            match SIZE {
                4 => store4(dst, out),
                8 => _mm_storel_epi64(dst.cast(), out),
                _ => store16(dst, out),
            }
            dst = dst.add(stride);
        }
    }
}

/// Byte 15 of `x` in every byte.
#[inline(always)]
unsafe fn broadcast_last(x: __m128i) -> __m128i {
    _mm_shuffle_epi32::<0xff>(_mm_shufflehi_epi16::<0xff>(_mm_unpackhi_epi8(x, x)))
}

/// The running sum of the 16 bytes of `x`, plus `left` in every byte.
#[inline(always)]
unsafe fn running_sum(x: __m128i, left: __m128i) -> __m128i {
    let s = _mm_add_epi8(x, _mm_slli_si128::<1>(x));
    let s = _mm_add_epi8(s, _mm_slli_si128::<2>(s));
    let s = _mm_add_epi8(s, _mm_slli_si128::<4>(s));
    let s = _mm_add_epi8(s, _mm_slli_si128::<8>(s));
    _mm_add_epi8(s, left)
}

/// The gradient alpha unfiltering of `row[x..]` (`row[x - 1]` done), 16 pixels at a time as
/// a running sum of the residuals and the top row's steps, for as long as no prediction
/// needs clamping. Returns where it stopped: at a block that clamps or with fewer than 16
/// pixels left.
pub(crate) fn unfilter_gradient(prev: &[u8], row: &mut [u8], mut x: usize) -> usize {
    let n = row.len().min(prev.len());
    assert!(x >= 1);
    // SAFETY: SSE2 is part of x86_64; the loads read prev[x - 1..x + 16] and row[x..x + 16]
    // and the store writes row[x..x + 16], in bounds for x + 16 <= n.
    unsafe {
        let zero = _mm_setzero_si128();
        let mut left = _mm_set1_epi8(row[x - 1] as i8);
        while x + 16 <= n {
            let top = load16(prev.as_ptr().add(x));
            let top_left = load16(prev.as_ptr().add(x - 1));
            let input = load16(row.as_ptr().add(x));
            let out = running_sum(_mm_add_epi8(input, _mm_sub_epi8(top, top_left)), left);
            // The unclamped predictions, from the left pixels this block would produce; one is
            // out of 0..=255 where its high byte is not zero.
            let before = _mm_or_si128(_mm_slli_si128::<1>(out), _mm_srli_si128::<15>(left));
            let pred = |b: __m128i, t: __m128i, tl: __m128i| {
                _mm_srli_epi16::<8>(_mm_sub_epi16(_mm_add_epi16(b, t), tl))
            };
            let lo = pred(
                _mm_unpacklo_epi8(before, zero),
                _mm_unpacklo_epi8(top, zero),
                _mm_unpacklo_epi8(top_left, zero),
            );
            let hi = pred(
                _mm_unpackhi_epi8(before, zero),
                _mm_unpackhi_epi8(top, zero),
                _mm_unpackhi_epi8(top_left, zero),
            );
            let in_range = _mm_cmpeq_epi8(_mm_packs_epi16(lo, hi), zero);
            if _mm_movemask_epi8(in_range) != 0xffff {
                break;
            }
            store16(row.as_mut_ptr().add(x), out);
            left = broadcast_last(out);
            x += 16;
        }
    }
    x
}

/// The horizontal alpha unfiltering of `row[x..]` (`row[x - 1]` done), a running sum 16
/// pixels at a time. Returns where it stopped, with fewer than 16 pixels left.
pub(crate) fn unfilter_horizontal(row: &mut [u8], mut x: usize) -> usize {
    assert!(x >= 1);
    // SAFETY: SSE2 is part of x86_64; the load reads and the store writes row[x..x + 16], in
    // bounds for x + 16 <= row.len().
    unsafe {
        let mut left = _mm_set1_epi8(row[x - 1] as i8);
        while x + 16 <= row.len() {
            let out = running_sum(load16(row.as_ptr().add(x)), left);
            store16(row.as_mut_ptr().add(x), out);
            left = broadcast_last(out);
            x += 16;
        }
    }
    x
}
