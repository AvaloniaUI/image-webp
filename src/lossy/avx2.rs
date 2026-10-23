//! AVX2 decoder kernels: libwebp has none for VP8 decoding, so these run the SSE2 kernels of
//! `sse2.rs` 256 bits wide, bit-exact with them. Callers check [`crate::wide::has_avx2`] first.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

/// 16 bytes at `p` in the low lane, 16 at `p + 32` in the high one.
#[inline(always)]
unsafe fn load_lanes(p: *const u8) -> __m256i {
    _mm256_loadu2_m128i(p.add(32).cast(), p.cast())
}

/// `yuv444_to_rgb` of `sse2.rs` on 16 pixels whose luma and chroma are in the high bytes.
#[inline(always)]
unsafe fn yuv444_to_rgb(y0: __m256i, u0: __m256i, v0: __m256i) -> [__m256i; 3] {
    let y1 = _mm256_mulhi_epu16(y0, _mm256_set1_epi16(19077));
    let r0 = _mm256_mulhi_epu16(v0, _mm256_set1_epi16(26149));
    let r2 = _mm256_add_epi16(_mm256_sub_epi16(y1, _mm256_set1_epi16(14234)), r0);
    let g0 = _mm256_mulhi_epu16(u0, _mm256_set1_epi16(6419));
    let g1 = _mm256_mulhi_epu16(v0, _mm256_set1_epi16(13320));
    let g4 = _mm256_sub_epi16(
        _mm256_add_epi16(y1, _mm256_set1_epi16(8708)),
        _mm256_add_epi16(g0, g1),
    );
    let b0 = _mm256_mulhi_epu16(u0, _mm256_set1_epi16(33050u16 as i16));
    let b2 = _mm256_subs_epu16(_mm256_adds_epu16(b0, y1), _mm256_set1_epi16(17685));
    [
        _mm256_srai_epi16::<6>(r2),
        _mm256_srai_epi16::<6>(g4),
        _mm256_srli_epi16::<6>(b2),
    ]
}

#[inline(always)]
unsafe fn planar_to_24b_step(i: [__m256i; 6]) -> [__m256i; 6] {
    let m = _mm256_set1_epi16(0x00ff);
    let even = |a, b| _mm256_packus_epi16(_mm256_and_si256(a, m), _mm256_and_si256(b, m));
    let odd = |a, b| _mm256_packus_epi16(_mm256_srli_epi16::<8>(a), _mm256_srli_epi16::<8>(b));
    [
        even(i[0], i[1]),
        even(i[2], i[3]),
        even(i[4], i[5]),
        odd(i[0], i[1]),
        odd(i[2], i[3]),
        odd(i[4], i[5]),
    ]
}

#[inline(always)]
unsafe fn store_lo(p: *mut u8, x: __m256i) {
    _mm_storeu_si128(p.cast(), _mm256_castsi256_si128(x));
}

#[inline(always)]
unsafe fn store_hi(p: *mut u8, x: __m256i) {
    _mm_storeu_si128(p.cast(), _mm256_extracti128_si256::<1>(x));
}

/// `sse2::yuv_to_rgb_row` 64 (RGB) or 16 (RGBA) pixels at a time: RGB runs libwebp's
/// `VP8YuvToRgb32_SSE2` on two blocks of 32 pixels, one in each lane.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_to_rgb_row<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
) -> usize {
    let n = y.len().min(u.len()).min(v.len()).min(out.len() / BPP);
    let mut i = 0;
    let (y, u, v, out) = (y.as_ptr(), u.as_ptr(), v.as_ptr(), out.as_mut_ptr());
    let zero = _mm256_setzero_si256();
    if BPP == 3 {
        while i + 64 <= n {
            let [ya, ua, va] = [y, u, v].map(|p| load_lanes(p.add(i)));
            let [yb, ub, vb] = [y, u, v].map(|p| load_lanes(p.add(i + 16)));
            let lo = |x| _mm256_unpacklo_epi8(zero, x);
            let hi = |x| _mm256_unpackhi_epi8(zero, x);
            let [r0, g0, b0] = yuv444_to_rgb(lo(ya), lo(ua), lo(va));
            let [r1, g1, b1] = yuv444_to_rgb(hi(ya), hi(ua), hi(va));
            let [r2, g2, b2] = yuv444_to_rgb(lo(yb), lo(ub), lo(vb));
            let [r3, g3, b3] = yuv444_to_rgb(hi(yb), hi(ub), hi(vb));
            let mut c = [
                _mm256_packus_epi16(r0, r1),
                _mm256_packus_epi16(r2, r3),
                _mm256_packus_epi16(g0, g1),
                _mm256_packus_epi16(g2, g3),
                _mm256_packus_epi16(b0, b1),
                _mm256_packus_epi16(b2, b3),
            ];
            for _ in 0..5 {
                c = planar_to_24b_step(c);
            }
            for (k, x) in c.into_iter().enumerate() {
                store_lo(out.add(3 * i + 16 * k), x);
                store_hi(out.add(3 * i + 96 + 16 * k), x);
            }
            i += 64;
        }
    } else {
        let alpha = _mm256_set1_epi16(255);
        let load = |p: *const u8| {
            _mm256_slli_epi16::<8>(_mm256_cvtepu8_epi16(_mm_loadu_si128(p.cast())))
        };
        while i + 16 <= n {
            let [r, g, b] = yuv444_to_rgb(load(y.add(i)), load(u.add(i)), load(v.add(i)));
            let rb = _mm256_packus_epi16(r, b);
            let ga = _mm256_packus_epi16(g, alpha);
            let rg = _mm256_unpacklo_epi8(rb, ga);
            let ba = _mm256_unpackhi_epi8(rb, ga);
            let (lo, hi) = (_mm256_unpacklo_epi16(rg, ba), _mm256_unpackhi_epi16(rg, ba));
            let p = out.add(4 * i);
            store_lo(p, lo);
            store_lo(p.add(16), hi);
            store_hi(p.add(32), lo);
            store_hi(p.add(48), hi);
            i += 16;
        }
    }
    i
}

/// `sse2::upsample_pairs` 32 pairs at a time, libwebp's `UPSAMPLE_32PIXELS` on 256 bits.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn upsample_pairs(near: &[u8], far: &[u8], out: &mut [u8]) -> usize {
    let n = near.len().min(far.len());
    let pairs = (out.len().saturating_sub(1) / 2).min(n.saturating_sub(1));
    let mut j = 0;
    let load = |p: &[u8], k: usize| _mm256_loadu_si256(p.as_ptr().add(k).cast());
    let one = _mm256_set1_epi8(1);
    while j + 32 <= pairs && j + 33 <= n {
        let (a, b) = (load(near, j), load(near, j + 1));
        let (c, d) = (load(far, j), load(far, j + 1));
        let s = _mm256_avg_epu8(a, d);
        let t = _mm256_avg_epu8(b, c);
        let st = _mm256_xor_si256(s, t);
        let ad = _mm256_xor_si256(a, d);
        let bc = _mm256_xor_si256(b, c);
        let lsb = _mm256_and_si256(_mm256_or_si256(_mm256_or_si256(ad, bc), st), one);
        let k = _mm256_sub_epi8(_mm256_avg_epu8(s, t), lsb);
        let get_m = |ij: __m256i, x: __m256i| {
            let lsb = _mm256_and_si256(
                _mm256_or_si256(_mm256_and_si256(ij, st), _mm256_xor_si256(k, x)),
                one,
            );
            _mm256_sub_epi8(_mm256_avg_epu8(k, x), lsb)
        };
        let even = _mm256_avg_epu8(a, get_m(bc, t));
        let odd = _mm256_avg_epu8(b, get_m(ad, s));
        let (lo, hi) = (_mm256_unpacklo_epi8(even, odd), _mm256_unpackhi_epi8(even, odd));
        let p = out.as_mut_ptr().add(2 * j + 1);
        store_lo(p, lo);
        store_lo(p.add(16), hi);
        store_hi(p.add(32), lo);
        store_hi(p.add(48), hi);
        j += 32;
    }
    j
}

use super::sse2::{load16, load16x4, load_uv, rd32, store16, store16x4, store4x4, store_uv};

#[inline(always)]
unsafe fn abs_diff(p: __m256i, q: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_subs_epu8(q, p), _mm256_subs_epu8(p, q))
}

#[inline(always)]
unsafe fn flip(x: __m256i) -> __m256i {
    _mm256_xor_si256(x, _mm256_set1_epi8(-128))
}

#[inline(always)]
unsafe fn signed_shift8b(x: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let lo = _mm256_srai_epi16::<11>(_mm256_unpacklo_epi8(zero, x));
    let hi = _mm256_srai_epi16::<11>(_mm256_unpackhi_epi8(zero, x));
    _mm256_packs_epi16(lo, hi)
}

#[inline(always)]
unsafe fn not_hev(p1: __m256i, p0: __m256i, q0: __m256i, q1: __m256i, hev_thresh: u8) -> __m256i {
    let t_max = _mm256_max_epu8(abs_diff(p1, p0), abs_diff(q1, q0));
    let t_max_h = _mm256_subs_epu8(t_max, _mm256_set1_epi8(hev_thresh as i8));
    _mm256_cmpeq_epi8(t_max_h, _mm256_setzero_si256())
}

#[inline(always)]
unsafe fn base_delta(p1: __m256i, p0: __m256i, q0: __m256i, q1: __m256i) -> __m256i {
    let p1_q1 = _mm256_subs_epi8(p1, q1);
    let q0_p0 = _mm256_subs_epi8(q0, p0);
    let s1 = _mm256_adds_epi8(p1_q1, q0_p0);
    let s2 = _mm256_adds_epi8(q0_p0, s1);
    _mm256_adds_epi8(q0_p0, s2)
}

#[inline(always)]
unsafe fn update_2_pixels(pi: &mut __m256i, qi: &mut __m256i, a0_lo: __m256i, a0_hi: __m256i) {
    let delta = _mm256_packs_epi16(_mm256_srai_epi16::<7>(a0_lo), _mm256_srai_epi16::<7>(a0_hi));
    *pi = flip(_mm256_adds_epi8(*pi, delta));
    *qi = flip(_mm256_subs_epi8(*qi, delta));
}

#[inline(always)]
unsafe fn needs_filter(p1: __m256i, p0: __m256i, q0: __m256i, q1: __m256i, thresh: u8) -> __m256i {
    let t3 = _mm256_srli_epi16::<1>(_mm256_and_si256(abs_diff(p1, q1), _mm256_set1_epi8(-2)));
    let t4 = abs_diff(p0, q0);
    let t6 = _mm256_adds_epu8(_mm256_adds_epu8(t4, t4), t3);
    let t7 = _mm256_subs_epu8(t6, _mm256_set1_epi8(thresh as i8));
    _mm256_cmpeq_epi8(t7, _mm256_setzero_si256())
}

#[inline(always)]
unsafe fn do_filter4(
    p1: &mut __m256i,
    p0: &mut __m256i,
    q0: &mut __m256i,
    q1: &mut __m256i,
    mask: __m256i,
    hev_thresh: u8,
) {
    let not_hev = not_hev(*p1, *p0, *q0, *q1, hev_thresh);
    let (sp1, sp0, sq0, sq1) = (flip(*p1), flip(*p0), flip(*q0), flip(*q1));
    let mut t1 = _mm256_andnot_si256(not_hev, _mm256_subs_epi8(sp1, sq1));
    let t2 = _mm256_subs_epi8(sq0, sp0);
    t1 = _mm256_adds_epi8(t1, t2);
    t1 = _mm256_adds_epi8(t1, t2);
    t1 = _mm256_adds_epi8(t1, t2);
    t1 = _mm256_and_si256(t1, mask);
    let t2 = signed_shift8b(_mm256_adds_epi8(t1, _mm256_set1_epi8(3)));
    let t3 = signed_shift8b(_mm256_adds_epi8(t1, _mm256_set1_epi8(4)));
    *p0 = flip(_mm256_adds_epi8(sp0, t2));
    *q0 = flip(_mm256_subs_epi8(sq0, t3));
    // signed (t3 + 1) >> 1
    let t3 = _mm256_sub_epi8(
        _mm256_avg_epu8(_mm256_add_epi8(t3, _mm256_set1_epi8(-128)), _mm256_setzero_si256()),
        _mm256_set1_epi8(64),
    );
    let t3 = _mm256_and_si256(not_hev, t3);
    *q1 = flip(_mm256_subs_epi8(sq1, t3));
    *p1 = flip(_mm256_adds_epi8(sp1, t3));
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn do_filter6(
    p2: &mut __m256i,
    p1: &mut __m256i,
    p0: &mut __m256i,
    q0: &mut __m256i,
    q1: &mut __m256i,
    q2: &mut __m256i,
    mask: __m256i,
    hev_thresh: u8,
) {
    let zero = _mm256_setzero_si256();
    let not_hev = not_hev(*p1, *p0, *q0, *q1, hev_thresh);
    let (sp2, sp1, mut sp0, mut sq0, sq1, sq2) =
        (flip(*p2), flip(*p1), flip(*p0), flip(*q0), flip(*q1), flip(*q2));
    let a = base_delta(sp1, sp0, sq0, sq1);
    let fl = _mm256_and_si256(a, _mm256_andnot_si256(not_hev, mask));
    let v3 = signed_shift8b(_mm256_adds_epi8(fl, _mm256_set1_epi8(3)));
    let v4 = signed_shift8b(_mm256_adds_epi8(fl, _mm256_set1_epi8(4)));
    sq0 = _mm256_subs_epi8(sq0, v4);
    sp0 = _mm256_adds_epi8(sp0, v3);

    let k9 = _mm256_set1_epi16(0x0900);
    let k63 = _mm256_set1_epi16(63);
    let f = _mm256_and_si256(a, _mm256_and_si256(not_hev, mask));
    let f9_lo = _mm256_mulhi_epi16(_mm256_unpacklo_epi8(zero, f), k9);
    let f9_hi = _mm256_mulhi_epi16(_mm256_unpackhi_epi8(zero, f), k9);
    let a2_lo = _mm256_add_epi16(f9_lo, k63);
    let a2_hi = _mm256_add_epi16(f9_hi, k63);
    let a1_lo = _mm256_add_epi16(a2_lo, f9_lo);
    let a1_hi = _mm256_add_epi16(a2_hi, f9_hi);
    let a0_lo = _mm256_add_epi16(a1_lo, f9_lo);
    let a0_hi = _mm256_add_epi16(a1_hi, f9_hi);
    (*p2, *q2, *p1, *q1, *p0, *q0) = (sp2, sq2, sp1, sq1, sp0, sq0);
    update_2_pixels(p2, q2, a2_lo, a2_hi);
    update_2_pixels(p1, q1, a1_lo, a1_hi);
    update_2_pixels(p0, q0, a0_lo, a0_hi);
}

#[inline(always)]
unsafe fn max_diff(m: __m256i, p3: __m256i, p2: __m256i, p1: __m256i, p0: __m256i) -> __m256i {
    let m = _mm256_max_epu8(m, abs_diff(p1, p0));
    let m = _mm256_max_epu8(m, abs_diff(p3, p2));
    _mm256_max_epu8(m, abs_diff(p2, p1))
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn complex_mask(
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    thresh: u8,
    ithresh: u8,
    mask: __m256i,
) -> __m256i {
    let diff = _mm256_subs_epu8(mask, _mm256_set1_epi8(ithresh as i8));
    let thresh_mask = _mm256_cmpeq_epi8(diff, _mm256_setzero_si256());
    _mm256_and_si256(thresh_mask, needs_filter(p1, p0, q0, q1, thresh))
}

/// A luma plane in the low lanes and the two chroma planes in the high ones.
#[derive(Clone, Copy)]
struct Planes {
    y: *mut u8,
    ys: usize,
    u: *mut u8,
    v: *mut u8,
    cs: usize,
}

impl Planes {
    #[inline(always)]
    fn at(self, dy: isize, dc: isize) -> Self {
        Self {
            y: self.y.wrapping_offset(dy),
            u: self.u.wrapping_offset(dc),
            v: self.v.wrapping_offset(dc),
            ..self
        }
    }

    /// Row `k` of the luma plane beside row `k` of the chroma planes.
    #[inline(always)]
    unsafe fn row(self, k: isize) -> __m256i {
        let (y, c) = (k * self.ys as isize, k * self.cs as isize);
        _mm256_set_m128i(
            load_uv(self.u.offset(c), self.v.offset(c)),
            load16(self.y.offset(y)),
        )
    }

    #[inline(always)]
    unsafe fn store_row(self, k: isize, x: __m256i) {
        let (y, c) = (k * self.ys as isize, k * self.cs as isize);
        store16(self.y.offset(y), _mm256_castsi256_si128(x));
        store_uv(
            _mm256_extracti128_si256::<1>(x),
            self.u.offset(c),
            self.v.offset(c),
        );
    }

    /// `sse2::load8x4` of 8 luma and 8 chroma rows.
    #[inline(always)]
    unsafe fn load8x4(l: *const u8, ls: usize, c: *const u8, cs: usize) -> (__m256i, __m256i) {
        let a = |k: usize| {
            _mm256_setr_epi32(
                rd32(l.add(k * ls)),
                rd32(l.add((k + 4) * ls)),
                rd32(l.add((k + 2) * ls)),
                rd32(l.add((k + 6) * ls)),
                rd32(c.add(k * cs)),
                rd32(c.add((k + 4) * cs)),
                rd32(c.add((k + 2) * cs)),
                rd32(c.add((k + 6) * cs)),
            )
        };
        let (a0, a1) = (a(0), a(1));
        let b0 = _mm256_unpacklo_epi8(a0, a1);
        let b1 = _mm256_unpackhi_epi8(a0, a1);
        let c0 = _mm256_unpacklo_epi16(b0, b1);
        let c1 = _mm256_unpackhi_epi16(b0, b1);
        (_mm256_unpacklo_epi32(c0, c1), _mm256_unpackhi_epi32(c0, c1))
    }

    /// `sse2::load16x4`: columns 0..4 of the 16 luma rows and of the 8 rows of each chroma plane.
    #[inline(always)]
    unsafe fn columns(self) -> [__m256i; 4] {
        let (t1, t2) = Self::load8x4(self.y, self.ys, self.u, self.cs);
        let (p0, q1) = Self::load8x4(self.y.add(8 * self.ys), self.ys, self.v, self.cs);
        [
            _mm256_unpacklo_epi64(t1, p0),
            _mm256_unpackhi_epi64(t1, p0),
            _mm256_unpacklo_epi64(t2, q1),
            _mm256_unpackhi_epi64(t2, q1),
        ]
    }

    #[inline(always)]
    unsafe fn store_columns(self, c: [__m256i; 4]) {
        let [p1, p0, q0, q1] = c;
        let p0_s = _mm256_unpacklo_epi8(p1, p0);
        let p1_s = _mm256_unpackhi_epi8(p1, p0);
        let q0_s = _mm256_unpacklo_epi8(q0, q1);
        let q1_s = _mm256_unpackhi_epi8(q0, q1);
        let (ys, cs) = (self.ys, self.cs);
        for (x, l, c) in [
            (_mm256_unpacklo_epi16(p0_s, q0_s), self.y, self.u),
            (_mm256_unpackhi_epi16(p0_s, q0_s), self.y.add(4 * ys), self.u.add(4 * cs)),
            (_mm256_unpacklo_epi16(p1_s, q1_s), self.y.add(8 * ys), self.v),
            (_mm256_unpackhi_epi16(p1_s, q1_s), self.y.add(12 * ys), self.v.add(4 * cs)),
        ] {
            store4x4(_mm256_castsi256_si128(x), l, ys);
            store4x4(_mm256_extracti128_si256::<1>(x), c, cs);
        }
    }
}

/// libwebp's `DoFilter` for one macroblock with the normal filter, the luma filters (`VP8HFilter16`,
/// `VP8HFilter16i`, `VP8VFilter16`, `VP8VFilter16i`) and the chroma ones (`VP8HFilter8`, ...) on
/// the same edge in one 256-bit register: `py` and `puv` are the macroblock origins in the planes,
/// `left`, `inner` and `top` the edges to filter. False, filtering nothing, without AVX2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_normal(
    y: &mut [u8],
    py: usize,
    ys: usize,
    u: &mut [u8],
    v: &mut [u8],
    puv: usize,
    cs: usize,
    [left, inner, top]: [bool; 3],
    limit: u8,
    ilevel: u8,
    hev: u8,
) -> bool {
    if !crate::wide::has_avx2() {
        return false;
    }
    let back = |stride: usize| if top { 4 * stride } else if left { 4 } else { 0 };
    assert!(py >= back(ys) && py + 15 * ys + 16 <= y.len());
    for c in [&*u, &*v] {
        assert!(puv >= back(cs) && puv + 7 * cs + 8 <= c.len());
    }
    let planes = Planes {
        y: y.as_mut_ptr(),
        ys,
        u: u.as_mut_ptr(),
        v: v.as_mut_ptr(),
        cs,
    };
    // SAFETY: the CPU has AVX2; the filters reach 4 rows above and 4 pixels left of the
    // macroblocks where they filter those edges, and no further than their last row and column.
    unsafe {
        filter_normal_avx2(
            planes.at(py as isize, puv as isize),
            [left, inner, top],
            limit,
            ilevel,
            hev,
        );
    }
    true
}

#[target_feature(enable = "avx2")]
unsafe fn filter_normal_avx2(m: Planes, [left, inner, top]: [bool; 3], limit: u8, ilevel: u8, hev: u8) {
    let (ys, cs) = (m.ys as isize, m.cs as isize);
    if left {
        let b = m.at(-4, -4);
        let [p3, mut p2, mut p1, mut p0] = b.columns();
        let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
        let [mut q0, mut q1, mut q2, q3] = m.columns();
        let mask = max_diff(mask, q3, q2, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, limit + 4, ilevel, mask);
        do_filter6(&mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev);
        b.store_columns([p3, p2, p1, p0]);
        m.store_columns([q0, q1, q2, q3]);
    }
    if inner {
        let luma = |b: Planes| load16x4(b.y, b.y.add(8 * b.ys), b.ys).map(|x| _mm256_castsi128_si256(x));
        let [mut p3, mut p2, mut p1, mut p0] = m.columns();
        let mut b = m;
        for k in 0..3 {
            let d = b.at(2, 2);
            b = b.at(4, 4);
            let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
            let [n3, n2, tmp1, tmp2] = if k == 0 { b.columns() } else { luma(b) };
            (p3, p2) = (n3, n2);
            let mask = max_diff(mask, p3, p2, tmp1, tmp2);
            let mask = complex_mask(p1, p0, p3, p2, limit, ilevel, mask);
            do_filter4(&mut p1, &mut p0, &mut p3, &mut p2, mask, hev);
            if k == 0 {
                d.store_columns([p1, p0, p3, p2]);
            } else {
                let c = [p1, p0, p3, p2].map(|x| _mm256_castsi256_si128(x));
                store16x4(c, d.y, d.y.add(8 * m.ys), m.ys);
            }
            (p1, p0) = (tmp1, tmp2);
        }
    }
    if top {
        let (t1, mut p2, mut p1, mut p0) = (m.row(-4), m.row(-3), m.row(-2), m.row(-1));
        let mask = max_diff(abs_diff(p1, p0), t1, p2, p1, p0);
        let (mut q0, mut q1, mut q2, t1) = (m.row(0), m.row(1), m.row(2), m.row(3));
        let mask = max_diff(mask, t1, q2, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, limit + 4, ilevel, mask);
        do_filter6(&mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev);
        for (k, x) in [(-3, p2), (-2, p1), (-1, p0), (0, q0), (1, q1), (2, q2)] {
            m.store_row(k, x);
        }
    }
    if inner {
        let luma = |b: Planes, k: isize| _mm256_castsi128_si256(load16(b.y.offset(k * ys)));
        let (mut p3, mut p2, mut p1, mut p0) = (m.row(0), m.row(1), m.row(2), m.row(3));
        let mut b = m;
        for k in 0..3 {
            let d = b.at(2 * ys, 2 * cs);
            b = b.at(4 * ys, 4 * cs);
            let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
            let [n3, n2, tmp1, tmp2] = match k {
                0 => [0, 1, 2, 3].map(|r| b.row(r)),
                _ => [0, 1, 2, 3].map(|r| luma(b, r)),
            };
            (p3, p2) = (n3, n2);
            let mask = max_diff(mask, p3, p2, tmp1, tmp2);
            let mask = complex_mask(p1, p0, p3, p2, limit, ilevel, mask);
            do_filter4(&mut p1, &mut p0, &mut p3, &mut p2, mask, hev);
            for (r, x) in [(0, p1), (1, p0), (2, p3), (3, p2)] {
                match k {
                    0 => d.store_row(r, x),
                    _ => store16(d.y.offset(r * ys), _mm256_castsi256_si128(x)),
                }
            }
            (p1, p0) = (tmp1, tmp2);
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    #[test]
    fn yuv_rows_match_sse2() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for len in [0usize, 15, 16, 17, 63, 64, 65, 100, 128, 191, 300] {
            let [y, u, v] = [0; 3].map(|_| (0..len).map(|_| rng.gen()).collect::<Vec<u8>>());
            for bpp in [3, 4] {
                let mut a = vec![7u8; len * bpp];
                let mut b = vec![7u8; len * bpp];
                // SAFETY: the CPU has AVX2.
                let done = unsafe {
                    match bpp {
                        3 => super::yuv_to_rgb_row::<3>(&y, &u, &v, &mut a),
                        _ => super::yuv_to_rgb_row::<4>(&y, &u, &v, &mut a),
                    }
                };
                let step = if bpp == 3 { 64 } else { 16 };
                assert_eq!(done, len / step * step);
                match bpp {
                    3 => super::super::sse2::yuv_to_rgb_row_from::<3>(&y, &u, &v, &mut b, 0),
                    _ => super::super::sse2::yuv_to_rgb_row_from::<4>(&y, &u, &v, &mut b, 0),
                };
                assert_eq!(a[..done * bpp], b[..done * bpp], "bpp {bpp} len {len}");
            }
        }
    }

    #[test]
    fn upsampled_pairs_match_sse2() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);
        for round in 0usize..600 {
            let width = 1 + round % 300;
            let n = width.div_ceil(2);
            let mut sample = || if round % 3 == 0 { rng.gen_range(0..2) * 255 } else { rng.gen() };
            let near: Vec<u8> = (0..n).map(|_| sample()).collect();
            let far: Vec<u8> = (0..n).map(|_| sample()).collect();
            let mut a = vec![0u8; width];
            let mut b = vec![0u8; width];
            // SAFETY: the CPU has AVX2.
            let done = unsafe { super::upsample_pairs(&near, &far, &mut a) };
            super::super::sse2::upsample_pairs_from(&near, &far, &mut b, 0);
            assert_eq!(a[1..1 + 2 * done], b[1..1 + 2 * done], "{width}");
        }
    }

    #[test]
    fn macroblock_filter_matches_sse2() {
        use super::super::sse2;
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);
        let (ys, cs) = (48, 24);
        let mut changed = 0;
        for round in 0..2000 {
            let flags = [round & 1 != 0, round & 2 != 0, round & 4 != 0];
            let (level, interior, hev) = (rng.gen_range(1..64u8), rng.gen_range(1..10u8), rng.gen_range(0..3));
            let limit = 2 * level + interior;
            let spread = [3, 10, 30, 127][round / 8 % 4];
            let mut plane = |w: usize, h: usize| {
                let base: Vec<i32> =
                    (0..w * h / 16).map(|_| 128 + rng.gen_range(-spread..=spread)).collect();
                (0..w * h)
                    .map(|i| {
                        let b = base[(i / w / 4) * (w / 4) + i % w / 4];
                        (b + rng.gen_range(-3..=3)).clamp(0, 255) as u8
                    })
                    .collect::<Vec<u8>>()
            };
            let (y, u, v) = (plane(ys, 48), plane(cs, 24), plane(cs, 24));
            let (py, puv) = (16 * ys + 16, 8 * cs + 8);
            let (mut ya, mut ua, mut va) = (y.clone(), u.clone(), v.clone());
            assert!(super::filter_normal(
                &mut ya, py, ys, &mut ua, &mut va, puv, cs, flags, limit, interior, hev
            ));
            changed += usize::from(ya != y && ua != u);
            let (mut yb, mut ub, mut vb) = (y, u, v);
            let [left, inner, top] = flags;
            if left {
                sse2::h16(&mut yb, py, ys, limit + 4, interior, hev);
                sse2::h8(&mut ub, &mut vb, puv, cs, limit + 4, interior, hev);
            }
            if inner {
                sse2::h16i(&mut yb, py, ys, limit, interior, hev);
                sse2::h8i(&mut ub, &mut vb, puv, cs, limit, interior, hev);
            }
            if top {
                sse2::v16(&mut yb, py, ys, limit + 4, interior, hev);
                sse2::v8(&mut ub, &mut vb, puv, cs, limit + 4, interior, hev);
            }
            if inner {
                sse2::v16i(&mut yb, py, ys, limit, interior, hev);
                sse2::v8i(&mut ub, &mut vb, puv, cs, limit, interior, hev);
            }
            assert_eq!((ya, ua, va), (yb, ub, vb), "round {round}");
        }
        assert!(changed > 1000, "{changed} rounds filtered");
    }

}
