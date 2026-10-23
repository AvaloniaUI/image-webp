//! SSE2 kernels ported from libwebp's `dsp/enc_sse2.c`, bit-exact with the scalar ones.
//! SSE2 is part of x86_64, so no runtime detection is needed. Each wrapper checks the
//! lengths its loads and stores rely on.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

use super::dsp::{Matrix, BPS, MAX_LEVEL, QFIX};

#[inline(always)]
unsafe fn loadl(p: &[u8], off: usize) -> __m128i {
    _mm_loadl_epi64(p.as_ptr().add(off).cast())
}

#[inline(always)]
unsafe fn load_i16x8(p: &[i16], off: usize) -> __m128i {
    _mm_loadu_si128(p.as_ptr().add(off).cast())
}

#[inline(always)]
unsafe fn hsum(v: __m128i) -> i32 {
    let mut tmp = [0i32; 4];
    _mm_storeu_si128(tmp.as_mut_ptr().cast(), v);
    tmp[3] + tmp[2] + tmp[1] + tmp[0]
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

/// One inverse transform of `input` (16 coefficients) added to the 4x4 block `r`.
#[inline]
pub(crate) fn itransform_one(r: &[u8], input: &[i16], dst: &mut [u8]) {
    unsafe {
        itransform_one_rows(r, input, dst);
    }
}

/// `itransform_one`, then the squared error of the result against the 4x4 block `src` and,
/// with `w`, `ttransform(src, result, w)`.
#[inline]
pub(crate) fn itransform_one_disto(
    r: &[u8],
    input: &[i16],
    dst: &mut [u8],
    src: &[u8],
    w: Option<&[u16; 16]>,
) -> (i32, i32) {
    let src = &src[..3 * BPS + 4];
    unsafe {
        let out = itransform_one_rows(r, input, dst);
        let zero = _mm_setzero_si128();
        let ld32 = |off: usize| {
            _mm_cvtsi32_si128(i32::from_le_bytes(src[off..off + 4].try_into().unwrap()))
        };
        let s01 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(ld32(0), ld32(BPS)), zero);
        let s23 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(ld32(2 * BPS), ld32(3 * BPS)), zero);
        let o01 = _mm_unpacklo_epi8(out, zero);
        let o23 = _mm_unpackhi_epi8(out, zero);
        let d01 = _mm_sub_epi16(s01, o01);
        let d23 = _mm_sub_epi16(s23, o23);
        let sse = hsum(_mm_add_epi32(_mm_madd_epi16(d01, d01), _mm_madd_epi16(d23, d23)));
        let disto = w.map_or(0, |w| {
            ttransform_rows(
                [
                    _mm_unpacklo_epi64(s01, o01),
                    _mm_unpackhi_epi64(s01, o01),
                    _mm_unpacklo_epi64(s23, o23),
                    _mm_unpackhi_epi64(s23, o23),
                ],
                w,
            )
        });
        (sse, disto)
    }
}

/// The inverse transform into `dst`, returning its four rows packed.
#[inline(always)]
unsafe fn itransform_one_rows(r: &[u8], input: &[i16], dst: &mut [u8]) -> __m128i {
    let input = &input[..16];
    let r = &r[..3 * BPS + 4];
    let dst = &mut dst[..3 * BPS + 4];
    let k1k2 = _mm_set_epi16(-30068, -30068, -30068, -30068, 20091, 20091, 20091, 20091);
    let k2k1 = _mm_set_epi16(20091, 20091, 20091, 20091, -30068, -30068, -30068, -30068);
    let zero = _mm_setzero_si128();
    let zero_four = _mm_set_epi16(0, 0, 0, 0, 4, 4, 4, 4);
    let in01 = load_i16x8(input, 0);
    let in23 = load_i16x8(input, 8);

    let pass = |x01: __m128i, x23: __m128i, round: __m128i, shift: bool| {
        let x1 = _mm_unpackhi_epi64(x01, x01);
        let x3 = _mm_unpackhi_epi64(x23, x23);
        let x01 = _mm_add_epi16(x01, round);
        let a_d3 = _mm_add_epi16(x01, x23);
        let b_c3 = _mm_sub_epi16(x01, x23);
        let c1d1 = _mm_mulhi_epi16(x1, k2k1);
        let c2d2 = _mm_mulhi_epi16(x3, k1k2);
        let c3 = _mm_unpackhi_epi64(b_c3, b_c3);
        let c4 = _mm_sub_epi16(c1d1, c2d2);
        let c = _mm_add_epi16(c3, c4);
        let d4u = _mm_add_epi16(c1d1, c2d2);
        let du = _mm_add_epi16(a_d3, d4u);
        let d = _mm_unpackhi_epi64(du, du);
        let comb_ab = _mm_unpacklo_epi64(a_d3, b_c3);
        let comb_dc = _mm_unpacklo_epi64(d, c);
        let mut tmp01 = _mm_add_epi16(comb_ab, comb_dc);
        let tmp32 = _mm_sub_epi16(comb_ab, comb_dc);
        let mut tmp23 = _mm_shuffle_epi32::<0b01_00_11_10>(tmp32);
        if shift {
            tmp01 = _mm_srai_epi16::<3>(tmp01);
            tmp23 = _mm_srai_epi16::<3>(tmp23);
        }
        let t0 = _mm_unpacklo_epi16(tmp01, tmp23);
        let t1 = _mm_unpackhi_epi16(tmp01, tmp23);
        (_mm_unpacklo_epi16(t0, t1), _mm_unpackhi_epi16(t0, t1))
    };
    let (t01, t23) = pass(in01, in23, zero, false);
    let (t01, t23) = pass(t01, t23, zero_four, true);

    let ld32 =
        |off: usize| _mm_cvtsi32_si128(i32::from_le_bytes(r[off..off + 4].try_into().unwrap()));
    let ref01 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(ld32(0), ld32(BPS)), zero);
    let ref23 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(ld32(2 * BPS), ld32(3 * BPS)), zero);
    let out = _mm_packus_epi16(_mm_add_epi16(ref01, t01), _mm_add_epi16(ref23, t23));
    let rows = [
        out,
        _mm_srli_si128::<4>(out),
        _mm_srli_si128::<8>(out),
        _mm_srli_si128::<12>(out),
    ];
    for (y, row) in rows.into_iter().enumerate() {
        dst[y * BPS..y * BPS + 4].copy_from_slice(&_mm_cvtsi128_si32(row).to_le_bytes());
    }
    out
}

/// Two inverse transforms of `input` (32 coefficients) for side-by-side 4x4 blocks.
#[inline]
pub(crate) fn itransform_two(r: &[u8], input: &[i16], dst: &mut [u8]) {
    let input = &input[..32];
    let r = &r[..3 * BPS + 8];
    let dst = &mut dst[..3 * BPS + 8];
    unsafe {
        let k1 = _mm_set1_epi16(20091);
        let k2 = _mm_set1_epi16(-30068);
        let tmp0 = load_i16x8(input, 0);
        let tmp1 = load_i16x8(input, 8);
        let tmp2 = load_i16x8(input, 16);
        let tmp3 = load_i16x8(input, 24);
        let in0 = _mm_unpacklo_epi64(tmp0, tmp2);
        let in1 = _mm_unpackhi_epi64(tmp0, tmp2);
        let in2 = _mm_unpacklo_epi64(tmp1, tmp3);
        let in3 = _mm_unpackhi_epi64(tmp1, tmp3);

        let pass =
            |x0: __m128i, x1: __m128i, x2: __m128i, x3: __m128i, dc_add: i16, shift: bool| {
                let dc = _mm_add_epi16(x0, _mm_set1_epi16(dc_add));
                let a = _mm_add_epi16(dc, x2);
                let b = _mm_sub_epi16(dc, x2);
                let c = _mm_add_epi16(
                    _mm_sub_epi16(x1, x3),
                    _mm_sub_epi16(_mm_mulhi_epi16(x1, k2), _mm_mulhi_epi16(x3, k1)),
                );
                let d = _mm_add_epi16(
                    _mm_add_epi16(x1, x3),
                    _mm_add_epi16(_mm_mulhi_epi16(x1, k1), _mm_mulhi_epi16(x3, k2)),
                );
                let mut t = [
                    _mm_add_epi16(a, d),
                    _mm_add_epi16(b, c),
                    _mm_sub_epi16(b, c),
                    _mm_sub_epi16(a, d),
                ];
                if shift {
                    for v in &mut t {
                        *v = _mm_srai_epi16::<3>(*v);
                    }
                }
                transpose_2_4x4(t[0], t[1], t[2], t[3])
            };
        let t = pass(in0, in1, in2, in3, 0, false);
        let t = pass(t[0], t[1], t[2], t[3], 4, true);

        let zero = _mm_setzero_si128();
        for (y, ty) in t.iter().enumerate() {
            let row = _mm_unpacklo_epi8(loadl(r, y * BPS), zero);
            let px = _mm_packus_epi16(_mm_add_epi16(row, *ty), _mm_add_epi16(row, *ty));
            _mm_storel_epi64(dst.as_mut_ptr().add(y * BPS).cast(), px);
        }
    }
}

#[inline(always)]
unsafe fn ftransform_pass1(in01: __m128i, in23: __m128i) -> (__m128i, __m128i) {
    let k937 = _mm_set1_epi32(937);
    let k1812 = _mm_set1_epi32(1812);
    let k88p = _mm_set_epi16(8, 8, 8, 8, 8, 8, 8, 8);
    let k88m = _mm_set_epi16(-8, 8, -8, 8, -8, 8, -8, 8);
    let k5352_2217p = _mm_set_epi16(2217, 5352, 2217, 5352, 2217, 5352, 2217, 5352);
    let k5352_2217m = _mm_set_epi16(-5352, 2217, -5352, 2217, -5352, 2217, -5352, 2217);
    let shuf01_p = _mm_shufflehi_epi16::<0b10_11_00_01>(in01);
    let shuf23_p = _mm_shufflehi_epi16::<0b10_11_00_01>(in23);
    let s01 = _mm_unpacklo_epi64(shuf01_p, shuf23_p);
    let s32 = _mm_unpackhi_epi64(shuf01_p, shuf23_p);
    let a01 = _mm_add_epi16(s01, s32);
    let a32 = _mm_sub_epi16(s01, s32);
    let tmp0 = _mm_madd_epi16(a01, k88p);
    let tmp2 = _mm_madd_epi16(a01, k88m);
    let tmp1 = _mm_srai_epi32::<9>(_mm_add_epi32(_mm_madd_epi16(a32, k5352_2217p), k1812));
    let tmp3 = _mm_srai_epi32::<9>(_mm_add_epi32(_mm_madd_epi16(a32, k5352_2217m), k937));
    let s03 = _mm_packs_epi32(tmp0, tmp2);
    let s12 = _mm_packs_epi32(tmp1, tmp3);
    let s_lo = _mm_unpacklo_epi16(s03, s12);
    let s_hi = _mm_unpackhi_epi16(s03, s12);
    let v23 = _mm_unpackhi_epi32(s_lo, s_hi);
    (
        _mm_unpacklo_epi32(s_lo, s_hi),
        _mm_shuffle_epi32::<0b01_00_11_10>(v23),
    )
}

#[inline(always)]
unsafe fn ftransform_pass2(v01: __m128i, v32: __m128i, out: &mut [i16]) {
    let out = &mut out[..16];
    let zero = _mm_setzero_si128();
    let seven = _mm_set1_epi16(7);
    let k5352_2217 = _mm_set_epi16(5352, 2217, 5352, 2217, 5352, 2217, 5352, 2217);
    let k2217_5352 = _mm_set_epi16(2217, -5352, 2217, -5352, 2217, -5352, 2217, -5352);
    let k12000_plus_one = _mm_set1_epi32(12000 + (1 << 16));
    let k51000 = _mm_set1_epi32(51000);
    let a32 = _mm_sub_epi16(v01, v32);
    let a22 = _mm_unpackhi_epi64(a32, a32);
    let b23 = _mm_unpacklo_epi16(a22, a32);
    let e1 = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_madd_epi16(b23, k5352_2217),
        k12000_plus_one,
    ));
    let e3 = _mm_srai_epi32::<16>(_mm_add_epi32(_mm_madd_epi16(b23, k2217_5352), k51000));
    let f1 = _mm_packs_epi32(e1, e1);
    let f3 = _mm_packs_epi32(e3, e3);
    let g1 = _mm_add_epi16(f1, _mm_cmpeq_epi16(a32, zero));
    let a01 = _mm_add_epi16(v01, v32);
    let a01_plus_7 = _mm_add_epi16(a01, seven);
    let a11 = _mm_unpackhi_epi64(a01, a01);
    let d0 = _mm_srai_epi16::<4>(_mm_add_epi16(a01_plus_7, a11));
    let d2 = _mm_srai_epi16::<4>(_mm_sub_epi16(a01_plus_7, a11));
    _mm_storeu_si128(out.as_mut_ptr().cast(), _mm_unpacklo_epi64(d0, g1));
    _mm_storeu_si128(out.as_mut_ptr().add(8).cast(), _mm_unpacklo_epi64(d2, f3));
}

#[inline]
pub(crate) fn ftransform(src: &[u8], r: &[u8], out: &mut [i16]) {
    let src = &src[..3 * BPS + 8];
    let r = &r[..3 * BPS + 8];
    unsafe {
        let zero = _mm_setzero_si128();
        let src_0 = _mm_unpacklo_epi16(loadl(src, 0), loadl(src, BPS));
        let src_1 = _mm_unpacklo_epi16(loadl(src, 2 * BPS), loadl(src, 3 * BPS));
        let ref_0 = _mm_unpacklo_epi16(loadl(r, 0), loadl(r, BPS));
        let ref_1 = _mm_unpacklo_epi16(loadl(r, 2 * BPS), loadl(r, 3 * BPS));
        let row01 = _mm_sub_epi16(
            _mm_unpacklo_epi8(src_0, zero),
            _mm_unpacklo_epi8(ref_0, zero),
        );
        let row23 = _mm_sub_epi16(
            _mm_unpacklo_epi8(src_1, zero),
            _mm_unpacklo_epi8(ref_1, zero),
        );
        let (v01, v32) = ftransform_pass1(row01, row23);
        ftransform_pass2(v01, v32, out);
    }
}

#[inline]
pub(crate) fn ftransform2(src: &[u8], r: &[u8], out: &mut [i16]) {
    let src = &src[..3 * BPS + 8];
    let r = &r[..3 * BPS + 8];
    let out = &mut out[..32];
    unsafe {
        let zero = _mm_setzero_si128();
        let diff = |y: usize| {
            _mm_sub_epi16(
                _mm_unpacklo_epi8(loadl(src, y * BPS), zero),
                _mm_unpacklo_epi8(loadl(r, y * BPS), zero),
            )
        };
        let (d0, d1, d2, d3) = (diff(0), diff(1), diff(2), diff(3));
        let (v01l, v32l) = ftransform_pass1(_mm_unpacklo_epi32(d0, d1), _mm_unpacklo_epi32(d2, d3));
        let (v01h, v32h) = ftransform_pass1(_mm_unpackhi_epi32(d0, d1), _mm_unpackhi_epi32(d2, d3));
        let (lo, hi) = out.split_at_mut(16);
        ftransform_pass2(v01l, v32l, lo);
        ftransform_pass2(v01h, v32h, hi);
    }
}

#[inline(always)]
unsafe fn subtract_and_accumulate(a: __m128i, b: __m128i) -> __m128i {
    let abs_a_b = _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    let zero = _mm_setzero_si128();
    let c0 = _mm_unpacklo_epi8(abs_a_b, zero);
    let c1 = _mm_unpackhi_epi8(abs_a_b, zero);
    _mm_add_epi32(_mm_madd_epi16(c0, c0), _mm_madd_epi16(c1, c1))
}

#[inline]
pub(crate) fn sse16xn(a: &[u8], b: &[u8], rows: usize) -> i32 {
    let a = &a[..(rows - 1) * BPS + 16];
    let b = &b[..(rows - 1) * BPS + 16];
    unsafe {
        let mut sum = _mm_setzero_si128();
        for y in 0..rows {
            let va = _mm_loadu_si128(a.as_ptr().add(y * BPS).cast());
            let vb = _mm_loadu_si128(b.as_ptr().add(y * BPS).cast());
            sum = _mm_add_epi32(sum, subtract_and_accumulate(va, vb));
        }
        hsum(sum)
    }
}

#[inline]
pub(crate) fn sse4x4(a: &[u8], b: &[u8]) -> i32 {
    let a = &a[..3 * BPS + 8];
    let b = &b[..3 * BPS + 8];
    unsafe {
        let zero = _mm_setzero_si128();
        let a01 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(loadl(a, 0), loadl(a, BPS)), zero);
        let a23 = _mm_unpacklo_epi8(
            _mm_unpacklo_epi32(loadl(a, 2 * BPS), loadl(a, 3 * BPS)),
            zero,
        );
        let b01 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(loadl(b, 0), loadl(b, BPS)), zero);
        let b23 = _mm_unpacklo_epi8(
            _mm_unpacklo_epi32(loadl(b, 2 * BPS), loadl(b, 3 * BPS)),
            zero,
        );
        let d0 = _mm_subs_epi16(a01, b01);
        let d1 = _mm_subs_epi16(a23, b23);
        hsum(_mm_add_epi32(
            _mm_madd_epi16(d0, d0),
            _mm_madd_epi16(d1, d1),
        ))
    }
}

/// Weighted sum of the Hadamard coefficients of `a` minus that of `b`.
#[inline]
pub(crate) fn ttransform(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32 {
    let a = &a[..3 * BPS + 8];
    let b = &b[..3 * BPS + 8];
    unsafe {
        let zero = _mm_setzero_si128();
        let ld = |y: usize| {
            _mm_unpacklo_epi8(
                _mm_unpacklo_epi32(loadl(a, y * BPS), loadl(b, y * BPS)),
                zero,
            )
        };
        ttransform_rows([ld(0), ld(1), ld(2), ld(3)], w)
    }
}

/// `ttransform` of rows holding the 16-bit pixels of row `y` of `a` then of `b`.
#[inline(always)]
unsafe fn ttransform_rows([t0, t1, t2, t3]: [__m128i; 4], w: &[u16; 16]) -> i32 {
    let zero = _mm_setzero_si128();
    let butterfly = |t0: __m128i, t1: __m128i, t2: __m128i, t3: __m128i| {
        let a0 = _mm_add_epi16(t0, t2);
        let a1 = _mm_add_epi16(t1, t3);
        let a2 = _mm_sub_epi16(t1, t3);
        let a3 = _mm_sub_epi16(t0, t2);
        [
            _mm_add_epi16(a0, a1),
            _mm_add_epi16(a3, a2),
            _mm_sub_epi16(a3, a2),
            _mm_sub_epi16(a0, a1),
        ]
    };
    let b4 = butterfly(t0, t1, t2, t3);
    let t = transpose_2_4x4(b4[0], b4[1], b4[2], b4[3]);
    let b4 = butterfly(t[0], t[1], t[2], t[3]);
    let w_0 = _mm_loadu_si128(w.as_ptr().cast());
    let w_8 = _mm_loadu_si128(w.as_ptr().add(8).cast());
    let abs = |v: __m128i| _mm_max_epi16(v, _mm_sub_epi16(zero, v));
    let a_b0 = abs(_mm_unpacklo_epi64(b4[0], b4[1]));
    let a_b2 = abs(_mm_unpacklo_epi64(b4[2], b4[3]));
    let b_b0 = abs(_mm_unpackhi_epi64(b4[0], b4[1]));
    let b_b2 = abs(_mm_unpackhi_epi64(b4[2], b4[3]));
    let sa = _mm_add_epi32(_mm_madd_epi16(a_b0, w_0), _mm_madd_epi16(a_b2, w_8));
    let sb = _mm_add_epi32(_mm_madd_epi16(b_b0, w_0), _mm_madd_epi16(b_b2, w_8));
    hsum(_mm_sub_epi32(sa, sb))
}

#[inline]
pub(crate) fn quantize_block(input: &mut [i16], out: &mut [i16], mtx: &Matrix) -> bool {
    let input = &mut input[..16];
    let out = &mut out[..16];
    unsafe {
        let max_coeff_2047 = _mm_set1_epi16(MAX_LEVEL as i16);
        let zero = _mm_setzero_si128();
        let in0 = load_i16x8(input, 0);
        let in8 = load_i16x8(input, 8);
        let iq0 = _mm_loadu_si128(mtx.iq.as_ptr().cast());
        let iq8 = _mm_loadu_si128(mtx.iq.as_ptr().add(8).cast());
        let q0 = _mm_loadu_si128(mtx.q.as_ptr().cast());
        let q8 = _mm_loadu_si128(mtx.q.as_ptr().add(8).cast());
        let sign0 = _mm_cmpgt_epi16(zero, in0);
        let sign8 = _mm_cmpgt_epi16(zero, in8);
        let coeff0 = _mm_add_epi16(
            _mm_sub_epi16(_mm_xor_si128(in0, sign0), sign0),
            _mm_loadu_si128(mtx.sharpen.as_ptr().cast()),
        );
        let coeff8 = _mm_add_epi16(
            _mm_sub_epi16(_mm_xor_si128(in8, sign8), sign8),
            _mm_loadu_si128(mtx.sharpen.as_ptr().add(8).cast()),
        );

        let h0 = _mm_mulhi_epu16(coeff0, iq0);
        let l0 = _mm_mullo_epi16(coeff0, iq0);
        let h8 = _mm_mulhi_epu16(coeff8, iq8);
        let l8 = _mm_mullo_epi16(coeff8, iq8);
        let bias = |k: usize| _mm_loadu_si128(mtx.bias.as_ptr().add(k).cast());
        let o00 =
            _mm_srai_epi32::<{ QFIX as i32 }>(_mm_add_epi32(_mm_unpacklo_epi16(l0, h0), bias(0)));
        let o04 =
            _mm_srai_epi32::<{ QFIX as i32 }>(_mm_add_epi32(_mm_unpackhi_epi16(l0, h0), bias(4)));
        let o08 =
            _mm_srai_epi32::<{ QFIX as i32 }>(_mm_add_epi32(_mm_unpacklo_epi16(l8, h8), bias(8)));
        let o12 =
            _mm_srai_epi32::<{ QFIX as i32 }>(_mm_add_epi32(_mm_unpackhi_epi16(l8, h8), bias(12)));
        let out0 = _mm_min_epi16(_mm_packs_epi32(o00, o04), max_coeff_2047);
        let out8 = _mm_min_epi16(_mm_packs_epi32(o08, o12), max_coeff_2047);
        let out0 = _mm_sub_epi16(_mm_xor_si128(out0, sign0), sign0);
        let out8 = _mm_sub_epi16(_mm_xor_si128(out8, sign8), sign8);

        _mm_storeu_si128(input.as_mut_ptr().cast(), _mm_mullo_epi16(out0, q0));
        _mm_storeu_si128(input.as_mut_ptr().add(8).cast(), _mm_mullo_epi16(out8, q8));

        let z0 = _mm_shufflehi_epi16::<0b10_01_11_00>(out0);
        let z0 = _mm_shuffle_epi32::<0b11_01_10_00>(z0);
        let z0 = _mm_shufflehi_epi16::<0b11_01_00_10>(z0);
        let z8 = _mm_shufflelo_epi16::<0b11_00_10_01>(out8);
        let z8 = _mm_shuffle_epi32::<0b11_01_10_00>(z8);
        let z8 = _mm_shufflelo_epi16::<0b01_11_10_00>(z8);
        _mm_storeu_si128(out.as_mut_ptr().cast(), z0);
        _mm_storeu_si128(out.as_mut_ptr().add(8).cast(), z8);
        let packed = _mm_packs_epi16(z0, z8);
        out.swap(3, 12);
        _mm_movemask_epi8(_mm_cmpeq_epi8(packed, zero)) != 0xffff
    }
}

/// The luma of the leading pixels of an RGBA row, 16 at a time, as libwebp's
/// `ConvertRGBToY_SSE2`: the green weight 33059 does not fit an i16, so it is split in two
/// products. Returns how many it did.
pub(crate) fn rgba_to_y_row(line: &[u8], out: &mut [u8]) -> usize {
    let n = out.len().min(line.len() / 4) & !15;
    // SAFETY: the loads read line[..n * 4] and the stores write out[..n].
    unsafe {
        let mask = _mm_set1_epi32(0x00ff_00ff);
        let k_rb = _mm_set1_epi32(6420 << 16 | 16839);
        let k_g0 = _mm_set1_epi32(33059 - 16384);
        let k_g1 = _mm_set1_epi32(16384);
        let rounder = _mm_set1_epi32((16 << 16) + (1 << 15));
        let luma = |p: *const u8| {
            let v = _mm_loadu_si128(p.cast());
            let rb = _mm_and_si128(v, mask);
            let ga = _mm_and_si128(_mm_srli_epi32::<8>(v), mask);
            let y = _mm_add_epi32(
                _mm_madd_epi16(rb, k_rb),
                _mm_add_epi32(_mm_madd_epi16(ga, k_g0), _mm_madd_epi16(ga, k_g1)),
            );
            _mm_srli_epi32::<16>(_mm_add_epi32(y, rounder))
        };
        for i in (0..n).step_by(16) {
            let p = line.as_ptr().add(i * 4);
            let lo = _mm_packs_epi32(luma(p), luma(p.add(16)));
            let hi = _mm_packs_epi32(luma(p.add(32)), luma(p.add(48)));
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_packus_epi16(lo, hi));
        }
    }
    n
}
