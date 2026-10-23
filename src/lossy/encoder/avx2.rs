//! AVX2 widenings of the kernels in `sse2.rs`. libwebp has no AVX2 for the VP8 encoder, so
//! each 128-bit lane runs the SSE2 kernel on a block of its own and the results equal the
//! SSE2 ones exactly. Callers check `crate::wide::has_avx2()` first.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

use super::dsp::{
    I4Recon, Matrix, BPS, I4DC4, I4HD4, I4_LOOKUP1, I4_LOOKUP2, I4_LOOKUP3, I4_MODE_OFFSETS,
    MAX_LEVEL, QFIX, ZIGZAG,
};

#[inline(always)]
fn ld32(p: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(p[off..off + 4].try_into().unwrap())
}

/// The same 128 bits in both lanes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn both(v: __m128i) -> __m256i {
    _mm256_broadcastsi128_si256(v)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_both<T>(p: &[T]) -> __m256i {
    assert!(core::mem::size_of_val(p) >= 16);
    both(_mm_loadu_si128(p.as_ptr().cast()))
}

/// The rows `y` and `y + 1` of the 4x4 blocks `a` (low lane) and `b` (high lane) as 16-bit.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn rows2(a: &[u8], b: &[u8], y: usize) -> __m256i {
    let a = _mm_unpacklo_epi32(
        _mm_cvtsi32_si128(ld32(a, y * BPS)),
        _mm_cvtsi32_si128(ld32(a, (y + 1) * BPS)),
    );
    let b = _mm_unpacklo_epi32(
        _mm_cvtsi32_si128(ld32(b, y * BPS)),
        _mm_cvtsi32_si128(ld32(b, (y + 1) * BPS)),
    );
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(a, b))
}

/// Rows `y` and `y + 1` of the 4x4 blocks `a` (low lane) and `b` (high lane), 16-bit and
/// interleaved in pairs as `sse2::ftransform_regs` loads them.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn rows2_paired(a: &[u8], b: &[u8], y: usize) -> __m256i {
    let a = _mm_unpacklo_epi16(
        _mm_cvtsi32_si128(ld32(a, y * BPS)),
        _mm_cvtsi32_si128(ld32(a, (y + 1) * BPS)),
    );
    let b = _mm_unpacklo_epi16(
        _mm_cvtsi32_si128(ld32(b, y * BPS)),
        _mm_cvtsi32_si128(ld32(b, (y + 1) * BPS)),
    );
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(a, b))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transpose_2_4x4(i0: __m256i, i1: __m256i, i2: __m256i, i3: __m256i) -> [__m256i; 4] {
    let t0_0 = _mm256_unpacklo_epi16(i0, i1);
    let t0_1 = _mm256_unpacklo_epi16(i2, i3);
    let t0_2 = _mm256_unpackhi_epi16(i0, i1);
    let t0_3 = _mm256_unpackhi_epi16(i2, i3);
    let t1_0 = _mm256_unpacklo_epi32(t0_0, t0_1);
    let t1_1 = _mm256_unpacklo_epi32(t0_2, t0_3);
    let t1_2 = _mm256_unpackhi_epi32(t0_0, t0_1);
    let t1_3 = _mm256_unpackhi_epi32(t0_2, t0_3);
    [
        _mm256_unpacklo_epi64(t1_0, t1_1),
        _mm256_unpackhi_epi64(t1_0, t1_1),
        _mm256_unpacklo_epi64(t1_2, t1_3),
        _mm256_unpackhi_epi64(t1_2, t1_3),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn ftransform_pass1(in01: __m256i, in23: __m256i) -> (__m256i, __m256i) {
    let k937 = _mm256_set1_epi32(937);
    let k1812 = _mm256_set1_epi32(1812);
    let k88p = _mm256_set1_epi16(8);
    let k88m = _mm256_set1_epi32(0xfff8_0008_u32 as i32);
    let k5352_2217p = _mm256_set1_epi32(2217 << 16 | 5352);
    let k5352_2217m = _mm256_set1_epi32((-5352i32 << 16) | 2217);
    let shuf01_p = _mm256_shufflehi_epi16::<0b10_11_00_01>(in01);
    let shuf23_p = _mm256_shufflehi_epi16::<0b10_11_00_01>(in23);
    let s01 = _mm256_unpacklo_epi64(shuf01_p, shuf23_p);
    let s32 = _mm256_unpackhi_epi64(shuf01_p, shuf23_p);
    let a01 = _mm256_add_epi16(s01, s32);
    let a32 = _mm256_sub_epi16(s01, s32);
    let tmp0 = _mm256_madd_epi16(a01, k88p);
    let tmp2 = _mm256_madd_epi16(a01, k88m);
    let tmp1 = _mm256_srai_epi32::<9>(_mm256_add_epi32(_mm256_madd_epi16(a32, k5352_2217p), k1812));
    let tmp3 = _mm256_srai_epi32::<9>(_mm256_add_epi32(_mm256_madd_epi16(a32, k5352_2217m), k937));
    let s03 = _mm256_packs_epi32(tmp0, tmp2);
    let s12 = _mm256_packs_epi32(tmp1, tmp3);
    let s_lo = _mm256_unpacklo_epi16(s03, s12);
    let s_hi = _mm256_unpackhi_epi16(s03, s12);
    let v23 = _mm256_unpackhi_epi32(s_lo, s_hi);
    (
        _mm256_unpacklo_epi32(s_lo, s_hi),
        _mm256_shuffle_epi32::<0b01_00_11_10>(v23),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn ftransform_pass2(v01: __m256i, v32: __m256i) -> [__m256i; 2] {
    let zero = _mm256_setzero_si256();
    let seven = _mm256_set1_epi16(7);
    let k5352_2217 = _mm256_set1_epi32(5352 << 16 | 2217);
    let k2217_5352 = _mm256_set1_epi32(2217 << 16 | (-5352i32 & 0xffff));
    let k12000_plus_one = _mm256_set1_epi32(12000 + (1 << 16));
    let k51000 = _mm256_set1_epi32(51000);
    let a32 = _mm256_sub_epi16(v01, v32);
    let a22 = _mm256_unpackhi_epi64(a32, a32);
    let b23 = _mm256_unpacklo_epi16(a22, a32);
    let e1 = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_madd_epi16(b23, k5352_2217),
        k12000_plus_one,
    ));
    let e3 = _mm256_srai_epi32::<16>(_mm256_add_epi32(_mm256_madd_epi16(b23, k2217_5352), k51000));
    let f1 = _mm256_packs_epi32(e1, e1);
    let f3 = _mm256_packs_epi32(e3, e3);
    let g1 = _mm256_add_epi16(f1, _mm256_cmpeq_epi16(a32, zero));
    let a01 = _mm256_add_epi16(v01, v32);
    let a01_plus_7 = _mm256_add_epi16(a01, seven);
    let a11 = _mm256_unpackhi_epi64(a01, a01);
    let d0 = _mm256_srai_epi16::<4>(_mm256_add_epi16(a01_plus_7, a11));
    let d2 = _mm256_srai_epi16::<4>(_mm256_sub_epi16(a01_plus_7, a11));
    [_mm256_unpacklo_epi64(d0, g1), _mm256_unpacklo_epi64(d2, f3)]
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn quant_shift(v: __m256i, bias: &[u32]) -> __m256i {
    _mm256_srai_epi32::<{ QFIX as i32 }>(_mm256_add_epi32(v, load_both(bias)))
}

/// `sse2::quantize_regs` of one block per lane: coefficients 0..8 in `in0`, 8..16 in `in8`.
/// Returns the dequantized coefficients the same way, and the levels (zigzag, levels 3 and 12
/// not yet swapped) of the low lane in `.1[0]` / `.1[1]`'s low halves and the high lane in
/// their high halves.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn quantize_regs(in0: __m256i, in8: __m256i, mtx: &Matrix) -> ([__m256i; 2], [__m256i; 2]) {
    let max_coeff_2047 = _mm256_set1_epi16(MAX_LEVEL as i16);
    let zero = _mm256_setzero_si256();
    let iq0 = load_both(&mtx.iq[..8]);
    let iq8 = load_both(&mtx.iq[8..]);
    let q0 = load_both(&mtx.q[..8]);
    let q8 = load_both(&mtx.q[8..]);
    let sign0 = _mm256_cmpgt_epi16(zero, in0);
    let sign8 = _mm256_cmpgt_epi16(zero, in8);
    let coeff0 = _mm256_add_epi16(
        _mm256_sub_epi16(_mm256_xor_si256(in0, sign0), sign0),
        load_both(&mtx.sharpen[..8]),
    );
    let coeff8 = _mm256_add_epi16(
        _mm256_sub_epi16(_mm256_xor_si256(in8, sign8), sign8),
        load_both(&mtx.sharpen[8..]),
    );
    let h0 = _mm256_mulhi_epu16(coeff0, iq0);
    let l0 = _mm256_mullo_epi16(coeff0, iq0);
    let h8 = _mm256_mulhi_epu16(coeff8, iq8);
    let l8 = _mm256_mullo_epi16(coeff8, iq8);
    let o00 = quant_shift(_mm256_unpacklo_epi16(l0, h0), &mtx.bias[0..4]);
    let o04 = quant_shift(_mm256_unpackhi_epi16(l0, h0), &mtx.bias[4..8]);
    let o08 = quant_shift(_mm256_unpacklo_epi16(l8, h8), &mtx.bias[8..12]);
    let o12 = quant_shift(_mm256_unpackhi_epi16(l8, h8), &mtx.bias[12..16]);
    let out0 = _mm256_min_epi16(_mm256_packs_epi32(o00, o04), max_coeff_2047);
    let out8 = _mm256_min_epi16(_mm256_packs_epi32(o08, o12), max_coeff_2047);
    let out0 = _mm256_sub_epi16(_mm256_xor_si256(out0, sign0), sign0);
    let out8 = _mm256_sub_epi16(_mm256_xor_si256(out8, sign8), sign8);
    let dequant = [_mm256_mullo_epi16(out0, q0), _mm256_mullo_epi16(out8, q8)];

    let z0 = _mm256_shufflehi_epi16::<0b10_01_11_00>(out0);
    let z0 = _mm256_shuffle_epi32::<0b11_01_10_00>(z0);
    let z0 = _mm256_shufflehi_epi16::<0b11_01_00_10>(z0);
    let z8 = _mm256_shufflelo_epi16::<0b11_00_10_01>(out8);
    let z8 = _mm256_shuffle_epi32::<0b11_01_10_00>(z8);
    let z8 = _mm256_shufflelo_epi16::<0b01_11_10_00>(z8);
    (dequant, [z0, z8])
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn itransform_pass<const SHIFT: bool>(
    x01: __m256i,
    x23: __m256i,
    round: __m256i,
) -> (__m256i, __m256i) {
    let k1k2 = _mm256_set_epi64x(
        0x8a8c_8a8c_8a8c_8a8c_u64 as i64,
        0x4e7b_4e7b_4e7b_4e7b,
        0x8a8c_8a8c_8a8c_8a8c_u64 as i64,
        0x4e7b_4e7b_4e7b_4e7b,
    );
    let k2k1 = _mm256_shuffle_epi32::<0b01_00_11_10>(k1k2);
    let x1 = _mm256_unpackhi_epi64(x01, x01);
    let x3 = _mm256_unpackhi_epi64(x23, x23);
    let x01 = _mm256_add_epi16(x01, round);
    let a_d3 = _mm256_add_epi16(x01, x23);
    let b_c3 = _mm256_sub_epi16(x01, x23);
    let c1d1 = _mm256_mulhi_epi16(x1, k2k1);
    let c2d2 = _mm256_mulhi_epi16(x3, k1k2);
    let c3 = _mm256_unpackhi_epi64(b_c3, b_c3);
    let c4 = _mm256_sub_epi16(c1d1, c2d2);
    let c = _mm256_add_epi16(c3, c4);
    let d4u = _mm256_add_epi16(c1d1, c2d2);
    let du = _mm256_add_epi16(a_d3, d4u);
    let d = _mm256_unpackhi_epi64(du, du);
    let comb_ab = _mm256_unpacklo_epi64(a_d3, b_c3);
    let comb_dc = _mm256_unpacklo_epi64(d, c);
    let mut tmp01 = _mm256_add_epi16(comb_ab, comb_dc);
    let tmp32 = _mm256_sub_epi16(comb_ab, comb_dc);
    let mut tmp23 = _mm256_shuffle_epi32::<0b01_00_11_10>(tmp32);
    if SHIFT {
        tmp01 = _mm256_srai_epi16::<3>(tmp01);
        tmp23 = _mm256_srai_epi16::<3>(tmp23);
    }
    let t0 = _mm256_unpacklo_epi16(tmp01, tmp23);
    let t1 = _mm256_unpackhi_epi16(tmp01, tmp23);
    (_mm256_unpacklo_epi16(t0, t1), _mm256_unpackhi_epi16(t0, t1))
}

/// The inverse transform of one block per lane, added to the prediction rows `ref01`/`ref23`
/// (16-bit), returning each lane's four rows packed.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn itransform_one_rows(
    [in01, in23]: [__m256i; 2],
    ref01: __m256i,
    ref23: __m256i,
) -> __m256i {
    let zero = _mm256_setzero_si256();
    let zero_four = _mm256_set_epi64x(0, 0x0004_0004_0004_0004, 0, 0x0004_0004_0004_0004);

    let (t01, t23) = itransform_pass::<false>(in01, in23, zero);
    let (t01, t23) = itransform_pass::<true>(t01, t23, zero_four);
    _mm256_packus_epi16(_mm256_add_epi16(ref01, t01), _mm256_add_epi16(ref23, t23))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hadamard4(t0: __m256i, t1: __m256i, t2: __m256i, t3: __m256i) -> [__m256i; 4] {
    let a0 = _mm256_add_epi16(t0, t2);
    let a1 = _mm256_add_epi16(t1, t3);
    let a2 = _mm256_sub_epi16(t1, t3);
    let a3 = _mm256_sub_epi16(t0, t2);
    [
        _mm256_add_epi16(a0, a1),
        _mm256_add_epi16(a3, a2),
        _mm256_sub_epi16(a3, a2),
        _mm256_sub_epi16(a0, a1),
    ]
}

/// `sse2::ttransform_rows` of one pair of blocks per lane, before the horizontal sum.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn ttransform_rows([t0, t1, t2, t3]: [__m256i; 4], w: &[u16; 16]) -> __m256i {
    let b4 = hadamard4(t0, t1, t2, t3);
    let t = transpose_2_4x4(b4[0], b4[1], b4[2], b4[3]);
    let b4 = hadamard4(t[0], t[1], t[2], t[3]);
    let w_0 = load_both(&w[..8]);
    let w_8 = load_both(&w[8..]);
    let a_b0 = _mm256_abs_epi16(_mm256_unpacklo_epi64(b4[0], b4[1]));
    let a_b2 = _mm256_abs_epi16(_mm256_unpacklo_epi64(b4[2], b4[3]));
    let b_b0 = _mm256_abs_epi16(_mm256_unpackhi_epi64(b4[0], b4[1]));
    let b_b2 = _mm256_abs_epi16(_mm256_unpackhi_epi64(b4[2], b4[3]));
    let sa = _mm256_add_epi32(_mm256_madd_epi16(a_b0, w_0), _mm256_madd_epi16(a_b2, w_8));
    let sb = _mm256_add_epi32(_mm256_madd_epi16(b_b0, w_0), _mm256_madd_epi16(b_b2, w_8));
    _mm256_sub_epi32(sa, sb)
}

/// `reconstruct4x4_disto_x2` of the ten 4x4 predictions in `preds`; `None` without AVX2.
#[inline]
pub(crate) fn reconstruct4x4_disto_modes(
    src: &[u8],
    preds: &[u8],
    mtx: &Matrix,
    w: Option<&[u16; 16]>,
) -> Option<[I4Recon; 10]> {
    if !crate::wide::has_avx2() {
        return None;
    }
    let mut out = [I4Recon::default(); 10];
    for (m, out) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let r = [
            &preds[I4_MODE_OFFSETS[2 * m]..],
            &preds[I4_MODE_OFFSETS[2 * m + 1]..],
        ];
        // SAFETY: the CPU supports AVX2.
        unsafe { reconstruct4x4_disto_x2(src, r, mtx, w, out) };
    }
    Some(out)
}

/// `sse2::reconstruct4x4_disto` of the 4x4 block `src` against the predictions `r[0]` and
/// `r[1]`, one per lane, the reconstructions packed into `I4Recon::pixels`. `disto` is the
/// raw `ttransform` sum.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn reconstruct4x4_disto_x2(
    src: &[u8],
    r: [&[u8]; 2],
    mtx: &Matrix,
    w: Option<&[u16; 16]>,
    out: &mut [I4Recon],
) {
    let (r0, r1) = (&r[0][..3 * BPS + 4], &r[1][..3 * BPS + 4]);
    let out = &mut out[..2];
    let [c0, c8] = ftransform_x2(src, r0, r1);
    let (dequant, [z0, z8]) = quantize_regs(c0, c8, mtx);
    let packed = _mm256_packs_epi16(z0, z8);
    let non_zero = !_mm256_movemask_epi8(_mm256_cmpeq_epi8(packed, _mm256_setzero_si256())) as u32;
    let mut lv = [[0i16; 8]; 4];
    _mm256_storeu_si256(lv[0..2].as_mut_ptr().cast(), z0);
    _mm256_storeu_si256(lv[2..4].as_mut_ptr().cast(), z8);
    for (k, o) in out.iter_mut().enumerate() {
        o.levels[..8].copy_from_slice(&lv[k]);
        o.levels[8..].copy_from_slice(&lv[2 + k]);
        o.levels.swap(3, 12);
        let nz = non_zero >> (16 * k) & 0xffff;
        let swapped = ((nz >> 3) ^ (nz >> 12)) & 1;
        o.nz = nz ^ (swapped << 3 | swapped << 12);
    }
    itransform_disto_x2(src, r0, r1, dequant, w, out);
}

/// `sse2::ftransform_regs` of `src - r0` (low lane) and `src - r1` (high lane).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn ftransform_x2(src: &[u8], r0: &[u8], r1: &[u8]) -> [__m256i; 2] {
    let src = &src[..3 * BPS + 4];
    let row01 = _mm256_sub_epi16(rows2_paired(src, src, 0), rows2_paired(r0, r1, 0));
    let row23 = _mm256_sub_epi16(rows2_paired(src, src, 2), rows2_paired(r0, r1, 2));
    let (v01, v32) = ftransform_pass1(row01, row23);
    ftransform_pass2(v01, v32)
}

/// The inverse transforms of `coeffs` (coefficients 0..8 and 8..16, a block per lane) added
/// to `r0` and `r1`, packed into `out[..2].pixels` with their SSE and raw TDisto against
/// `src`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn itransform_disto_x2(
    src: &[u8],
    r0: &[u8],
    r1: &[u8],
    coeffs: [__m256i; 2],
    w: Option<&[u16; 16]>,
    out: &mut [I4Recon],
) {
    let src = &src[..3 * BPS + 4];
    let zero = _mm256_setzero_si256();
    let packed = itransform_one_rows(coeffs, rows2(r0, r1, 0), rows2(r0, r1, 2));
    let s01 = rows2(src, src, 0);
    let s23 = rows2(src, src, 2);
    let o01 = _mm256_unpacklo_epi8(packed, zero);
    let o23 = _mm256_unpackhi_epi8(packed, zero);
    let d01 = _mm256_sub_epi16(s01, o01);
    let d23 = _mm256_sub_epi16(s23, o23);
    let sse = _mm256_add_epi32(_mm256_madd_epi16(d01, d01), _mm256_madd_epi16(d23, d23));
    let disto = match w {
        Some(w) => ttransform_rows(
            [
                _mm256_unpacklo_epi64(s01, o01),
                _mm256_unpackhi_epi64(s01, o01),
                _mm256_unpacklo_epi64(s23, o23),
                _mm256_unpackhi_epi64(s23, o23),
            ],
            w,
        ),
        None => zero,
    };
    let sums = _mm256_hadd_epi32(sse, disto);
    let sums = _mm256_hadd_epi32(sums, sums);
    let mut px = [[0u8; 16]; 2];
    _mm256_storeu_si256(px.as_mut_ptr().cast(), packed);
    let sums: [i32; 8] = core::mem::transmute(sums);
    for (k, o) in out[..2].iter_mut().enumerate() {
        o.pixels = px[k];
        o.d = sums[4 * k];
        o.disto = sums[4 * k + 1];
    }
}

/// `sse2::ftransform` of `src` against each of the ten 4x4 predictions in `preds`, two per
/// register.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ftransform_modes(src: &[u8], preds: &[u8], out: &mut [[i16; 16]; 10]) {
    for (m, out) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let r0 = &preds[I4_MODE_OFFSETS[2 * m]..][..3 * BPS + 4];
        let r1 = &preds[I4_MODE_OFFSETS[2 * m + 1]..][..3 * BPS + 4];
        let [c0, c8] = ftransform_x2(src, r0, r1);
        store_i16x16(&mut out[0], _mm256_permute2x128_si256::<0x20>(c0, c8));
        store_i16x16(&mut out[1], _mm256_permute2x128_si256::<0x31>(c0, c8));
    }
}

/// `itransform_one_disto` of each `coeffs[m]` added to the prediction of mode `m` in
/// `preds`, two per register, into `out[m]`'s `pixels`, `d` and raw `disto`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn itransform_disto_modes(
    src: &[u8],
    preds: &[u8],
    coeffs: &[[i16; 16]; 10],
    w: Option<&[u16; 16]>,
    out: &mut [I4Recon; 10],
) {
    for (m, out) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let r0 = &preds[I4_MODE_OFFSETS[2 * m]..][..3 * BPS + 4];
        let r1 = &preds[I4_MODE_OFFSETS[2 * m + 1]..][..3 * BPS + 4];
        let a = load_i16x16(&coeffs[2 * m]);
        let b = load_i16x16(&coeffs[2 * m + 1]);
        let c = [
            _mm256_permute2x128_si256::<0x20>(a, b),
            _mm256_permute2x128_si256::<0x31>(a, b),
        ];
        itransform_disto_x2(src, r0, r1, c, w, out);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load16(p: &[u8], off: usize) -> __m256i {
    _mm256_cvtepu8_epi16(_mm_loadu_si128(p[off..off + 16].as_ptr().cast()))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_i16x16(p: &[i16]) -> __m256i {
    _mm256_loadu_si256(p[..16].as_ptr().cast())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_i16x16(p: &mut [i16], v: __m256i) {
    _mm256_storeu_si256(p[..16].as_mut_ptr().cast(), v)
}

/// The coefficients of the four 4x4 blocks side by side at `src - r`: blocks 0 | 2 rows
/// 0-1, 0 | 2 rows 2-3, then the same of blocks 1 | 3.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn ftransform4_regs(src: &[u8], r: &[u8]) -> [__m256i; 4] {
    let src = &src[..3 * BPS + 16];
    let r = &r[..3 * BPS + 16];
    let d0 = _mm256_sub_epi16(load16(src, 0), load16(r, 0));
    let d1 = _mm256_sub_epi16(load16(src, BPS), load16(r, BPS));
    let d2 = _mm256_sub_epi16(load16(src, 2 * BPS), load16(r, 2 * BPS));
    let d3 = _mm256_sub_epi16(load16(src, 3 * BPS), load16(r, 3 * BPS));
    let (v01l, v32l) =
        ftransform_pass1(_mm256_unpacklo_epi32(d0, d1), _mm256_unpacklo_epi32(d2, d3));
    let (v01h, v32h) =
        ftransform_pass1(_mm256_unpackhi_epi32(d0, d1), _mm256_unpackhi_epi32(d2, d3));
    let [l0, l8] = ftransform_pass2(v01l, v32l);
    let [h0, h8] = ftransform_pass2(v01h, v32h);
    [l0, l8, h0, h8]
}

/// `sse2::ftransform2` of the two pairs of 4x4 blocks side by side at `src - r`, the left
/// pair into `lo` and the right one into `hi`, 32 coefficients each.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ftransform4(src: &[u8], r: &[u8], lo: &mut [i16], hi: &mut [i16]) {
    let [l0, l8, h0, h8] = ftransform4_regs(src, r);
    store_i16x16(&mut lo[..16], _mm256_permute2x128_si256::<0x20>(l0, l8));
    store_i16x16(&mut lo[16..32], _mm256_permute2x128_si256::<0x20>(h0, h8));
    store_i16x16(&mut hi[..16], _mm256_permute2x128_si256::<0x31>(l0, l8));
    store_i16x16(&mut hi[16..32], _mm256_permute2x128_si256::<0x31>(h0, h8));
}

/// `dsp::collect_histogram`'s counts of the blocks in the 16-pixel rows at `rows` of
/// `src - r`, into `distribution`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn histogram_rows(
    src: &[u8],
    r: &[u8],
    rows: &[usize],
    distribution: &mut [[u16; 32]; 4],
) {
    let max = _mm256_set1_epi16(31);
    for &off in rows {
        let c = ftransform4_regs(&src[off..], &r[off..]);
        let bin = |v| _mm256_min_epu16(_mm256_srli_epi16::<3>(_mm256_abs_epi16(v)), max);
        let mut bins = [0u8; 64];
        let b01 = _mm256_packus_epi16(bin(c[0]), bin(c[1]));
        let b23 = _mm256_packus_epi16(bin(c[2]), bin(c[3]));
        _mm256_storeu_si256(bins.as_mut_ptr().cast(), b01);
        _mm256_storeu_si256(bins.as_mut_ptr().add(32).cast(), b23);
        for b in bins.as_chunks::<4>().0 {
            for (h, &b) in distribution.iter_mut().zip(b) {
                h[usize::from(b)] += 1;
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn itransform_two_pass<const SHIFT: bool>(
    x0: __m256i,
    x1: __m256i,
    x2: __m256i,
    x3: __m256i,
    dc_add: i16,
) -> [__m256i; 4] {
    let k1 = _mm256_set1_epi16(20091);
    let k2 = _mm256_set1_epi16(-30068);
    let dc = _mm256_add_epi16(x0, _mm256_set1_epi16(dc_add));
    let a = _mm256_add_epi16(dc, x2);
    let b = _mm256_sub_epi16(dc, x2);
    let c = _mm256_add_epi16(
        _mm256_sub_epi16(x1, x3),
        _mm256_sub_epi16(_mm256_mulhi_epi16(x1, k2), _mm256_mulhi_epi16(x3, k1)),
    );
    let d = _mm256_add_epi16(
        _mm256_add_epi16(x1, x3),
        _mm256_add_epi16(_mm256_mulhi_epi16(x1, k1), _mm256_mulhi_epi16(x3, k2)),
    );
    let mut t = [
        _mm256_add_epi16(a, d),
        _mm256_add_epi16(b, c),
        _mm256_sub_epi16(b, c),
        _mm256_sub_epi16(a, d),
    ];
    if SHIFT {
        for v in &mut t {
            *v = _mm256_srai_epi16::<3>(*v);
        }
    }
    transpose_2_4x4(t[0], t[1], t[2], t[3])
}

/// `sse2::itransform_two` of the pairs of blocks `lo` and `hi` (32 coefficients each) added
/// to the 16-pixel rows of `r`, into `dst`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn itransform4(r: &[u8], lo: &[i16], hi: &[i16], dst: &mut [u8]) {
    let r = &r[..3 * BPS + 16];
    let dst = &mut dst[..3 * BPS + 16];
    let (b0, b1) = (load_i16x16(&lo[..16]), load_i16x16(&lo[16..32]));
    let (b2, b3) = (load_i16x16(&hi[..16]), load_i16x16(&hi[16..32]));
    let tmp0 = _mm256_permute2x128_si256::<0x20>(b0, b2);
    let tmp1 = _mm256_permute2x128_si256::<0x31>(b0, b2);
    let tmp2 = _mm256_permute2x128_si256::<0x20>(b1, b3);
    let tmp3 = _mm256_permute2x128_si256::<0x31>(b1, b3);
    let t = itransform_two_pass::<false>(
        _mm256_unpacklo_epi64(tmp0, tmp2),
        _mm256_unpackhi_epi64(tmp0, tmp2),
        _mm256_unpacklo_epi64(tmp1, tmp3),
        _mm256_unpackhi_epi64(tmp1, tmp3),
        0,
    );
    let t = itransform_two_pass::<true>(t[0], t[1], t[2], t[3], 4);
    for y in [0, 2] {
        let a = _mm256_add_epi16(load16(r, y * BPS), t[y]);
        let b = _mm256_add_epi16(load16(r, (y + 1) * BPS), t[y + 1]);
        let px = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(a, b));
        let row0 = &mut dst[y * BPS..y * BPS + 16];
        _mm_storeu_si128(row0.as_mut_ptr().cast(), _mm256_castsi256_si128(px));
        let row1 = &mut dst[(y + 1) * BPS..(y + 1) * BPS + 16];
        _mm_storeu_si128(row1.as_mut_ptr().cast(), _mm256_extracti128_si256::<1>(px));
    }
}

/// `dsp::tdisto16x16`: the 4x4 blocks two per register, each lane `sse2::ttransform`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn tdisto16x16(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32 {
    let a = &a[..15 * BPS + 16];
    let b = &b[..15 * BPS + 16];
    let mut acc = _mm256_setzero_si256();
    for y in (0..16).step_by(4) {
        let (mut left, mut right) = ([_mm256_setzero_si256(); 4], [_mm256_setzero_si256(); 4]);
        for (i, (l, r)) in left.iter_mut().zip(&mut right).enumerate() {
            let off = (y + i) * BPS;
            let va = _mm_loadu_si128(a[off..off + 16].as_ptr().cast());
            let vb = _mm_loadu_si128(b[off..off + 16].as_ptr().cast());
            *l = _mm256_cvtepu8_epi16(_mm_unpacklo_epi32(va, vb));
            *r = _mm256_cvtepu8_epi16(_mm_unpackhi_epi32(va, vb));
        }
        let t01 = ttransform_rows(left, w);
        let t23 = ttransform_rows(right, w);
        let h = _mm256_hadd_epi32(t01, t23);
        let h = _mm256_hadd_epi32(h, h);
        acc = _mm256_add_epi32(acc, _mm256_srai_epi32::<5>(_mm256_abs_epi32(h)));
    }
    let acc: [i32; 8] = core::mem::transmute(acc);
    acc[0] + acc[1] + acc[4] + acc[5]
}

/// For each 8-bit mask of coefficients `8 * HALF..8 * HALF + 8` (natural order), the same bits
/// at their zigzag positions.
const fn zigzag_bits<const HALF: usize>() -> [u16; 256] {
    let mut inv = [0usize; 16];
    let mut n = 0;
    while n < 16 {
        inv[ZIGZAG[n]] = n;
        n += 1;
    }
    let mut out = [0u16; 256];
    let mut m = 0;
    while m < 256 {
        let mut j = 0;
        while j < 8 {
            if m >> j & 1 != 0 {
                out[m] |= 1 << inv[8 * HALF + j];
            }
            j += 1;
        }
        m += 1;
    }
    out
}

static ZIGZAG_LO: [u16; 256] = zigzag_bits::<0>();
static ZIGZAG_HI: [u16; 256] = zigzag_bits::<1>();

/// The trellis' last candidate position: the highest `n` in `first..16` whose coefficient
/// `input[ZIGZAG[n]]` squared is above `thresh`, else `first - 1`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn trellis_last(input: &[i16; 16], first: usize, thresh: i32) -> i32 {
    let v = load_i16x16(input);
    let lo = _mm256_mullo_epi16(v, v);
    let hi = _mm256_mulhi_epi16(v, v);
    let t = _mm256_set1_epi32(thresh);
    let above = |sq| _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(sq, t))) as u32;
    // Lanes hold coefficients 0..4 | 8..12 and 4..8 | 12..16.
    let a = above(_mm256_unpacklo_epi16(lo, hi));
    let b = above(_mm256_unpackhi_epi16(lo, hi));
    let natural = (a & 0xf) | (b & 0xf) << 4 | (a >> 4) << 8 | (b >> 4) << 12;
    let zz = ZIGZAG_LO[(natural & 0xff) as usize] | ZIGZAG_HI[(natural >> 8) as usize];
    let zz = u32::from(zz) & !((1u32 << first) - 1);
    if zz == 0 {
        first as i32 - 1
    } else {
        31 - zz.leading_zeros() as i32
    }
}

/// Four Y in each lane's 32-bit words from the 16-bit `rb` (r, b pairs) and `g0` (g, 0 pairs).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn luma4(rb: __m256i, g0: __m256i) -> __m256i {
    let k_rb = _mm256_set1_epi32(6420 << 16 | 16839);
    let k_g0 = _mm256_set1_epi32(33059 - 16384);
    let k_g1 = _mm256_set1_epi32(16384);
    let rounder = _mm256_set1_epi32((16 << 16) + (1 << 15));
    let y = _mm256_add_epi32(
        _mm256_madd_epi16(rb, k_rb),
        _mm256_add_epi32(_mm256_madd_epi16(g0, k_g0), _mm256_madd_epi16(g0, k_g1)),
    );
    _mm256_srli_epi32::<16>(_mm256_add_epi32(y, rounder))
}

/// `sse2::rgba_to_y_row`, 32 pixels at a time. Returns how many it did.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgba_to_y_row(line: &[u8], out: &mut [u8]) -> usize {
    let n = out.len().min(line.len() / 4) & !31;
    let line = &line[..n * 4];
    let out = &mut out[..n];
    let mask = _mm256_set1_epi32(0x00ff_00ff);
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (px, out) in line
        .as_chunks::<128>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<32>().0)
    {
        let mut y = [_mm256_setzero_si256(); 4];
        for (k, y) in y.iter_mut().enumerate() {
            let v = _mm256_loadu_si256(px[32 * k..32 * k + 32].as_ptr().cast());
            let rb = _mm256_and_si256(v, mask);
            let ga = _mm256_and_si256(_mm256_srli_epi32::<8>(v), mask);
            *y = luma4(rb, ga);
        }
        let packed = _mm256_packus_epi16(
            _mm256_packs_epi32(y[0], y[1]),
            _mm256_packs_epi32(y[2], y[3]),
        );
        let packed = _mm256_permutevar8x32_epi32(packed, order);
        _mm256_storeu_si256(out.as_mut_ptr().cast(), packed);
    }
    n
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn split6(v: [__m256i; 6]) -> [__m256i; 6] {
    [
        _mm256_unpacklo_epi8(v[0], v[3]),
        _mm256_unpackhi_epi8(v[0], v[3]),
        _mm256_unpacklo_epi8(v[1], v[4]),
        _mm256_unpackhi_epi8(v[1], v[4]),
        _mm256_unpacklo_epi8(v[2], v[5]),
        _mm256_unpackhi_epi8(v[2], v[5]),
    ]
}

/// Eight Y as 16-bit from the 16-bit `r`, `g` and `b` of each lane.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn luma8(r: __m256i, g: __m256i, b: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let lo = luma4(_mm256_unpacklo_epi16(r, b), _mm256_unpacklo_epi16(g, zero));
    let hi = luma4(_mm256_unpackhi_epi16(r, b), _mm256_unpackhi_epi16(g, zero));
    _mm256_packs_epi32(lo, hi)
}

/// `sse2::rgb_to_y_row`, 64 pixels at a time: each lane deinterleaves 32 of them as the
/// SSE2 kernel does. Returns how many pixels it converted.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb_to_y_row(line: &[u8], out: &mut [u8]) -> usize {
    let n = out.len().min(line.len() / 3) & !63;
    let line = &line[..n * 3];
    let out = &mut out[..n];
    let zero = _mm256_setzero_si256();
    for (px, out) in line
        .as_chunks::<192>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<64>().0)
    {
        let mut v = [zero; 6];
        for (k, v) in v.iter_mut().enumerate() {
            let lo = _mm_loadu_si128(px[16 * k..16 * k + 16].as_ptr().cast());
            let hi = _mm_loadu_si128(px[96 + 16 * k..112 + 16 * k].as_ptr().cast());
            *v = _mm256_set_m128i(hi, lo);
        }
        let planes = split6(split6(split6(split6(split6(v)))));
        let mut y = [zero; 2];
        for (j, y) in y.iter_mut().enumerate() {
            let (r, g, b) = (planes[j], planes[2 + j], planes[4 + j]);
            let y0 = luma8(
                _mm256_unpacklo_epi8(r, zero),
                _mm256_unpacklo_epi8(g, zero),
                _mm256_unpacklo_epi8(b, zero),
            );
            let y1 = luma8(
                _mm256_unpackhi_epi8(r, zero),
                _mm256_unpackhi_epi8(g, zero),
                _mm256_unpackhi_epi8(b, zero),
            );
            *y = _mm256_packus_epi16(y0, y1);
        }
        // y[j] holds pixels 16j.. of the first 32 in its low lane, of the second in its high.
        let first = _mm256_permute2x128_si256::<0x20>(y[0], y[1]);
        let second = _mm256_permute2x128_si256::<0x31>(y[0], y[1]);
        _mm256_storeu_si256(out.as_mut_ptr().cast(), first);
        _mm256_storeu_si256(out.as_mut_ptr().add(32).cast(), second);
    }
    n
}

/// The `pshufb` indices of `pred_luma4`'s rows into `avg2` (`TABLE` 0) or `avg3` (1), 0x80
/// where the other one holds the byte: per row DC TM VE HE in the low lane and RD VR LD VL
/// in the high one, then HD HU of rows 0-1 and 2-3.
const fn pred4_lookup<const TABLE: u8>() -> [[u8; 32]; 5] {
    let mut out = [[0x80u8; 32]; 5];
    let mut r = 0;
    while r < 5 {
        let mut i = 0;
        while i < 32 {
            let v = if r == 4 {
                I4_LOOKUP1[32 + i]
            } else if i < 16 {
                I4_LOOKUP3[16 * r + i]
            } else {
                I4_LOOKUP2[16 * r + i - 16]
            };
            if v >> 4 == TABLE {
                out[r][i] = v & 15;
            }
            i += 1;
        }
        r += 1;
    }
    out
}

const PRED4_AVG2: [[u8; 32]; 5] = pred4_lookup::<0>();
const PRED4_AVG3: [[u8; 32]; 5] = pred4_lookup::<1>();

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pred4_row(avg2: __m256i, avg3: __m256i, r: usize) -> __m256i {
    let lo = _mm256_loadu_si256(PRED4_AVG2[r].as_ptr().cast());
    let hi = _mm256_loadu_si256(PRED4_AVG3[r].as_ptr().cast());
    _mm256_or_si256(_mm256_shuffle_epi8(avg2, lo), _mm256_shuffle_epi8(avg3, hi))
}

/// `dsp::pred_luma4` as `neon::pred_luma4` computes it: the averages of the edge picked into
/// whole rows of four modes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pred_luma4(dst: &mut [u8], b: &[u8], t: usize) {
    let edge = &b[t - 5..t + 11];
    let dst = &mut dst[I4DC4..I4DC4 + 7 * BPS + 8];
    // L K J I X A B C D E F G H
    let edge = _mm_loadu_si128(edge.as_ptr().cast());
    let prev = _mm_shuffle_epi8(edge, _mm_loadu_si128(I4_LOOKUP1.as_ptr().cast()));
    let next = _mm_srli_si128::<2>(prev);
    let avg2 = _mm_avg_epu8(prev, edge);
    let lsb = _mm_and_si128(_mm_xor_si128(prev, next), _mm_set1_epi8(1));
    let avg3 = _mm_avg_epu8(_mm_subs_epu8(_mm_avg_epu8(prev, next), lsb), edge);
    let dc_in = _mm_and_si128(edge, _mm_setr_epi32(-1, 0xffff_ff00_u32 as i32, 0xff, 0));
    let sad = _mm_sad_epu8(dc_in, _mm_setzero_si128());
    let dc = (_mm_cvtsi128_si32(sad) + _mm_extract_epi16::<4>(sad) + 4) >> 3;
    let avg3 = _mm_insert_epi8::<14>(avg3, dc);
    let avg3 = _mm_insert_epi8::<15>(avg3, _mm_cvtsi128_si32(edge) & 0xff);
    let (avg2, avg3) = (both(avg2), both(avg3));

    // TrueMotion: left + top - top-left, saturated, row y in bytes 4y..4y+4.
    let z = -128;
    let top = _mm_shuffle_epi8(
        edge,
        _mm_setr_epi8(5, z, 6, z, 7, z, 8, z, 5, z, 6, z, 7, z, 8, z),
    );
    let d = _mm_sub_epi16(
        top,
        _mm_shuffle_epi8(
            edge,
            _mm_setr_epi8(4, z, 4, z, 4, z, 4, z, 4, z, 4, z, 4, z, 4, z),
        ),
    );
    let l01 = _mm_shuffle_epi8(
        edge,
        _mm_setr_epi8(3, z, 3, z, 3, z, 3, z, 2, z, 2, z, 2, z, 2, z),
    );
    let l23 = _mm_shuffle_epi8(
        edge,
        _mm_setr_epi8(1, z, 1, z, 1, z, 1, z, 0, z, 0, z, 0, z, 0, z),
    );
    let tm = _mm_packus_epi16(_mm_add_epi16(l01, d), _mm_add_epi16(l23, d));
    let tm = [
        _mm_shuffle_epi32::<0x00>(tm),
        tm,
        _mm_shuffle_epi32::<0xaa>(tm),
        _mm_shuffle_epi32::<0xff>(tm),
    ];
    for (y, tm) in tm.into_iter().enumerate() {
        let row = pred4_row(avg2, avg3, y);
        let row = _mm256_blend_epi32::<0b10>(row, _mm256_castsi128_si256(tm));
        _mm256_storeu_si256(dst[y * BPS..y * BPS + 32].as_mut_ptr().cast(), row);
    }
    let hd_hu = pred4_row(avg2, avg3, 4);
    let (lo, hi) = (
        _mm256_castsi256_si128(hd_hu),
        _mm256_extracti128_si256::<1>(hd_hu),
    );
    let o = I4HD4 - I4DC4;
    for (y, v) in [
        lo,
        _mm_unpackhi_epi64(lo, lo),
        hi,
        _mm_unpackhi_epi64(hi, hi),
    ]
    .into_iter()
    .enumerate()
    {
        _mm_storel_epi64(dst[o + y * BPS..o + y * BPS + 8].as_mut_ptr().cast(), v);
    }
}

#[cfg(test)]
mod tests {
    use super::super::dsp::{bias, I4Recon, Matrix, BPS, QFIX};
    use rand::{Rng, SeedableRng};

    pub(super) fn random_matrix(rng: &mut impl Rng, round: usize) -> Matrix {
        let mut mtx = Matrix::default();
        let q0: u16 = rng.gen_range(4..=314);
        let q1: u16 = rng.gen_range(4..=284);
        mtx.q = [q1; 16];
        mtx.q[0] = q0;
        for j in 0..16 {
            mtx.iq[j] = ((1 << QFIX) / u32::from(mtx.q[j])) as u16;
            mtx.bias[j] = bias([96, 110, 115][round % 3]);
            mtx.zthresh[j] = ((1 << QFIX) - 1 - mtx.bias[j]) / u32::from(mtx.iq[j]);
            mtx.sharpen[j] = if round % 2 == 0 {
                (j as u16 * 6 * mtx.q[j]) >> 11
            } else {
                0
            };
        }
        mtx
    }

    pub(super) fn random_pixels(rng: &mut impl Rng, round: usize, out: &mut [u8]) {
        let spread = [255u8, 16, 3, 0][round % 4];
        let base: u8 = rng.gen();
        for v in out {
            *v = if spread == 0 {
                rng.gen_range(0..2) * 255
            } else {
                base.wrapping_add(rng.gen_range(0..=spread))
            };
        }
    }

    #[test]
    fn reconstruct4x4_disto_x2_matches_sse2() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let w = [38u16, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2];
        for round in 0..20000 {
            let mut src = [0u8; 4 * BPS];
            let mut preds = [0u8; 4 * BPS + 40];
            random_pixels(&mut rng, round, &mut src);
            random_pixels(&mut rng, round + 1, &mut preds);
            let mtx = random_matrix(&mut rng, round);
            let (o0, o1) = (rng.gen_range(0..40), rng.gen_range(0..40));
            for w in [None, Some(&w)] {
                let mut got = [I4Recon::default(); 2];
                unsafe {
                    super::reconstruct4x4_disto_x2(
                        &src,
                        [&preds[o0..], &preds[o1..]],
                        &mtx,
                        w,
                        &mut got,
                    )
                };
                for (g, off) in got.iter().zip([o0, o1]) {
                    let mut levels = [0i16; 16];
                    let mut dst = [0u8; 4 * BPS];
                    let (nz, d, disto) = super::super::sse2::reconstruct4x4_disto(
                        &src,
                        &preds[off..],
                        &mtx,
                        &mut levels,
                        &mut dst,
                        w,
                    );
                    let pixels: [u8; 16] = std::array::from_fn(|i| dst[(i / 4) * BPS + i % 4]);
                    assert_eq!(
                        (g.nz, g.d, g.disto, g.levels, g.pixels),
                        (nz, d, disto, levels, pixels)
                    );
                }
            }
        }
    }

    #[test]
    fn mode_kernels_match_sse2() {
        if !crate::wide::has_avx2() {
            return;
        }
        use super::super::dsp::{I4_MODE_OFFSETS, PRED_SIZE};
        use super::super::sse2;
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let w = [38u16, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2];
        let mut preds = vec![0u8; PRED_SIZE];
        for round in 0..2000 {
            let mut src = [0u8; 4 * BPS];
            random_pixels(&mut rng, round, &mut src);
            random_pixels(&mut rng, round + 1, &mut preds);
            let mut coeffs = [[0i16; 16]; 10];
            unsafe { super::ftransform_modes(&src, &preds, &mut coeffs) };
            for (m, c) in coeffs.iter_mut().enumerate() {
                let mut want = [0i16; 16];
                sse2::ftransform(&src, &preds[I4_MODE_OFFSETS[m]..], &mut want);
                assert_eq!(*c, want);
                for v in c.iter_mut() {
                    *v = rng.gen_range(-2048..=2048) >> [0, 3, 6, 9][round % 4];
                }
            }
            for w in [None, Some(&w)] {
                let mut got = [I4Recon::default(); 10];
                unsafe { super::itransform_disto_modes(&src, &preds, &coeffs, w, &mut got) };
                for (m, g) in got.iter().enumerate() {
                    let mut dst = [0u8; 4 * BPS];
                    let r = &preds[I4_MODE_OFFSETS[m]..];
                    let (d, disto) = sse2::itransform_one_disto(r, &coeffs[m], &mut dst, &src, w);
                    let pixels: [u8; 16] = std::array::from_fn(|i| dst[(i / 4) * BPS + i % 4]);
                    assert_eq!((g.d, g.disto, g.pixels), (d, disto, pixels));
                }
            }
        }
    }

    #[test]
    fn trellis_last_matches_scalar() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);
        for round in 0..100000 {
            let mut input = [0i16; 16];
            for v in &mut input {
                *v = match round % 3 {
                    0 => rng.gen(),
                    1 => rng.gen_range(-40..=40),
                    _ => rng.gen_range(-2..=2) * rng.gen_range(0..2000),
                };
            }
            let thresh = rng.gen_range(0..2000) * [1, 1, 100][round % 3];
            for first in [0, 1] {
                let mut want = first as i32 - 1;
                for n in (first..16).rev() {
                    let j = super::ZIGZAG[n];
                    if i32::from(input[j]) * i32::from(input[j]) > thresh {
                        want = n as i32;
                        break;
                    }
                }
                assert_eq!(unsafe { super::trellis_last(&input, first, thresh) }, want);
            }
        }
    }

    #[test]
    fn row_kernels_match_sse2() {
        if !crate::wide::has_avx2() {
            return;
        }
        use super::super::sse2;
        let mut rng = rand::rngs::StdRng::seed_from_u64(8);
        let w = [38u16, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2];
        for round in 0..5000 {
            let mut a = [0u8; 16 * BPS];
            let mut b = [0u8; 16 * BPS];
            random_pixels(&mut rng, round, &mut a);
            random_pixels(&mut rng, round + 1, &mut b);

            let (mut lo, mut hi) = ([0i16; 32], [0i16; 32]);
            unsafe { super::ftransform4(&a, &b, &mut lo, &mut hi) };
            let (mut lo2, mut hi2) = ([0i16; 32], [0i16; 32]);
            sse2::ftransform2(&a, &b, &mut lo2);
            sse2::ftransform2(&a[8..], &b[8..], &mut hi2);
            assert_eq!((lo, hi), (lo2, hi2));

            let mut coeffs = [0i16; 64];
            for c in &mut coeffs {
                *c = rng.gen_range(-2048..=2048) >> [0, 3, 6, 9][round % 4];
            }
            let (mut d1, mut d2) = ([0u8; 4 * BPS], [0u8; 4 * BPS]);
            unsafe { super::itransform4(&b, &coeffs[..32], &coeffs[32..], &mut d1) };
            sse2::itransform_two(&b, &coeffs[..32], &mut d2);
            sse2::itransform_two(&b[8..], &coeffs[32..], &mut d2[8..]);
            assert_eq!(d1, d2);

            let mut t = 0;
            for y in (0..16 * BPS).step_by(4 * BPS) {
                for x in (0..16).step_by(4) {
                    t += sse2::ttransform(&a[x + y..], &b[x + y..], &w).abs() >> 5;
                }
            }
            assert_eq!(unsafe { super::tdisto16x16(&a, &b, &w) }, t);

            let mut h1 = [[0u16; 32]; 4];
            unsafe { super::histogram_rows(&a, &b, &[0, 4 * BPS, 8 * BPS, 12 * BPS], &mut h1) };
            let mut h2 = [0u16; 32];
            for off in super::super::dsp::SCAN {
                let mut out = [0i16; 16];
                sse2::ftransform(&a[off..], &b[off..], &mut out);
                for v in out {
                    h2[usize::from((v.unsigned_abs() >> 3).min(31))] += 1;
                }
            }
            let h1: [u16; 32] = std::array::from_fn(|k| h1.iter().map(|h| h[k]).sum());
            assert_eq!(h1, h2);
        }
    }
}
