/// 16 bit fixed point version of cos(PI/8) * sqrt(2) - 1
const CONST1: i64 = 20091;
/// 16 bit fixed point version of sin(PI/8) * sqrt(2)
const CONST2: i64 = 35468;

// inverse discrete cosine transform, used in decoding
pub(crate) fn idct4x4(block: &mut [i32]) {
    // The intermediate results may overflow the types, so we stretch the type.
    fn fetch(block: &[i32], idx: usize) -> i64 {
        i64::from(block[idx])
    }

    // Perform one length check up front to avoid subsequent bounds checks in this function
    assert!(block.len() >= 16);

    for i in 0usize..4 {
        let a1 = fetch(block, i) + fetch(block, 8 + i);
        let b1 = fetch(block, i) - fetch(block, 8 + i);

        let t1 = (fetch(block, 4 + i) * CONST2) >> 16;
        let t2 = fetch(block, 12 + i) + ((fetch(block, 12 + i) * CONST1) >> 16);
        let c1 = t1 - t2;

        let t1 = fetch(block, 4 + i) + ((fetch(block, 4 + i) * CONST1) >> 16);
        let t2 = (fetch(block, 12 + i) * CONST2) >> 16;
        let d1 = t1 + t2;

        block[i] = (a1 + d1) as i32;
        block[4 + i] = (b1 + c1) as i32;
        block[4 * 3 + i] = (a1 - d1) as i32;
        block[4 * 2 + i] = (b1 - c1) as i32;
    }

    for i in 0usize..4 {
        let a1 = fetch(block, 4 * i) + fetch(block, 4 * i + 2);
        let b1 = fetch(block, 4 * i) - fetch(block, 4 * i + 2);

        let t1 = (fetch(block, 4 * i + 1) * CONST2) >> 16;
        let t2 = fetch(block, 4 * i + 3) + ((fetch(block, 4 * i + 3) * CONST1) >> 16);
        let c1 = t1 - t2;

        let t1 = fetch(block, 4 * i + 1) + ((fetch(block, 4 * i + 1) * CONST1) >> 16);
        let t2 = (fetch(block, 4 * i + 3) * CONST2) >> 16;
        let d1 = t1 + t2;

        block[4 * i] = ((a1 + d1 + 4) >> 3) as i32;
        block[4 * i + 3] = ((a1 - d1 + 4) >> 3) as i32;
        block[4 * i + 1] = ((b1 + c1 + 4) >> 3) as i32;
        block[4 * i + 2] = ((b1 - c1 + 4) >> 3) as i32;
    }
}

// The kinds of inverse transform a block needs, after libwebp's `NzCodeBits` classes.
/// A block without coefficients to add.
pub(crate) const NO_COEFFS: u8 = 0;
/// A block with only a DC coefficient (libwebp's `TransformDC`).
pub(crate) const DC_ONLY: u8 = 1;
/// A luma block with coefficients only among the first three in zigzag order, large enough that
/// libwebp's `TransformAC3` gives other pixels than the 16-bit full transform.
pub(crate) const AC3: u8 = 2;
/// A block that needs the full inverse transform.
pub(crate) const FULL: u8 = 3;

/// The kind of transform of a chroma block, from the end `n` of its coefficients as libwebp's
/// `GetCoeffs` returns it and its DC.
#[inline(always)]
pub(crate) fn kind(n: usize, dc: i32) -> u8 {
    if n > 1 {
        FULL
    } else if dc != 0 {
        DC_ONLY
    } else {
        NO_COEFFS
    }
}

/// The kind of transform of a luma block, as [`kind`].
#[inline(always)]
pub(crate) fn luma_kind(n: usize, block: &[i32; 16]) -> u8 {
    if n > 1 {
        // libwebp has no SIMD TransformAC3 on x86: its C version computes in 32 bits. The 16-bit
        // full transform gives the same pixels while the coefficients cannot wrap.
        if cfg!(target_arch = "x86_64")
            && n <= 3
            && ((block[0] + 8192) | (block[1] + 8192) | (block[4] + 8192)) as u32 >= 16384
        {
            AC3
        } else {
            FULL
        }
    } else if block[0] != 0 {
        DC_ONLY
    } else {
        NO_COEFFS
    }
}

/// Adds the inverse transform of luma `block` (16 coefficients), of `kind`, to the 4x4 pixels
/// at `pos` of `dst`, as libwebp's `DoTransform`.
#[inline]
pub(crate) fn add_coeffs(dst: &mut [u8], block: &[i32], kind: u8, pos: usize, stride: usize) {
    match kind {
        NO_COEFFS => {}
        DC_ONLY => add_dc((block[0] + 4) >> 3, dst, pos, stride),
        AC3 => transform_ac3_add(block, dst, pos, stride),
        _ => transform_add(block, dst, pos, stride),
    }
}

/// Adds the inverse transforms of the four `blocks` of a chroma plane, of `kinds`, to its 8x8
/// pixels at `pos` of `dst`. As libwebp's `DoUVTransform` on x86, a plane with AC coefficients
/// in any block takes the full transform on all four, two at a time.
#[inline(always)]
pub(crate) fn add_coeffs_uv(
    dst: &mut [u8],
    blocks: &[i32],
    kinds: &[u8],
    pos: usize,
    stride: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if kinds[..4].contains(&FULL) {
        super::simd::transform_add_two(&blocks[..32], dst, pos, stride);
        super::simd::transform_add_two(&blocks[32..64], dst, pos + 4 * stride, stride);
        return;
    }
    for (i, &kind) in kinds[..4].iter().enumerate() {
        let block = &blocks[i * 16..][..16];
        let pos = pos + (i / 2) * 4 * stride + (i % 2) * 4;
        match kind {
            NO_COEFFS => {}
            DC_ONLY => add_dc((block[0] + 4) >> 3, dst, pos, stride),
            _ => transform_add(block, dst, pos, stride),
        }
    }
}

#[inline(always)]
fn add_dc(dc: i32, dst: &mut [u8], pos: usize, stride: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    super::simd::add_dc(dc, dst, pos, stride);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    add_dc_portable(dc, dst, pos, stride);
}

#[inline(always)]
fn transform_add(block: &[i32], dst: &mut [u8], pos: usize, stride: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    super::simd::transform_add(block, dst, pos, stride);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    transform_add_portable(block, dst, pos, stride);
}

/// libwebp's `TransformAC3_C`: the inverse transform of `block`, whose only coefficients are
/// 0, 1 and 4, added to the 4x4 pixels at `pos` of `dst`.
#[cold]
#[inline(never)]
fn transform_ac3_add(block: &[i32], dst: &mut [u8], pos: usize, stride: usize) {
    let mul1 = |a: i32| ((a * 20091) >> 16) + a;
    let mul2 = |a: i32| (a * 35468) >> 16;
    let a = block[0] + 4;
    let (c4, d4) = (mul2(block[4]), mul1(block[4]));
    let (c1, d1) = (mul2(block[1]), mul1(block[1]));
    for (y, dc) in [a + d4, a + c4, a - c4, a - d4].into_iter().enumerate() {
        let row = &mut dst[pos + y * stride..][..4];
        for (p, v) in row.iter_mut().zip([dc + d1, dc + c1, dc - c1, dc - d1]) {
            *p = (i32::from(*p) + (v >> 3)).clamp(0, 255) as u8;
        }
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn add_dc_portable(dc: i32, dst: &mut [u8], pos: usize, stride: usize) {
    for y in 0..4 {
        for p in &mut dst[pos + y * stride..][..4] {
            *p = (i32::from(*p) + dc).clamp(0, 255) as u8;
        }
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn transform_add_portable(block: &[i32], dst: &mut [u8], pos: usize, stride: usize) {
    let mut b: [i32; 16] = block[..16].try_into().unwrap();
    idct4x4(&mut b);
    for (y, row) in b.chunks_exact(4).enumerate() {
        for (p, &a) in dst[pos + y * stride..][..4].iter_mut().zip(row) {
            *p = (a + i32::from(*p)).clamp(0, 255) as u8;
        }
    }
}

// 14.3 inverse walsh-hadamard transform, used in decoding
pub(crate) fn iwht4x4(block: &mut [i32]) {
    // Perform one length check up front to avoid subsequent bounds checks in this function
    assert!(block.len() >= 16);

    for i in 0usize..4 {
        let a1 = block[i] + block[12 + i];
        let b1 = block[4 + i] + block[8 + i];
        let c1 = block[4 + i] - block[8 + i];
        let d1 = block[i] - block[12 + i];

        block[i] = a1 + b1;
        block[4 + i] = c1 + d1;
        block[8 + i] = a1 - b1;
        block[12 + i] = d1 - c1;
    }

    for block in block.chunks_exact_mut(4) {
        let a1 = block[0] + block[3];
        let b1 = block[1] + block[2];
        let c1 = block[1] - block[2];
        let d1 = block[0] - block[3];

        let a2 = a1 + b1;
        let b2 = c1 + d1;
        let c2 = a1 - b1;
        let d2 = d1 - c1;

        block[0] = (a2 + 3) >> 3;
        block[1] = (b2 + 3) >> 3;
        block[2] = (c2 + 3) >> 3;
        block[3] = (d2 + 3) >> 3;
    }
}

#[cfg(test)]
pub(crate) fn dct4x4(block: &mut [i32; 16]) {
    // The intermediate results may overflow the types, so we stretch the type.
    fn fetch(block: &[i32], idx: usize) -> i64 {
        i64::from(block[idx])
    }

    // vertical
    for i in 0..4 {
        let a = (fetch(block, i * 4) + fetch(block, i * 4 + 3)) * 8;
        let b = (fetch(block, i * 4 + 1) + fetch(block, i * 4 + 2)) * 8;
        let c = (fetch(block, i * 4 + 1) - fetch(block, i * 4 + 2)) * 8;
        let d = (fetch(block, i * 4) - fetch(block, i * 4 + 3)) * 8;

        block[i * 4] = (a + b) as i32;
        block[i * 4 + 2] = (a - b) as i32;
        block[i * 4 + 1] = ((c * 2217 + d * 5352 + 14500) >> 12) as i32;
        block[i * 4 + 3] = ((d * 2217 - c * 5352 + 7500) >> 12) as i32;
    }

    // horizontal
    for i in 0..4 {
        let a = fetch(block, i) + fetch(block, i + 12);
        let b = fetch(block, i + 4) + fetch(block, i + 8);
        let c = fetch(block, i + 4) - fetch(block, i + 8);
        let d = fetch(block, i) - fetch(block, i + 12);

        block[i] = ((a + b + 7) >> 4) as i32;
        block[i + 8] = ((a - b + 7) >> 4) as i32;
        block[i + 4] = (((c * 2217 + d * 5352 + 12000) >> 16) + if d != 0 { 1 } else { 0 }) as i32;
        block[i + 12] = ((d * 2217 - c * 5352 + 51000) >> 16) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_inverse() {
        const BLOCK: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];

        let mut dct_block = BLOCK.clone();

        dct4x4(&mut dct_block);

        let mut inverse_dct_block = dct_block.clone();

        idct4x4(&mut inverse_dct_block);

        assert_eq!(BLOCK, inverse_dct_block);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn simd_transform_matches_portable() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for round in 0..20000 {
            let range = [8, 100, 1000][round % 3];
            let block: [i32; 16] = core::array::from_fn(|_| rng.gen_range(-range..=range));
            let stride = 9;
            let dst: Vec<u8> = (0..4 * stride).map(|_| rng.gen()).collect();
            let (mut a, mut b) = (dst.clone(), dst);
            transform_add_portable(&block, &mut a, 1, stride);
            crate::lossy::simd::transform_add(&block, &mut b, 1, stride);
            assert_eq!(a, b, "{block:?}");
            let dc = (block[0] * 4 + 4) >> 3;
            add_dc_portable(dc, &mut a, 1, stride);
            crate::lossy::simd::add_dc(dc, &mut b, 1, stride);
            assert_eq!(a, b, "dc {dc}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn two_blocks_match_one_at_a_time() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let stride = 13;
        for round in 0..20000 {
            let range = [8, 1000, 32767][round % 3];
            let blocks: [i32; 32] = core::array::from_fn(|_| rng.gen_range(-range..=range));
            let dst: Vec<u8> = (0..4 * stride).map(|_| rng.gen()).collect();
            let (mut a, mut b) = (dst.clone(), dst);
            crate::lossy::simd::transform_add(&blocks[..16], &mut a, 1, stride);
            crate::lossy::simd::transform_add(&blocks[16..], &mut a, 5, stride);
            crate::lossy::simd::transform_add_two(&blocks, &mut b, 1, stride);
            assert_eq!(a, b, "{blocks:?}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ac3_matches_the_32_bit_transform() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);
        let stride = 9;
        for round in 0..30000 {
            let range = [100, 2048, 8192, 32767][round % 4];
            let mut block = [0i32; 16];
            for k in [0, 1, 4] {
                block[k] = rng.gen_range(-range..=range);
            }
            if round < 8 {
                block[0] = [32767, -32768][round & 1];
                block[1] = [32767, -32768][(round >> 1) & 1];
                block[4] = [32767, -32768][(round >> 2) & 1];
            }
            let dst: Vec<u8> = (0..4 * stride).map(|_| rng.gen()).collect();
            let (mut a, mut b, mut c) = (dst.clone(), dst.clone(), dst);
            transform_add_portable(&block, &mut a, 1, stride);
            transform_ac3_add(&block, &mut b, 1, stride);
            assert_eq!(a, b, "{block:?}");
            add_coeffs(&mut c, &block, luma_kind(3, &block), 1, stride);
            assert_eq!(a, c, "{block:?}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn chroma_planes_with_ac_take_the_full_transform_on_every_block() {
        let stride = 9;
        for dc in i32::from(i16::MIN)..=i32::from(i16::MAX) {
            let mut blocks = [0i32; 64];
            blocks[0] = dc;
            blocks[16 + 1] = 1;
            let dst: Vec<u8> = (0..8 * stride).map(|i| (i * 37) as u8).collect();
            let (mut a, mut b) = (dst.clone(), dst);
            add_coeffs_uv(&mut a, &blocks, &[DC_ONLY, FULL, 0, 0], 1, stride);
            crate::lossy::simd::transform_add(&blocks[..16], &mut b, 1, stride);
            crate::lossy::simd::transform_add(&blocks[16..32], &mut b, 5, stride);
            assert_eq!(a, b, "dc {dc}");
        }
    }
}
