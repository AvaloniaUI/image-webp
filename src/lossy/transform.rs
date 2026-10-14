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

/// A block without coefficients to add.
pub(crate) const NO_COEFFS: u8 = 0;
/// A block with only a DC coefficient (libwebp's `TransformDC`).
pub(crate) const DC_ONLY: u8 = 1;
/// A block that needs the full inverse transform.
pub(crate) const FULL: u8 = 2;

/// Adds the inverse transform of `block` (16 coefficients), of kind [`NO_COEFFS`],
/// [`DC_ONLY`] or [`FULL`], to the 4x4 pixels at `pos` of `dst`.
#[inline]
pub(crate) fn add_coeffs(dst: &mut [u8], block: &[i32], kind: u8, pos: usize, stride: usize) {
    match kind {
        NO_COEFFS => {}
        DC_ONLY => {
            let dc = (block[0] + 4) >> 3;
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            super::simd::add_dc(dc, dst, pos, stride);
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            add_dc_portable(dc, dst, pos, stride);
        }
        _ => {
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            super::simd::transform_add(block, dst, pos, stride);
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            transform_add_portable(block, dst, pos, stride);
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
}
