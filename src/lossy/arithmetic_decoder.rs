// The coefficient reading follows libwebp's GetCoeffsFast and VP8GetBit: Copyright Google Inc.,
// BSD 3-Clause, see LICENSE-LIBWEBP.

use crate::decoder::DecodingError;

use super::common::{PROB_DCT_CAT, ZIGZAG};
use super::TreeNode;

/// Token probabilities of one plane, by band, context and tree node.
pub(crate) type PlaneProbs = [[[u8; 11]; 3]; 8];

/// The band of each coefficient position, and of the position past the last one.
const BANDS: [u8; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];

/// Where `coeffs` reads its bits from.
trait Bits {
    fn bit(&mut self, prob: u8) -> bool;
    fn sign(&mut self) -> bool;
}

impl Bits for FastDecoder<'_> {
    #[inline(always)]
    fn bit(&mut self, prob: u8) -> bool {
        self.fast_read_bit(prob)
    }

    #[inline(always)]
    fn sign(&mut self) -> bool {
        self.fast_read_sign()
    }
}

struct ColdBits<'a> {
    decoder: &'a mut ArithmeticDecoder,
    res: &'a mut BitResultAccumulator,
}

impl Bits for ColdBits<'_> {
    fn bit(&mut self, prob: u8) -> bool {
        self.decoder.cold_read_bit(prob).or_accumulate(self.res)
    }

    fn sign(&mut self) -> bool {
        self.bit(128)
    }
}

/// libwebp's `GetLargeValue`: the magnitude of a token of 2 or more.
#[inline(always)]
fn large_value(b: &mut impl Bits, p: &[u8; 11]) -> i32 {
    let mut bit = |prob| b.bit(prob);
    if !bit(p[3]) {
        if !bit(p[4]) {
            2
        } else {
            3 + i32::from(bit(p[5]))
        }
    } else if !bit(p[6]) {
        if !bit(p[7]) {
            5 + i32::from(bit(159))
        } else {
            let v = 7 + 2 * i32::from(bit(165));
            v + i32::from(bit(145))
        }
    } else {
        let bit1 = usize::from(bit(p[8]));
        let bit0 = usize::from(bit(p[9 + bit1]));
        let cat = 2 * bit1 + bit0;
        let mut v = 0;
        for &t in PROB_DCT_CAT[2 + cat].iter().take_while(|&&t| t != 0) {
            v += v + i32::from(bit(t));
        }
        v + 3 + (8 << cat)
    }
}

/// libwebp's `GetCoeffsFast`: tokens from position `first` on, dequantized by `dq` (DC, AC)
/// into `out` in raster order, as libwebp's 16-bit coefficients. Returns the position after the last token read.
#[inline(always)]
fn coeffs(
    b: &mut impl Bits,
    probs: &PlaneProbs,
    ctx: usize,
    first: usize,
    dq: [i32; 2],
    out: &mut [i32; 16],
) -> usize {
    let mut n = first;
    let mut p = &probs[usize::from(BANDS[n])][ctx];
    while n < 16 {
        if !b.bit(p[0]) {
            return n;
        }
        while !b.bit(p[1]) {
            n += 1;
            if n == 16 {
                return 16;
            }
            p = &probs[usize::from(BANDS[n])][0];
        }
        let p_ctx = &probs[usize::from(BANDS[n + 1])];
        let v = if !b.bit(p[2]) {
            p = &p_ctx[1];
            1
        } else {
            let v = large_value(b, p);
            p = &p_ctx[2];
            v
        };
        let v = if b.sign() { -v } else { v };
        out[usize::from(ZIGZAG[n]) & 15] = i32::from((v * dq[usize::from(n > 0)]) as i16);
        n += 1;
    }
    16
}

#[must_use]
#[repr(transparent)]
pub(crate) struct BitResult<T> {
    value_if_not_past_eof: T,
}

#[must_use]
pub(crate) struct BitResultAccumulator;

impl<T> BitResult<T> {
    const fn ok(value: T) -> Self {
        Self {
            value_if_not_past_eof: value,
        }
    }

    /// Instead of checking this result now, accumulate the burden of checking
    /// into an accumulator. This accumulator must be checked in the end.
    #[inline(always)]
    pub(crate) fn or_accumulate(self, acc: &mut BitResultAccumulator) -> T {
        let _ = acc;
        self.value_if_not_past_eof
    }
}

impl<T: Default> BitResult<T> {
    fn err() -> Self {
        Self {
            value_if_not_past_eof: T::default(),
        }
    }
}

#[cfg_attr(test, derive(Debug))]
pub(crate) struct ArithmeticDecoder {
    chunks: Box<[[u8; 4]]>,
    state: State,
    final_bytes: [u8; 3],
    final_bytes_remaining: i8,
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy)]
struct State {
    chunk_index: usize,
    value: u64,
    /// The range minus one, as libwebp keeps it.
    range: u32,
    bit_count: i32,
}

#[cfg_attr(test, derive(Debug))]
struct FastDecoder<'a> {
    chunks: &'a [[u8; 4]],
    uncommitted_state: State,
    save_state: &'a mut State,
}

impl ArithmeticDecoder {
    pub(crate) fn new() -> ArithmeticDecoder {
        let state = State {
            chunk_index: 0,
            value: 0,
            range: 254,
            bit_count: -8,
        };
        ArithmeticDecoder {
            chunks: Box::new([]),
            state,
            final_bytes: [0; 3],
            final_bytes_remaining: Self::FINAL_BYTES_REMAINING_EOF,
        }
    }

    pub(crate) fn init(&mut self, mut buf: Vec<[u8; 4]>, len: usize) -> Result<(), DecodingError> {
        let mut final_bytes = [0; 3];
        let final_bytes_remaining = if len == 4 * buf.len() {
            0
        } else {
            // Pop the last chunk (which is partial), then get length.
            let Some(last_chunk) = buf.pop() else {
                return Err(DecodingError::NotEnoughInitData);
            };
            let len_rounded_down = 4 * buf.len();
            let num_bytes_popped = len - len_rounded_down;
            debug_assert!(num_bytes_popped <= 3);
            final_bytes[..num_bytes_popped].copy_from_slice(&last_chunk[..num_bytes_popped]);
            for i in num_bytes_popped..4 {
                debug_assert_eq!(last_chunk[i], 0, "unexpected {last_chunk:?}");
            }
            num_bytes_popped as i8
        };

        let chunks = buf.into_boxed_slice();
        let state = State {
            chunk_index: 0,
            value: 0,
            range: 254,
            bit_count: -8,
        };
        *self = Self {
            chunks,
            state,
            final_bytes,
            final_bytes_remaining,
        };
        Ok(())
    }

    /// Start a span of reading operations from the buffer, without stopping
    /// when the buffer runs out. For all valid webp images, the buffer will not
    /// run out prematurely. Conversely if the buffer ends early, the webp image
    /// cannot be correctly decoded and any intermediate results need to be
    /// discarded anyway.
    ///
    /// Each call to `start_accumulated_result` must be followed by a call to
    /// `check` on the *same* `ArithmeticDecoder`.
    #[inline(always)]
    pub(crate) fn start_accumulated_result(&mut self) -> BitResultAccumulator {
        BitResultAccumulator
    }

    /// Check that the read operations done so far were all valid.
    #[inline(always)]
    pub(crate) fn check<T>(
        &self,
        acc: BitResultAccumulator,
        value_if_not_past_eof: T,
    ) -> Result<T, DecodingError> {
        // The accumulator does not store any state because doing so is
        // too computationally expensive. Passing it around is a bit of
        // formality (that is optimized out) to ensure we call `check` .
        // Instead we check whether we have read past the end of the file.
        let BitResultAccumulator = acc;

        if self.is_past_eof() {
            Err(DecodingError::BitStreamError)
        } else {
            Ok(value_if_not_past_eof)
        }
    }

    fn keep_accumulating<T>(
        &self,
        acc: BitResultAccumulator,
        value_if_not_past_eof: T,
    ) -> BitResult<T> {
        // The BitResult will be checked later by a different accumulator.
        // Because it does not carry state, that is fine.
        let BitResultAccumulator = acc;

        BitResult::ok(value_if_not_past_eof)
    }

    // Do not inline this because inlining seems to worsen performance.
    #[inline(never)]
    pub(crate) fn read_bool(&mut self, probability: u8) -> BitResult<bool> {
        if let Some(b) = self.fast().read_bool(probability) {
            return BitResult::ok(b);
        }

        self.cold_read_bool(probability)
    }

    // Do not inline this because inlining seems to worsen performance.
    #[inline(never)]
    pub(crate) fn read_flag(&mut self) -> BitResult<bool> {
        if let Some(b) = self.fast().read_flag() {
            return BitResult::ok(b);
        }

        self.cold_read_flag()
    }

    // Do not inline this because inlining seems to worsen performance.
    #[inline(never)]
    pub(crate) fn read_literal(&mut self, n: u8) -> BitResult<u8> {
        if let Some(v) = self.fast().read_literal(n) {
            return BitResult::ok(v);
        }

        self.cold_read_literal(n)
    }

    // Do not inline this because inlining seems to worsen performance.
    #[inline(never)]
    pub(crate) fn read_optional_signed_value(&mut self, n: u8) -> BitResult<i32> {
        if let Some(v) = self.fast().read_optional_signed_value(n) {
            return BitResult::ok(v);
        }

        self.cold_read_optional_signed_value(n)
    }

    // This is generic and inlined just to skip the first bounds check.
    #[inline]
    pub(crate) fn read_with_tree<const N: usize>(&mut self, tree: &[TreeNode; N]) -> BitResult<i8> {
        let first_node = tree[0];
        self.read_with_tree_with_first_node(tree, first_node)
    }

    // Do not inline this because inlining significantly worsens performance.
    #[inline(never)]
    pub(crate) fn read_with_tree_with_first_node(
        &mut self,
        tree: &[TreeNode],
        first_node: TreeNode,
    ) -> BitResult<i8> {
        if let Some(v) = self.fast().read_with_tree(tree, first_node) {
            return BitResult::ok(v);
        }

        self.cold_read_with_tree(tree, usize::from(first_node.index))
    }

    /// Reads the coefficient tokens of one block, see [`coeffs`]; `out` must be zero from
    /// position `first` on, its DC from the Y2 block is kept when `first` is 1.
    pub(crate) fn read_coeffs(
        &mut self,
        probs: &PlaneProbs,
        ctx: usize,
        first: usize,
        dq: [i32; 2],
        out: &mut [i32; 16],
    ) -> Result<usize, DecodingError> {
        let dc = out[0];
        let mut fast = self.fast();
        let n = coeffs(&mut fast, probs, ctx, first, dq, out);
        if fast.commit_if_valid(()).is_some() {
            return Ok(n);
        }
        *out = [0; 16];
        out[0] = dc;
        self.cold_read_coeffs(probs, ctx, first, dq, out)
    }

    #[cold]
    #[inline(never)]
    fn cold_read_coeffs(
        &mut self,
        probs: &PlaneProbs,
        ctx: usize,
        first: usize,
        dq: [i32; 2],
        out: &mut [i32; 16],
    ) -> Result<usize, DecodingError> {
        let mut res = self.start_accumulated_result();
        let n = coeffs(
            &mut ColdBits {
                decoder: self,
                res: &mut res,
            },
            probs,
            ctx,
            first,
            dq,
            out,
        );
        self.check(res, n)
    }

    // As a similar (but different) speedup to BitResult, the FastDecoder reads
    // bits under an assumption and validates it at the end.
    //
    // The idea here is that for normal-sized webp images, the vast majority
    // of bits are somewhere other than in the last four bytes. Therefore we
    // can pretend the buffer has infinite size. After we are done reading,
    // we check if we actually read past the end of `self.chunks`.
    // If so, we backtrack (or rather we discard `uncommitted_state`)
    // and try again with the slow approach. This might result in doing double
    // work for those last few bytes -- in fact we even keep retrying the fast
    // method to save an if-statement --, but more than make up for that by
    // speeding up reading from the other thousands or millions of bytes.
    fn fast(&mut self) -> FastDecoder<'_> {
        FastDecoder {
            chunks: &self.chunks,
            uncommitted_state: self.state,
            save_state: &mut self.state,
        }
    }

    const FINAL_BYTES_REMAINING_EOF: i8 = -0xE;

    fn load_from_final_bytes(&mut self) {
        match self.final_bytes_remaining {
            1.. => {
                self.final_bytes_remaining -= 1;
                let byte = self.final_bytes[0];
                self.final_bytes.rotate_left(1);
                self.state.value <<= 8;
                self.state.value |= u64::from(byte);
                self.state.bit_count += 8;
            }
            0 => {
                // libwebp seems to (sometimes?) allow bitstreams that read one byte past the end.
                // This replicates that logic.
                self.final_bytes_remaining -= 1;
                self.state.value <<= 8;
                self.state.bit_count += 8;
            }
            _ => {
                self.final_bytes_remaining = Self::FINAL_BYTES_REMAINING_EOF;
            }
        }
    }

    fn is_past_eof(&self) -> bool {
        self.final_bytes_remaining == Self::FINAL_BYTES_REMAINING_EOF
    }

    fn cold_read_bit(&mut self, probability: u8) -> BitResult<bool> {
        if self.state.bit_count < 0 {
            if let Some(chunk) = self.chunks.get(self.state.chunk_index).copied() {
                let v = u32::from_be_bytes(chunk);
                self.state.chunk_index += 1;
                self.state.value <<= 32;
                self.state.value |= u64::from(v);
                self.state.bit_count += 32;
            } else {
                self.load_from_final_bytes();
                if self.is_past_eof() {
                    return BitResult::err();
                }
            }
        }
        debug_assert!(self.state.bit_count >= 0);

        let probability = u32::from(probability);
        let mut range = self.state.range + 1;
        let split = 1 + (((range - 1) * probability) >> 8);
        let bigsplit = u64::from(split) << self.state.bit_count;

        let retval = if let Some(new_value) = self.state.value.checked_sub(bigsplit) {
            range -= split;
            self.state.value = new_value;
            true
        } else {
            range = split;
            false
        };
        debug_assert!(range > 0);

        // Compute shift required to satisfy `range >= 128`.
        // Apply that shift to `range` and `self.state.bitcount`.
        //
        // Subtract 24 because we only care about leading zeros in the
        // lowest byte of `range` which is a `u32`.
        let shift = range.leading_zeros().saturating_sub(24);
        range <<= shift;
        self.state.bit_count -= shift as i32;
        debug_assert!(range >= 128);
        self.state.range = range - 1;

        BitResult::ok(retval)
    }

    #[cold]
    #[inline(never)]
    fn cold_read_bool(&mut self, probability: u8) -> BitResult<bool> {
        self.cold_read_bit(probability)
    }

    #[cold]
    #[inline(never)]
    fn cold_read_flag(&mut self) -> BitResult<bool> {
        self.cold_read_bit(128)
    }

    #[cold]
    #[inline(never)]
    fn cold_read_literal(&mut self, n: u8) -> BitResult<u8> {
        let mut v = 0u8;
        let mut res = self.start_accumulated_result();

        for _ in 0..n {
            let b = self.cold_read_flag().or_accumulate(&mut res);
            v = (v << 1) + u8::from(b);
        }

        self.keep_accumulating(res, v)
    }

    #[cold]
    #[inline(never)]
    fn cold_read_optional_signed_value(&mut self, n: u8) -> BitResult<i32> {
        let mut res = self.start_accumulated_result();
        let flag = self.cold_read_flag().or_accumulate(&mut res);
        if !flag {
            // We should not read further bits if the flag is not set.
            return self.keep_accumulating(res, 0);
        }
        let magnitude = self.cold_read_literal(n).or_accumulate(&mut res);
        let sign = self.cold_read_flag().or_accumulate(&mut res);

        let value = if sign {
            -i32::from(magnitude)
        } else {
            i32::from(magnitude)
        };
        self.keep_accumulating(res, value)
    }

    #[cold]
    #[inline(never)]
    fn cold_read_with_tree(&mut self, tree: &[TreeNode], start: usize) -> BitResult<i8> {
        let mut index = start;
        let mut res = self.start_accumulated_result();

        loop {
            let node = tree[index];
            let prob = node.prob;
            let b = self.cold_read_bit(prob).or_accumulate(&mut res);
            let t = if b { node.right } else { node.left };
            let new_index = usize::from(t);
            if new_index < tree.len() {
                index = new_index;
            } else {
                let value = TreeNode::value_from_branch(t);
                return self.keep_accumulating(res, value);
            }
        }
    }
}

impl FastDecoder<'_> {
    fn commit_if_valid<T>(self, value_if_not_past_eof: T) -> Option<T> {
        // If `chunk_index > self.chunks.len()`, it means we used zeroes
        // instead of an actual chunk and `value_if_not_past_eof` is nonsense.
        if self.uncommitted_state.chunk_index <= self.chunks.len() {
            *self.save_state = self.uncommitted_state;
            Some(value_if_not_past_eof)
        } else {
            None
        }
    }

    fn read_bool(mut self, probability: u8) -> Option<bool> {
        let bit = self.fast_read_bit(probability);
        self.commit_if_valid(bit)
    }

    fn read_flag(mut self) -> Option<bool> {
        let value = self.fast_read_flag();
        self.commit_if_valid(value)
    }

    fn read_literal(mut self, n: u8) -> Option<u8> {
        let value = self.fast_read_literal(n);
        self.commit_if_valid(value)
    }

    fn read_optional_signed_value(mut self, n: u8) -> Option<i32> {
        let flag = self.fast_read_flag();
        if !flag {
            // We should not read further bits if the flag is not set.
            return self.commit_if_valid(0);
        }
        let magnitude = self.fast_read_literal(n);
        let sign = self.fast_read_flag();
        let value = if sign {
            -i32::from(magnitude)
        } else {
            i32::from(magnitude)
        };
        self.commit_if_valid(value)
    }

    fn read_with_tree(mut self, tree: &[TreeNode], first_node: TreeNode) -> Option<i8> {
        let value = self.fast_read_with_tree(tree, first_node);
        self.commit_if_valid(value)
    }

    #[inline(always)]
    fn fast_read_bit(&mut self, probability: u8) -> bool {
        let State {
            mut chunk_index,
            mut value,
            range,
            mut bit_count,
        } = self.uncommitted_state;

        if bit_count < 0 {
            let chunk = self.chunks.get(chunk_index).copied();
            // We ignore invalid data inside the `fast_` functions,
            // but we increase `chunk_index` below, so we can check
            // whether we read invalid data in `commit_if_valid`.
            let chunk = chunk.unwrap_or_default();

            let v = u32::from_be_bytes(chunk);
            chunk_index += 1;
            value <<= 32;
            value |= u64::from(v);
            bit_count += 32;
        }
        debug_assert!(bit_count >= 0);

        // libwebp's `VP8GetBit`: `range` is the range minus one, `new_range` the range.
        debug_assert!((127..=254).contains(&range));
        let split = (range * u32::from(probability)) >> 8;
        let retval = (value >> bit_count) > u64::from(split);
        let new_range = if retval {
            value -= u64::from(split + 1) << bit_count;
            range - split
        } else {
            split + 1
        };

        // `new_range` fits a byte, so its leading zeros there are the shift back to 128..=254.
        debug_assert!((1..=254).contains(&new_range));
        let shift = (new_range as u8).leading_zeros();
        bit_count -= shift as i32;
        self.uncommitted_state = State {
            chunk_index,
            value,
            range: (new_range << shift) - 1,
            bit_count,
        };
        retval
    }

    fn fast_read_flag(&mut self) -> bool {
        let State {
            mut chunk_index,
            mut value,
            mut range,
            mut bit_count,
        } = self.uncommitted_state;

        if bit_count < 0 {
            let chunk = self.chunks.get(chunk_index).copied();
            // We ignore invalid data inside the `fast_` functions,
            // but we increase `chunk_index` below, so we can check
            // whether we read invalid data in `commit_if_valid`.
            let chunk = chunk.unwrap_or_default();

            let v = u32::from_be_bytes(chunk);
            chunk_index += 1;
            value <<= 32;
            value |= u64::from(v);
            bit_count += 32;
        }
        debug_assert!(bit_count >= 0);

        range += 1;
        debug_assert!((128..=255).contains(&range));
        let half_range = range / 2;
        let split = range - half_range;
        let bigsplit = u64::from(split) << bit_count;

        let retval = if let Some(new_value) = value.checked_sub(bigsplit) {
            range = half_range;
            value = new_value;
            true
        } else {
            range = split;
            false
        };

        // Compute shift required to satisfy `range >= 128`.
        // A `range` of 64..127 requires a shift of 1. No shift if `range` is 128.
        // Apply that shift to `range` and `self.bitcount`.
        debug_assert!((64..=128).contains(&range));
        let shift = if range == 0x80 { 0 } else { 1 };
        range <<= shift;
        bit_count -= shift;

        debug_assert!((128..=254).contains(&range));
        self.uncommitted_state = State {
            chunk_index,
            value,
            range: range - 1,
            bit_count,
        };
        retval
    }

    /// `fast_read_flag` for a sign: never the first bit of a stream, so the range is at
    /// most 254 and the renormalization always shifts by one.
    #[inline(always)]
    fn fast_read_sign(&mut self) -> bool {
        let State {
            mut chunk_index,
            mut value,
            range,
            mut bit_count,
        } = self.uncommitted_state;

        if bit_count < 0 {
            let chunk = self.chunks.get(chunk_index).copied().unwrap_or_default();
            chunk_index += 1;
            value <<= 32;
            value |= u64::from(u32::from_be_bytes(chunk));
            bit_count += 32;
        }

        // libwebp's `VP8GetSigned`, with `range` the range minus one.
        debug_assert!((127..=253).contains(&range));
        let split = range >> 1;
        let retval = (value >> bit_count) > u64::from(split);
        let new_range = if retval {
            value -= u64::from(split + 1) << bit_count;
            range - split
        } else {
            split + 1
        };
        self.uncommitted_state = State {
            chunk_index,
            value,
            range: (new_range << 1) - 1,
            bit_count: bit_count - 1,
        };
        retval
    }

    fn fast_read_literal(&mut self, n: u8) -> u8 {
        let mut v = 0u8;
        for _ in 0..n {
            let b = self.fast_read_flag();
            v = (v << 1) + u8::from(b);
        }
        v
    }

    fn fast_read_with_tree(&mut self, tree: &[TreeNode], mut node: TreeNode) -> i8 {
        loop {
            let prob = node.prob;
            let b = self.fast_read_bit(prob);
            let i = if b { node.right } else { node.left };
            let Some(next_node) = tree.get(usize::from(i)) else {
                return TreeNode::value_from_branch(i);
            };
            node = *next_node;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_decoder_hello_short() {
        let mut decoder = ArithmeticDecoder::new();
        let data = b"hel";
        let size = data.len();
        let mut buf = vec![[0u8; 4]; 1];
        buf.as_mut_slice().as_flattened_mut()[..size].copy_from_slice(&data[..]);
        decoder.init(buf, size).unwrap();
        let mut res = decoder.start_accumulated_result();
        assert_eq!(false, decoder.read_flag().or_accumulate(&mut res));
        assert_eq!(true, decoder.read_bool(10).or_accumulate(&mut res));
        assert_eq!(false, decoder.read_bool(250).or_accumulate(&mut res));
        assert_eq!(1, decoder.read_literal(1).or_accumulate(&mut res));
        assert_eq!(5, decoder.read_literal(3).or_accumulate(&mut res));
        assert_eq!(64, decoder.read_literal(8).or_accumulate(&mut res));
        assert_eq!(185, decoder.read_literal(8).or_accumulate(&mut res));
        decoder.check(res, ()).unwrap();
    }

    #[test]
    fn test_arithmetic_decoder_hello_long() {
        let mut decoder = ArithmeticDecoder::new();
        let data = b"hello world";
        let size = data.len();
        let mut buf = vec![[0u8; 4]; (size + 3) / 4];
        buf.as_mut_slice().as_flattened_mut()[..size].copy_from_slice(&data[..]);
        decoder.init(buf, size).unwrap();
        let mut res = decoder.start_accumulated_result();
        assert_eq!(false, decoder.read_flag().or_accumulate(&mut res));
        assert_eq!(true, decoder.read_bool(10).or_accumulate(&mut res));
        assert_eq!(false, decoder.read_bool(250).or_accumulate(&mut res));
        assert_eq!(1, decoder.read_literal(1).or_accumulate(&mut res));
        assert_eq!(5, decoder.read_literal(3).or_accumulate(&mut res));
        assert_eq!(64, decoder.read_literal(8).or_accumulate(&mut res));
        assert_eq!(185, decoder.read_literal(8).or_accumulate(&mut res));
        assert_eq!(31, decoder.read_literal(8).or_accumulate(&mut res));
        decoder.check(res, ()).unwrap();
    }

    #[test]
    fn coeffs_keep_the_y2_dc_near_the_end() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..500 {
            let len: usize = rng.gen_range(1..40);
            let mut buf = vec![[0u8; 4]; len.div_ceil(4)];
            for b in &mut buf.as_mut_slice().as_flattened_mut()[..len] {
                *b = rng.gen();
            }
            let mut probs: PlaneProbs = [[[0; 11]; 3]; 8];
            for p in probs.as_flattened_mut().as_flattened_mut() {
                *p = rng.gen_range(1..=255);
            }
            let (mut fast, mut cold) = (ArithmeticDecoder::new(), ArithmeticDecoder::new());
            fast.init(buf.clone(), len).unwrap();
            cold.init(buf, len).unwrap();
            for _ in 0..len {
                let (mut a, mut b) = ([0; 16], [0; 16]);
                (a[0], b[0]) = (775, 775);
                let na = fast.read_coeffs(&probs, 0, 1, [3, 5], &mut a);
                let nb = cold.cold_read_coeffs(&probs, 0, 1, [3, 5], &mut b);
                if na.is_err() || nb.is_err() {
                    assert_eq!(na.is_err(), nb.is_err());
                    break;
                }
                assert_eq!((na.unwrap(), a), (nb.unwrap(), b));
            }
        }
    }

    #[test]
    fn test_arithmetic_decoder_uninit() {
        let mut decoder = ArithmeticDecoder::new();
        let mut res = decoder.start_accumulated_result();
        let _ = decoder.read_flag().or_accumulate(&mut res);
        let result = decoder.check(res, ());
        assert!(result.is_err());
    }
}
