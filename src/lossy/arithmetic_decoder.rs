// The coefficient reading follows libwebp's GetCoeffsFast and VP8GetBit: Copyright Google Inc.,
// BSD 3-Clause, see LICENSE-LIBWEBP.

use std::io::SeekFrom;

use crate::decoder::DecodingError;
use crate::ReadSeek;

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
        self.decoder.cold_read_sign().or_accumulate(self.res)
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

/// The most bytes a partition holds at once.
const WINDOW: usize = 64 * 1024;
/// Bytes a macroblock may need: 25 blocks of at most 304 bits read, each of at most 7 bits of
/// data, are 6650 bytes, and the reader looks 8 bytes ahead.
const MARGIN: usize = 16 * 1024;

/// A partition's bits from a window of the partition that [`top_up`](Self::top_up) refills
/// from the data, read as libwebp's 64-bit reader reads them: seven bytes at a time while
/// eight are left, then one at a time. A corrupt stream can carry bits past the top of
/// `value`, which the load sizes then decide.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct ArithmeticDecoder {
    /// The window's `len` bytes and [`PAD`] zeros, which loads past the end read.
    bytes: Vec<u8>,
    len: usize,
    state: State,
    /// Whether a read went past the end of the data.
    eof: bool,
    /// Where the bytes not yet in the window start, and where the partition ends.
    next: u64,
    end: u64,
}

const PAD: usize = 8;

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy)]
struct State {
    /// Where the bytes not yet in `value` start.
    pos: usize,
    value: u64,
    /// The range minus one, as libwebp keeps it.
    range: u32,
    bit_count: i32,
}

impl State {
    /// `VP8GetBit` once `bit_count` is not negative. libwebp compares the low 32 bits of the
    /// value at the bit position, which a corrupt stream carries bits above.
    #[inline(always)]
    fn bit(&mut self, probability: u8) -> bool {
        let split = (self.range * u32::from(probability)) >> 8;
        let bit = (self.value >> self.bit_count) as u32 > split;
        let range = if bit {
            self.value -= u64::from(split + 1) << self.bit_count;
            self.range - split
        } else {
            split + 1
        };
        // `range` fits a byte, so its leading zeros there are the shift back to 128..=254.
        let shift = (range as u8).leading_zeros();
        self.bit_count -= shift as i32;
        self.range = (range << shift) - 1;
        bit
    }

    /// `VP8GetSigned` once `bit_count` is not negative: a bit of probability one half whose
    /// renormalization is always one bit.
    #[inline(always)]
    fn sign(&mut self) -> bool {
        let split = self.range >> 1;
        let value = (self.value >> self.bit_count) as u32;
        let mask = ((split.wrapping_sub(value) as i32) >> 31) as u32;
        self.value -= u64::from((split + 1) & mask) << self.bit_count;
        self.range = self.range.wrapping_add(mask) | 1;
        self.bit_count -= 1;
        mask != 0
    }
}

#[cfg_attr(test, derive(Debug))]
struct FastDecoder<'a> {
    bytes: &'a [u8],
    len: usize,
    uncommitted_state: State,
    save_state: &'a mut State,
}

impl ArithmeticDecoder {
    const START: State = State {
        pos: 0,
        value: 0,
        range: 254,
        bit_count: -8,
    };

    pub(crate) fn new() -> Self {
        ArithmeticDecoder {
            bytes: vec![0; PAD],
            len: 0,
            state: Self::START,
            eof: true,
            next: 0,
            end: 0,
        }
    }

    /// Reads `data`.
    pub(crate) fn init(&mut self, data: &[u8]) {
        let mut bytes = Vec::with_capacity(data.len() + PAD);
        bytes.extend_from_slice(data);
        bytes.resize(data.len() + PAD, 0);
        *self = Self {
            bytes,
            len: data.len(),
            state: Self::START,
            eof: false,
            next: 0,
            end: 0,
        };
    }

    /// Reads the bytes `start..end` of the data a window at a time, refilled by
    /// [`top_up`](Self::top_up).
    pub(crate) fn init_window(&mut self, r: &mut dyn ReadSeek, start: u64, end: u64) -> Result<(), DecodingError> {
        *self = Self {
            bytes: Vec::with_capacity(WINDOW.min((end - start) as usize) + PAD),
            len: 0,
            state: Self::START,
            eof: false,
            next: start,
            end,
        };
        self.top_up(r)
    }

    /// Refills the window from `r` unless it holds what a macroblock may read or the rest of
    /// the partition.
    pub(crate) fn top_up(&mut self, r: &mut dyn ReadSeek) -> Result<(), DecodingError> {
        let held = self.len - self.state.pos;
        if held >= MARGIN || self.next == self.end {
            return Ok(());
        }
        self.bytes.truncate(self.len);
        self.bytes.drain(..self.state.pos);
        self.state.pos = 0;
        let left = self.end - self.next;
        let last = left <= (WINDOW - held) as u64;
        let read = if last { left as usize } else { (WINDOW - held) & !3 };
        self.bytes.resize(held + read, 0);
        r.seek(SeekFrom::Start(self.next))?;
        r.read_exact(&mut self.bytes[held..])?;
        self.len = held + read;
        self.bytes.resize(self.len + PAD, 0);
        self.next += read as u64;
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
    // of bits are somewhere other than in the last seven bytes. Therefore we
    // can pretend the buffer has infinite size. After we are done reading,
    // we check if we actually read past the end of `self.bytes`.
    // If so, we backtrack (or rather we discard `uncommitted_state`)
    // and try again with the slow approach. This might result in doing double
    // work for those last few bytes -- in fact we even keep retrying the fast
    // method to save an if-statement --, but more than make up for that by
    // speeding up reading from the other thousands or millions of bytes.
    fn fast(&mut self) -> FastDecoder<'_> {
        FastDecoder {
            bytes: &self.bytes,
            len: self.len,
            uncommitted_state: self.state,
            save_state: &mut self.state,
        }
    }

    fn is_past_eof(&self) -> bool {
        self.eof
    }

    fn cold_read_bit(&mut self, probability: u8) -> BitResult<bool> {
        if !self.cold_load() {
            return BitResult::err();
        }
        BitResult::ok(self.state.bit(probability))
    }

    fn cold_read_sign(&mut self) -> BitResult<bool> {
        if !self.cold_load() {
            return BitResult::err();
        }
        BitResult::ok(self.state.sign())
    }

    /// `VP8LoadNewBytes` where `bit_count` is negative, false past the end of the data.
    fn cold_load(&mut self) -> bool {
        if self.state.bit_count < 0 {
            let pos = self.state.pos;
            if pos + 8 <= self.len {
                let w = u64::from_be_bytes(self.bytes[pos..pos + 8].try_into().unwrap());
                self.state.pos += 7;
                self.state.value = self.state.value << 56 | w >> 8;
                self.state.bit_count += 56;
            } else if pos < self.len {
                self.state.pos += 1;
                self.state.value = self.state.value << 8 | u64::from(self.bytes[pos]);
                self.state.bit_count += 8;
            } else {
                self.eof = true;
                return false;
            }
        }
        true
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
        // libwebp loads seven bytes while eight are left and then one at a time: a load
        // short of that read the padding, and `value_if_not_past_eof` is nonsense.
        let pos = self.uncommitted_state.pos;
        if pos < self.len || pos == self.save_state.pos {
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

    /// Seven more bytes into `value`, as `VP8LoadNewBytes` with eight bytes left; with fewer
    /// it reads the padding, and `commit_if_valid` turns down what was read then.
    #[inline(always)]
    fn load(&mut self) {
        let st = &mut self.uncommitted_state;
        let at = st.pos.min(self.len);
        let w = u64::from_be_bytes(self.bytes[at..at + 8].try_into().unwrap());
        st.pos += 7;
        st.value = st.value << 56 | w >> 8;
        st.bit_count += 56;
    }

    #[inline(always)]
    fn fast_read_bit(&mut self, probability: u8) -> bool {
        if self.uncommitted_state.bit_count < 0 {
            self.load();
        }
        self.uncommitted_state.bit(probability)
    }

    fn fast_read_flag(&mut self) -> bool {
        self.fast_read_bit(128)
    }

    #[inline(always)]
    fn fast_read_sign(&mut self) -> bool {
        if self.uncommitted_state.bit_count < 0 {
            self.load();
        }
        self.uncommitted_state.sign()
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
        decoder.init(b"hel");
        let mut res = decoder.start_accumulated_result();
        assert_eq!(false, decoder.read_flag().or_accumulate(&mut res));
        assert_eq!(true, decoder.read_bool(10).or_accumulate(&mut res));
        assert_eq!(false, decoder.read_bool(250).or_accumulate(&mut res));
        assert_eq!(1, decoder.read_literal(1).or_accumulate(&mut res));
        assert_eq!(5, decoder.read_literal(3).or_accumulate(&mut res));
        assert_eq!(64, decoder.read_literal(8).or_accumulate(&mut res));
        // The last literal needs a byte past the data, which libwebp counts as an error.
        let _ = decoder.read_literal(8).or_accumulate(&mut res);
        assert!(decoder.check(res, ()).is_err());
    }

    #[test]
    fn test_arithmetic_decoder_hello_long() {
        let mut decoder = ArithmeticDecoder::new();
        decoder.init(b"hello world");
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
            let buf: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
            let mut probs: PlaneProbs = [[[0; 11]; 3]; 8];
            for p in probs.as_flattened_mut().as_flattened_mut() {
                *p = rng.gen_range(1..=255);
            }
            let (mut fast, mut cold) = (ArithmeticDecoder::new(), ArithmeticDecoder::new());
            fast.init(&buf);
            cold.init(&buf);
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
