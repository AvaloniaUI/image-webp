//! Resampling rows to another size with the fixed-point arithmetic of libwebp's rescaler
//! (`utils/rescaler_utils.c`, `dsp/rescaler.c`), so a scaled decode gives libwebp's pixels:
//! box averaging where a side shrinks, bilinear interpolation where it grows.

const FIX: u32 = 32;
const ONE: u64 = 1 << FIX;
const ROUNDER: u64 = ONE >> 1;

fn frac(x: u64, y: u64) -> u32 {
    ((x << FIX) / y) as u32
}

fn mult_fix(x: u32, y: u32) -> u32 {
    ((u64::from(x) * u64::from(y) + ROUNDER) >> FIX) as u32
}

fn mult_fix_floor(x: u32, y: u32) -> u32 {
    ((u64::from(x) * u64::from(y)) >> FIX) as u32
}

fn clip(v: u32) -> u8 {
    v.min(255) as u8
}

/// Rows of `channels` interleaved 8-bit samples, `src_width`×`src_height` in,
/// `dst_width`×`dst_height` out, imported and exported a row at a time.
pub(crate) struct Rescaler {
    x_expand: bool,
    y_expand: bool,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
    /// Source pixels per output pixel where a shrinking side divides evenly.
    whole_shrink: Option<usize>,
    /// Per output pixel of any other shrink, its first source pixel (then the source width) and
    /// the weight its last one carries into the next output. Most take `run_count` source pixels.
    starts: Vec<u32>,
    weights: Vec<u32>,
    run_count: usize,
    x_add: i32,
    x_sub: i32,
    fx_scale: u32,
    y_add: i32,
    y_sub: i32,
    y_accum: i32,
    fy_scale: u32,
    fxy_scale: u32,
    src_y: usize,
    dst_y: usize,
    irow: Vec<u32>,
    frow: Vec<u32>,
}

impl Rescaler {
    pub(crate) fn new(
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
        channels: usize,
    ) -> Self {
        let x_expand = src_width < dst_width;
        let y_expand = src_height < dst_height;
        let (x_add, x_sub) = match x_expand {
            true => (dst_width as i32 - 1, src_width as i32 - 1),
            false => (src_width as i32, dst_width as i32),
        };
        let (y_add, y_sub) = match y_expand {
            true => (src_height as i32 - 1, dst_height as i32 - 1),
            false => (src_height as i32, dst_height as i32),
        };
        let fx_scale = if x_expand { 0 } else { frac(1, x_sub as u64) };
        let (fy_scale, fxy_scale) = match y_expand {
            true => (frac(1, x_add as u64), 0),
            false => {
                let ratio = dst_height as u64 * ONE / (x_add as u64 * y_add as u64);
                // A ratio of one does not fit: exported as is.
                let fxy = u32::try_from(ratio).unwrap_or(0);
                (frac(1, y_sub as u64), fxy)
            }
        };
        let whole_shrink = (!x_expand && x_add % x_sub == 0).then(|| (x_add / x_sub) as usize);
        let (mut starts, mut weights) = (Vec::new(), Vec::new());
        if !x_expand && whole_shrink.is_none() {
            let (mut accum, mut x) = (0, 0);
            for _ in 0..dst_width {
                starts.push(x);
                accum += x_add;
                while accum > 0 {
                    accum -= x_sub;
                    x += 1;
                }
                weights.push((-accum) as u32);
            }
            starts.push(x);
        }
        let count = |n: usize| {
            starts
                .windows(2)
                .filter(|s| (s[1] - s[0]) as usize == n)
                .count()
        };
        let lo = (x_add / x_sub.max(1)) as usize;
        let run_count = if count(lo + 1) > count(lo) {
            lo + 1
        } else {
            lo
        };
        Self {
            x_expand,
            y_expand,
            src_width,
            src_height,
            dst_width,
            dst_height,
            channels,
            whole_shrink,
            starts,
            weights,
            run_count,
            x_add,
            x_sub,
            fx_scale,
            y_add,
            y_sub,
            y_accum: if y_expand { y_sub } else { y_add },
            fy_scale,
            fxy_scale,
            src_y: 0,
            dst_y: 0,
            irow: vec![0; dst_width * channels],
            frow: vec![0; dst_width * channels],
        }
    }

    /// Bytes in an output row.
    pub(crate) fn dst_row_len(&self) -> usize {
        self.dst_width * self.channels
    }

    pub(crate) fn input_done(&self) -> bool {
        self.src_y >= self.src_height
    }

    /// Whether a row is ready to export; no row is imported until it is.
    pub(crate) fn has_pending_output(&self) -> bool {
        self.dst_y < self.dst_height && self.y_accum <= 0
    }

    /// Imports one row unless an output row is pending; returns whether it did.
    pub(crate) fn import(&mut self, src: &[u8]) -> bool {
        if self.has_pending_output() || self.input_done() {
            return false;
        }
        if self.y_expand {
            std::mem::swap(&mut self.irow, &mut self.frow);
        }
        if let Some(k) = self.whole_shrink {
            let sum = (!self.y_expand).then_some(&mut self.irow[..]);
            shrink_whole(
                src,
                self.channels,
                k,
                self.x_sub as u32,
                &mut self.frow,
                sum,
            );
        } else if self.x_expand {
            self.import_expand(src);
        } else {
            self.import_shrink(src);
        }
        if !self.y_expand && self.whole_shrink.is_none() {
            for (i, f) in self.irow.iter_mut().zip(&self.frow) {
                *i = i.wrapping_add(*f);
            }
        }
        self.src_y += 1;
        self.y_accum -= self.y_sub;
        true
    }

    fn import_expand(&mut self, src: &[u8]) {
        match self.channels {
            1 => self.import_expand_n::<1>(src),
            2 => self.import_expand_n::<2>(src),
            3 => self.import_expand_n::<3>(src),
            4 => self.import_expand_n::<4>(src),
            n => unreachable!("{n} channels"),
        }
    }

    fn import_shrink(&mut self, src: &[u8]) {
        match self.channels {
            1 => self.import_shrink_n::<1>(src),
            2 => self.import_shrink_n::<2>(src),
            3 => self.import_shrink_n::<3>(src),
            4 => self.import_shrink_n::<4>(src),
            n => unreachable!("{n} channels"),
        }
    }

    fn import_expand_n<const N: usize>(&mut self, src: &[u8]) {
        let px = |x: usize| -> [u32; N] { std::array::from_fn(|c| u32::from(src[x * N + c])) };
        let (x_add, x_sub) = (self.x_add, self.x_sub);
        let mut x_in = 1;
        let mut accum = x_add;
        let mut left = px(0);
        let mut right = if self.src_width > 1 { px(1) } else { left };
        for (x_out, out) in self.frow.chunks_exact_mut(N).enumerate() {
            for c in 0..N {
                out[c] = right[c]
                    .wrapping_mul(x_add as u32)
                    .wrapping_add(left[c].wrapping_sub(right[c]).wrapping_mul(accum as u32));
            }
            accum -= x_sub;
            if accum < 0 && x_out + 1 < self.dst_width {
                left = right;
                x_in += 1;
                right = px(x_in);
                accum += x_add;
            }
        }
    }

    fn import_shrink_n<const N: usize>(&mut self, src: &[u8]) {
        let src = &src[..self.src_width * N];
        let (x_sub, fx_scale) = (self.x_sub as u32, self.fx_scale);
        #[cfg(target_arch = "x86_64")]
        if sse2::shrink_spans(
            src,
            N,
            &self.starts,
            &self.weights,
            self.run_count,
            x_sub,
            fx_scale,
            &mut self.frow,
        ) {
            return;
        }
        for x in 0..self.dst_width {
            shrink_span::<N>(
                src,
                &self.starts,
                &self.weights,
                x_sub,
                fx_scale,
                x,
                &mut self.frow,
            );
        }
    }

    /// Exports the pending row into `dst`.
    pub(crate) fn export(&mut self, dst: &mut [u8]) {
        debug_assert!(self.has_pending_output());
        let dst = &mut dst[..self.dst_row_len()];
        if self.y_expand {
            if self.y_accum == 0 {
                for (d, &f) in dst.iter_mut().zip(&self.frow) {
                    *d = clip(mult_fix(f, self.fy_scale));
                }
            } else {
                let b = frac((-self.y_accum) as u64, self.y_sub as u64);
                let a = (ONE - u64::from(b)) as u32;
                for ((d, &f), &i) in dst.iter_mut().zip(&self.frow).zip(&self.irow) {
                    let v = u64::from(a) * u64::from(f) + u64::from(b) * u64::from(i);
                    let j = ((v + ROUNDER) >> FIX) as u32;
                    *d = clip(mult_fix(j, self.fy_scale));
                }
            }
        } else if self.fxy_scale != 0 {
            let yscale = self.fy_scale.wrapping_mul((-self.y_accum) as u32);
            #[cfg(target_arch = "x86_64")]
            let done = sse2::export_shrink(dst, &mut self.irow, &self.frow, yscale, self.fxy_scale);
            #[cfg(not(target_arch = "x86_64"))]
            let done = 0;
            export_shrink(
                &mut dst[done..],
                &mut self.irow[done..],
                &self.frow[done..],
                yscale,
                self.fxy_scale,
            );
        } else {
            for (d, i) in dst.iter_mut().zip(self.irow.iter_mut()) {
                *d = *i as u8;
                *i = 0;
            }
        }
        self.y_accum += self.y_add;
        self.dst_y += 1;
    }
}

fn export_shrink(dst: &mut [u8], irow: &mut [u32], frow: &[u32], yscale: u32, fxy_scale: u32) {
    for ((d, i), &f) in dst.iter_mut().zip(irow.iter_mut()).zip(frow) {
        if yscale != 0 {
            let frac = mult_fix_floor(f, yscale);
            *d = clip(mult_fix(i.wrapping_sub(frac), fxy_scale));
            *i = frac;
        } else {
            *d = clip(mult_fix(*i, fxy_scale));
            *i = 0;
        }
    }
}

/// A shrink by a whole `k` source pixels per output pixel, which leaves libwebp's arithmetic no
/// fraction to carry between outputs; each output is added into `sum` as well where given.
fn shrink_whole(
    src: &[u8],
    channels: usize,
    k: usize,
    x_sub: u32,
    out: &mut [u32],
    sum: Option<&mut [u32]>,
) {
    match channels {
        1 => shrink_whole_n::<1>(src, k, x_sub, out, sum),
        2 => shrink_whole_n::<2>(src, k, x_sub, out, sum),
        3 => shrink_whole_n::<3>(src, k, x_sub, out, sum),
        4 => shrink_whole_n::<4>(src, k, x_sub, out, sum),
        n => unreachable!("{n} channels"),
    }
}

fn shrink_whole_n<const N: usize>(
    src: &[u8],
    k: usize,
    x_sub: u32,
    out: &mut [u32],
    mut sum: Option<&mut [u32]>,
) {
    #[cfg(target_arch = "x86_64")]
    let done = sse2::shrink_whole(src, N, k, x_sub, out, sum.as_deref_mut());
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    let pixels = src
        .chunks_exact(N * k)
        .zip(out.as_chunks_mut::<N>().0)
        .enumerate();
    for (x, (px, o)) in pixels.skip(done / N) {
        for (c, o) in o.iter_mut().enumerate() {
            *o = px
                .iter()
                .skip(c)
                .step_by(N)
                .map(|&p| u32::from(p))
                .sum::<u32>()
                * x_sub;
        }
        if let Some(sum) = sum.as_deref_mut() {
            for (s, &o) in sum[x * N..][..N].iter_mut().zip(&*o) {
                *s = s.wrapping_add(o);
            }
        }
    }
}

/// Output pixel `x` of a shrink that carries a fraction of its last source pixel into the next.
fn shrink_span<const N: usize>(
    src: &[u8],
    starts: &[u32],
    weights: &[u32],
    x_sub: u32,
    fx_scale: u32,
    x: usize,
    out: &mut [u32],
) {
    let (a, b) = (starts[x] as usize, starts[x + 1] as usize);
    for c in 0..N {
        let carry = match x {
            0 => 0,
            _ => mult_fix(
                u32::from(src[(a - 1) * N + c]).wrapping_mul(weights[x - 1]),
                fx_scale,
            ),
        };
        let pixels = src[a * N + c..b * N].iter().step_by(N);
        let sum = pixels.fold(carry, |s, &p| s.wrapping_add(u32::from(p)));
        let frac = u32::from(src[(b - 1) * N + c]).wrapping_mul(weights[x]);
        out[x * N + c] = sum.wrapping_mul(x_sub).wrapping_sub(frac);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod sse2 {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn emit(out: &mut [u32], sum: &mut Option<&mut [u32]>, i: usize, v: __m128i) {
        _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), v);
        if let Some(sum) = sum {
            let p = sum.as_mut_ptr().add(i).cast();
            _mm_storeu_si128(p, _mm_add_epi32(_mm_loadu_si128(p), v));
        }
    }

    /// [`super::shrink_whole`] for the outputs whole vectors cover; returns how many it wrote.
    /// The products go through `madd`, so `x_sub` must fit an i16.
    pub(super) fn shrink_whole(
        src: &[u8],
        channels: usize,
        k: usize,
        x_sub: u32,
        out: &mut [u32],
        mut sum: Option<&mut [u32]>,
    ) -> usize {
        let n = out.len();
        assert!(src.len() >= n * k && sum.as_ref().is_none_or(|s| s.len() >= n));
        if x_sub > i16::MAX as u32 {
            return 0;
        }
        let s = src.as_ptr();
        let mut i = 0;
        // SAFETY: each load ends within the `k` source samples of the outputs it writes, each
        // store within the `n` outputs.
        unsafe {
            let z = _mm_setzero_si128();
            let xs = _mm_set1_epi32(x_sub as i32);
            match (channels, k) {
                (1, 1) => {
                    while i + 16 <= n {
                        let v = _mm_loadu_si128(s.add(i).cast());
                        let (lo, hi) = (_mm_unpacklo_epi8(v, z), _mm_unpackhi_epi8(v, z));
                        emit(
                            out,
                            &mut sum,
                            i,
                            _mm_madd_epi16(_mm_unpacklo_epi16(lo, z), xs),
                        );
                        emit(
                            out,
                            &mut sum,
                            i + 4,
                            _mm_madd_epi16(_mm_unpackhi_epi16(lo, z), xs),
                        );
                        emit(
                            out,
                            &mut sum,
                            i + 8,
                            _mm_madd_epi16(_mm_unpacklo_epi16(hi, z), xs),
                        );
                        emit(
                            out,
                            &mut sum,
                            i + 12,
                            _mm_madd_epi16(_mm_unpackhi_epi16(hi, z), xs),
                        );
                        i += 16;
                    }
                }
                (1, 2) => {
                    let xs = _mm_set1_epi16(x_sub as i16);
                    while i + 8 <= n {
                        let v = _mm_loadu_si128(s.add(2 * i).cast());
                        emit(
                            out,
                            &mut sum,
                            i,
                            _mm_madd_epi16(_mm_unpacklo_epi8(v, z), xs),
                        );
                        emit(
                            out,
                            &mut sum,
                            i + 4,
                            _mm_madd_epi16(_mm_unpackhi_epi8(v, z), xs),
                        );
                        i += 8;
                    }
                }
                (1, 4) => {
                    let (xs, low) = (_mm_set1_epi16(x_sub as i16), _mm_set1_epi16(0xff));
                    while i + 4 <= n {
                        let v = _mm_loadu_si128(s.add(4 * i).cast());
                        let pairs = _mm_add_epi16(_mm_and_si128(v, low), _mm_srli_epi16::<8>(v));
                        emit(out, &mut sum, i, _mm_madd_epi16(pairs, xs));
                        i += 4;
                    }
                }
                (1, 8) => {
                    while i + 4 <= n {
                        let a = _mm_sad_epu8(_mm_loadu_si128(s.add(8 * i).cast()), z);
                        let b = _mm_sad_epu8(_mm_loadu_si128(s.add(8 * i + 16).cast()), z);
                        let v = _mm_unpacklo_epi64(
                            _mm_shuffle_epi32::<0b1000>(a),
                            _mm_shuffle_epi32::<0b1000>(b),
                        );
                        emit(out, &mut sum, i, _mm_madd_epi16(v, xs));
                        i += 4;
                    }
                }
                (4, 2..=128) => {
                    while i + 4 <= n {
                        let p = s.add(i * k);
                        let mut acc = z;
                        for j in (0..k - 1).step_by(2) {
                            let v = _mm_unpacklo_epi8(_mm_loadl_epi64(p.add(4 * j).cast()), z);
                            acc = _mm_add_epi16(acc, _mm_add_epi16(v, _mm_srli_si128::<8>(v)));
                        }
                        if k % 2 == 1 {
                            let v = _mm_cvtsi32_si128(
                                p.add(4 * (k - 1)).cast::<i32>().read_unaligned(),
                            );
                            acc = _mm_add_epi16(acc, _mm_unpacklo_epi8(v, z));
                        }
                        emit(
                            out,
                            &mut sum,
                            i,
                            _mm_madd_epi16(_mm_unpacklo_epi16(acc, z), xs),
                        );
                        i += 4;
                    }
                }
                _ => {}
            }
        }
        i
    }

    /// [`super::shrink_span`] for every output pixel of one channel, four outputs to a vector where
    /// four in a row take `k` source pixels each. False for more channels.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn shrink_spans(
        src: &[u8],
        channels: usize,
        starts: &[u32],
        weights: &[u32],
        k: usize,
        x_sub: u32,
        fx_scale: u32,
        out: &mut [u32],
    ) -> bool {
        let n = weights.len();
        assert!(starts.len() == n + 1 && out.len() >= n * channels);
        if x_sub > i16::MAX as u32 || channels != 1 {
            return false;
        }
        let s = src.as_ptr();
        // SAFETY: each load is checked against `src` through `starts` before it is made, each
        // store ends within the first `n` outputs.
        unsafe {
            let z = _mm_setzero_si128();
            let (xs, ones) = (_mm_set1_epi32(x_sub as i32), _mm_set1_epi16(1));
            let (fx, round) = (
                _mm_set1_epi64x(i64::from(fx_scale)),
                _mm_set1_epi64x(1 << 31),
            );
            let px = |i: usize| _mm_cvtsi32_si128(s.add(i).cast::<i32>().read_unaligned());
            let mut x = 0;
            while x < n {
                let a = starts[x] as usize;
                let whole = x >= 1 && x + 4 <= n && a >= 1 && starts[x + 4] as usize == a + 4 * k;
                if !(whole && matches!(k, 1 | 2 | 4 | 8) && a + 4 * k <= src.len()) {
                    super::shrink_span::<1>(src, starts, weights, x_sub, fx_scale, x, out);
                    x += 1;
                    continue;
                }
                let p = s.add(a);
                let (sums, last) = match k {
                    1 => {
                        let v = _mm_unpacklo_epi16(_mm_unpacklo_epi8(px(a), z), z);
                        (v, v)
                    }
                    2 => {
                        let v = _mm_unpacklo_epi8(_mm_loadl_epi64(p.cast()), z);
                        (_mm_madd_epi16(v, ones), _mm_srli_epi32::<16>(v))
                    }
                    4 => {
                        let v = _mm_loadu_si128(p.cast());
                        let pairs = _mm_add_epi16(
                            _mm_and_si128(v, _mm_set1_epi16(0xff)),
                            _mm_srli_epi16::<8>(v),
                        );
                        (_mm_madd_epi16(pairs, ones), _mm_srli_epi32::<24>(v))
                    }
                    _ => {
                        let (u, v) = (_mm_loadu_si128(p.cast()), _mm_loadu_si128(p.add(16).cast()));
                        let (su, sv) = (_mm_sad_epu8(u, z), _mm_sad_epu8(v, z));
                        let (lu, lv) = (_mm_srli_epi64::<56>(u), _mm_srli_epi64::<56>(v));
                        let pick = |a, b| {
                            _mm_unpacklo_epi64(
                                _mm_shuffle_epi32::<0b1000>(a),
                                _mm_shuffle_epi32::<0b1000>(b),
                            )
                        };
                        (pick(su, sv), pick(lu, lv))
                    }
                };
                let prev = _mm_or_si128(
                    _mm_slli_si128::<4>(last),
                    _mm_cvtsi32_si128(i32::from(*p.sub(1))),
                );
                let w = _mm_loadu_si128(weights.as_ptr().add(x).cast());
                let wp = _mm_loadu_si128(weights.as_ptr().add(x - 1).cast());
                let carry = mult_fix(_mm_madd_epi16(prev, wp), fx, round);
                let v = _mm_sub_epi32(
                    _mm_madd_epi16(_mm_add_epi32(sums, carry), xs),
                    _mm_madd_epi16(last, w),
                );
                _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), v);
                x += 4;
            }
        }
        true
    }

    #[inline(always)]
    unsafe fn mult_fix(v: __m128i, scale: __m128i, round: __m128i) -> __m128i {
        let even = _mm_srli_epi64::<32>(_mm_add_epi64(_mm_mul_epu32(v, scale), round));
        let odd = _mm_add_epi64(_mm_mul_epu32(_mm_srli_epi64::<32>(v), scale), round);
        _mm_or_si128(even, _mm_and_si128(odd, _mm_set1_epi64x(-1 << 32)))
    }

    #[inline(always)]
    unsafe fn clip(v: __m128i) -> __m128i {
        let small = _mm_cmpeq_epi32(_mm_srli_epi32::<8>(v), _mm_setzero_si128());
        _mm_or_si128(
            _mm_and_si128(v, small),
            _mm_andnot_si128(small, _mm_set1_epi32(255)),
        )
    }

    /// The shrinking export for the samples whole vectors cover; returns how many it wrote.
    pub(super) fn export_shrink(
        dst: &mut [u8],
        irow: &mut [u32],
        frow: &[u32],
        yscale: u32,
        fxy_scale: u32,
    ) -> usize {
        let n = dst.len();
        assert!(irow.len() >= n && frow.len() >= n);
        let mut i = 0;
        // SAFETY: every load and store ends within the first `n` samples of the rows.
        unsafe {
            let (round, floor) = (_mm_set1_epi64x(1 << 31), _mm_setzero_si128());
            let (ys, fxy) = (
                _mm_set1_epi64x(i64::from(yscale)),
                _mm_set1_epi64x(i64::from(fxy_scale)),
            );
            while i + 16 <= n {
                let mut d = [_mm_setzero_si128(); 4];
                for (g, d) in d.iter_mut().enumerate() {
                    let ip = irow.as_mut_ptr().add(i + 4 * g).cast::<__m128i>();
                    let v = _mm_loadu_si128(ip);
                    let frac = match yscale {
                        0 => floor,
                        _ => mult_fix(
                            _mm_loadu_si128(frow.as_ptr().add(i + 4 * g).cast()),
                            ys,
                            floor,
                        ),
                    };
                    _mm_storeu_si128(ip, frac);
                    *d = clip(mult_fix(_mm_sub_epi32(v, frac), fxy, round));
                }
                let bytes =
                    _mm_packus_epi16(_mm_packs_epi32(d[0], d[1]), _mm_packs_epi32(d[2], d[3]));
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), bytes);
                i += 16;
            }
        }
        i
    }
}

/// libwebp's `WebPRescalerGetScaledDimensions` leaves given sizes as they are; the decoders
/// turn off the loop filter where both sides shrink below three quarters.
pub(crate) fn bypasses_filter(width: u32, height: u32, scaled: (u32, u32)) -> bool {
    scaled.0 < width * 3 / 4 && scaled.1 < height * 3 / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, ch: usize) -> Vec<u8> {
        let mut r = Rescaler::new(sw, sh, dw, dh, ch);
        let mut out = vec![0; dw * dh * ch];
        let mut y = 0;
        let mut rows = src.chunks(sw * ch);
        while y < dh {
            if r.has_pending_output() {
                r.export(&mut out[y * dw * ch..]);
                y += 1;
            } else {
                assert!(r.import(rows.next().unwrap()));
            }
        }
        out
    }

    #[test]
    fn identity_and_halving() {
        let src: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        assert_eq!(run(&src, 8, 8, 8, 8, 1), src);
        let half = run(&src, 8, 8, 4, 4, 1);
        assert_eq!(half[0], ((0 + 4 + 32 + 36) as f32 / 4.0).round() as u8);
        let up = run(&[0, 255], 2, 1, 4, 1, 1);
        assert_eq!(up, [0, 85, 170, 255]);
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// libwebp's shrinking import, a source pixel at a time.
    fn reference_shrink(
        src: &[u8],
        n: usize,
        x_add: i32,
        x_sub: i32,
        fx_scale: u32,
        out: &mut [u32],
    ) {
        let mut pixels = src.chunks_exact(n);
        let mut sum = vec![0u32; n];
        let mut accum = 0i32;
        for out in out.chunks_exact_mut(n) {
            let mut base = vec![0u32; n];
            accum += x_add;
            while accum > 0 {
                accum -= x_sub;
                let p = pixels.next().unwrap();
                for c in 0..n {
                    base[c] = u32::from(p[c]);
                    sum[c] = sum[c].wrapping_add(base[c]);
                }
            }
            for c in 0..n {
                let frac = base[c].wrapping_mul((-accum) as u32);
                out[c] = sum[c].wrapping_mul(x_sub as u32).wrapping_sub(frac);
                sum[c] = mult_fix(frac, fx_scale);
            }
        }
    }

    #[test]
    fn span_shrinks_match_the_pixel_loop() {
        let mut rng = Rng(11);
        for channels in 1..=4 {
            for (sw, dw) in [
                (3, 2),
                (5, 2),
                (7, 3),
                (101, 50),
                (6001, 3000),
                (3001, 751),
                (1001, 999),
                (17, 2),
                (9, 8),
                (3001, 3000),
                (6001, 1500),
                (6001, 750),
                (12003, 1500),
                (2000, 15),
            ] {
                let src: Vec<u8> = (0..sw * channels).map(|_| rng.next() as u8).collect();
                let mut r = Rescaler::new(sw, 4, dw, 2, channels);
                assert_eq!(r.whole_shrink, None);
                assert!(r.import(&src));
                let mut want = vec![0; dw * channels];
                reference_shrink(&src, channels, r.x_add, r.x_sub, r.fx_scale, &mut want);
                assert_eq!(r.frow, want, "{channels} channels, {sw} -> {dw}");
            }
        }
    }

    #[test]
    fn whole_shrinks_match_the_fraction_carrying_import() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for channels in 1..=4 {
            for k in [1, 2, 3, 4, 8, 9] {
                for dw in [1, 3, 4, 7, 16, 33, 100] {
                    let sw = dw * k;
                    let src: Vec<u8> = (0..sw * channels).map(|_| rng.next() as u8).collect();
                    let rows = |rng: &mut Rng, whole: bool, y_expand: bool| {
                        let mut r =
                            Rescaler::new(sw, 3, dw, if y_expand { 5 } else { 2 }, channels);
                        assert_eq!(r.whole_shrink, Some(k));
                        r.irow.iter_mut().for_each(|i| *i = rng.next() as u32);
                        if whole {
                            assert!(r.import(&src));
                        } else {
                            if y_expand {
                                std::mem::swap(&mut r.irow, &mut r.frow);
                            }
                            reference_shrink(
                                &src,
                                channels,
                                r.x_add,
                                r.x_sub,
                                r.fx_scale,
                                &mut r.frow,
                            );
                            if !y_expand {
                                for (i, f) in r.irow.iter_mut().zip(&r.frow) {
                                    *i = i.wrapping_add(*f);
                                }
                            }
                        }
                        (r.irow, r.frow)
                    };
                    let seed = rng.0;
                    let whole = (rows(&mut rng, true, false), rows(&mut rng, true, true));
                    rng.0 = seed;
                    assert_eq!(
                        whole,
                        (rows(&mut rng, false, false), rows(&mut rng, false, true)),
                        "{channels} channels, {sw} -> {dw}"
                    );
                }
            }
        }
    }

    #[test]
    fn export_shrink_matches_scalar() {
        let mut rng = Rng(7);
        for n in [1, 15, 16, 17, 48, 61] {
            for yscale in [0, 1 << 31, 0x1234_5678, u32::MAX] {
                for fxy in [1, 0x0100_0000, 0x4000_0000, u32::MAX] {
                    let small = rng.next().is_multiple_of(2);
                    let irow: Vec<u32> = (0..n)
                        .map(|_| {
                            if small {
                                (rng.next() % 4096) as u32
                            } else {
                                rng.next() as u32
                            }
                        })
                        .collect();
                    let frow: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
                    let (mut a, mut b) = (vec![0; n], vec![0; n]);
                    let (mut ia, mut ib) = (irow.clone(), irow);
                    let mut r = Rescaler::new(2 * n, 4, n, 2, 1);
                    (r.irow, r.frow, r.fy_scale, r.fxy_scale, r.y_accum) =
                        (ia.clone(), frow.clone(), yscale, fxy, -1);
                    r.export(&mut a);
                    ia = r.irow;
                    export_shrink(&mut b, &mut ib, &frow, yscale, fxy);
                    assert_eq!(
                        (a, ia),
                        (b, ib),
                        "{n} samples, yscale {yscale:x}, fxy {fxy:x}"
                    );
                }
            }
        }
    }
}
