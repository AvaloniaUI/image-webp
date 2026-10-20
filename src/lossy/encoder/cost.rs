//! Probabilities, bit costs and the token buffer of libwebp's `cost_enc.c`, `dsp/cost.c`,
//! `token_enc.c` and `frame_enc.c`.

use super::bit_writer::BitWriter;
use super::tables::{
    COEFFS_PROBA0, COEFFS_UPDATE_PROBA, ENC_BANDS, ENTROPY_COST, LEVEL_CODES, LEVEL_FIXED_COSTS,
};

pub(crate) const NUM_TYPES: usize = 4;
pub(crate) const NUM_BANDS: usize = 8;
pub(crate) const NUM_CTX: usize = 3;
pub(crate) const NUM_PROBAS: usize = 11;
pub(crate) const MAX_VARIABLE_LEVEL: usize = 67;
const MAX_LEVEL: u16 = 2047;

pub(crate) type ProbaArray = [[[u8; NUM_PROBAS]; NUM_CTX]; NUM_BANDS];
pub(crate) type StatsArray = [[[u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS];
pub(crate) type CostTable = [u16; MAX_VARIABLE_LEVEL + 1];

pub(crate) const CAT3: [u8; 3] = [173, 148, 140];
pub(crate) const CAT4: [u8; 4] = [176, 155, 140, 135];
pub(crate) const CAT5: [u8; 5] = [180, 157, 141, 134, 130];
pub(crate) const CAT6: [u8; 11] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];

#[inline]
pub(crate) fn bit_cost(bit: bool, proba: u8) -> i32 {
    i32::from(if bit {
        ENTROPY_COST[255 - proba as usize]
    } else {
        ENTROPY_COST[proba as usize]
    })
}

#[inline]
pub(crate) fn level_cost(table: &CostTable, level: usize) -> i32 {
    i32::from(LEVEL_FIXED_COSTS[level]) + i32::from(table[level.min(MAX_VARIABLE_LEVEL)])
}

#[inline]
pub(crate) fn record_stats(bit: bool, stats: &mut u32) -> bool {
    let mut p = *stats;
    if p >= 0xfffe_0000 {
        p = ((p + 1) >> 1) & 0x7fff_7fff;
    }
    p += 0x0001_0000 + u32::from(bit);
    *stats = p;
    bit
}

pub(crate) struct Proba {
    pub(crate) segments: [u8; 3],
    pub(crate) skip_proba: u8,
    pub(crate) coeffs: [ProbaArray; NUM_TYPES],
    pub(crate) stats: [StatsArray; NUM_TYPES],
    pub(crate) level_cost: Box<[[[CostTable; NUM_CTX]; NUM_BANDS]; NUM_TYPES]>,
    /// `level_cost` indexed by coefficient position instead of band.
    remapped_costs: Box<[[[CostTable; NUM_CTX]; 16]; NUM_TYPES]>,
    pub(crate) dirty: bool,
    pub(crate) use_skip_proba: bool,
    pub(crate) nb_skip: i32,
}

impl Proba {
    pub(crate) fn new() -> Self {
        Self {
            segments: [255; 3],
            skip_proba: 0,
            coeffs: COEFFS_PROBA0,
            stats: [[[[0; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
            level_cost: Box::new([[[[0; MAX_VARIABLE_LEVEL + 1]; NUM_CTX]; NUM_BANDS]; NUM_TYPES]),
            remapped_costs: Box::new([[[[0; MAX_VARIABLE_LEVEL + 1]; NUM_CTX]; 16]; NUM_TYPES]),
            dirty: true,
            use_skip_proba: false,
            nb_skip: 0,
        }
    }

    /// Cost table for coefficient `n` (0..=16) of `ctype` in context `ctx`.
    #[inline]
    pub(crate) fn costs(&self, ctype: usize, n: usize, ctx: usize) -> &CostTable {
        &self.level_cost[ctype][ENC_BANDS[n] as usize][ctx]
    }

    /// Cost tables of `ctype` by coefficient position (0..16) and context.
    #[inline]
    pub(crate) fn costs_by_position(&self, ctype: usize) -> &[[CostTable; NUM_CTX]; 16] {
        &self.remapped_costs[ctype]
    }

    pub(crate) fn calculate_level_costs(&mut self) {
        if !self.dirty {
            return;
        }
        for ctype in 0..NUM_TYPES {
            for band in 0..NUM_BANDS {
                for ctx in 0..NUM_CTX {
                    let p = &self.coeffs[ctype][band][ctx];
                    let table = &mut self.level_cost[ctype][band][ctx];
                    let cost0 = if ctx > 0 { bit_cost(true, p[0]) } else { 0 };
                    let cost_base = bit_cost(true, p[1]) + cost0;
                    table[0] = (bit_cost(false, p[1]) + cost0) as u16;
                    for v in 1..=MAX_VARIABLE_LEVEL {
                        table[v] = (cost_base + variable_level_cost(v, p)) as u16;
                    }
                }
            }
            for n in 0..16 {
                self.remapped_costs[ctype][n] = self.level_cost[ctype][ENC_BANDS[n] as usize];
            }
        }
        self.dirty = false;
    }

    pub(crate) fn reset_token_stats(&mut self) {
        self.stats = [[[[0; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES];
    }

    /// Picks the probabilities worth updating from the collected statistics.
    pub(crate) fn finalize_token_probas(&mut self) -> i32 {
        let mut has_changed = false;
        let mut size = 0;
        for t in 0..NUM_TYPES {
            for b in 0..NUM_BANDS {
                for c in 0..NUM_CTX {
                    for p in 0..NUM_PROBAS {
                        let stats = self.stats[t][b][c][p];
                        let nb = (stats & 0xffff) as i32;
                        let total = ((stats >> 16) & 0xffff) as i32;
                        let update_proba = COEFFS_UPDATE_PROBA[t][b][c][p];
                        let old_p = COEFFS_PROBA0[t][b][c][p];
                        let new_p = calc_token_proba(nb, total);
                        let old_cost =
                            branch_cost(nb, total, old_p) + bit_cost(false, update_proba);
                        let new_cost =
                            branch_cost(nb, total, new_p) + bit_cost(true, update_proba) + 8 * 256;
                        let use_new_p = old_cost > new_cost;
                        size += bit_cost(use_new_p, update_proba);
                        if use_new_p {
                            self.coeffs[t][b][c][p] = new_p;
                            has_changed |= new_p != old_p;
                            size += 8 * 256;
                        } else {
                            self.coeffs[t][b][c][p] = old_p;
                        }
                    }
                }
            }
        }
        self.dirty = has_changed;
        size
    }

    pub(crate) fn write(&self, bw: &mut BitWriter) {
        for t in 0..NUM_TYPES {
            for b in 0..NUM_BANDS {
                for c in 0..NUM_CTX {
                    for p in 0..NUM_PROBAS {
                        let p0 = self.coeffs[t][b][c][p];
                        let update = p0 != COEFFS_PROBA0[t][b][c][p];
                        if bw.put_bit(update, COEFFS_UPDATE_PROBA[t][b][c][p]) {
                            bw.put_bits(u32::from(p0), 8);
                        }
                    }
                }
            }
        }
        if bw.put_bit_uniform(self.use_skip_proba) {
            bw.put_bits(u32::from(self.skip_proba), 8);
        }
    }
}

fn variable_level_cost(level: usize, probas: &[u8; NUM_PROBAS]) -> i32 {
    let mut pattern = LEVEL_CODES[level - 1][0];
    let mut bits = LEVEL_CODES[level - 1][1];
    let mut cost = 0;
    let mut i = 2;
    while pattern != 0 {
        if pattern & 1 != 0 {
            cost += bit_cost(bits & 1 != 0, probas[i]);
        }
        bits >>= 1;
        pattern >>= 1;
        i += 1;
    }
    cost
}

fn calc_token_proba(nb: i32, total: i32) -> u8 {
    if nb != 0 {
        (255 - nb * 255 / total) as u8
    } else {
        255
    }
}

fn branch_cost(nb: i32, total: i32, proba: u8) -> i32 {
    nb * bit_cost(true, proba) + (total - nb) * bit_cost(false, proba)
}

/// A block of quantized levels (zigzag order) about to be coded.
pub(crate) struct Residual<'a> {
    pub(crate) first: usize,
    pub(crate) last: i32,
    pub(crate) coeffs: &'a [i16],
    pub(crate) coeff_type: usize,
}

impl<'a> Residual<'a> {
    pub(crate) fn new(first: usize, coeff_type: usize, coeffs: &'a [i16]) -> Self {
        let last = coeffs[..16]
            .iter()
            .rposition(|&c| c != 0)
            .map_or(-1, |n| n as i32);
        Self {
            first,
            last,
            coeffs,
            coeff_type,
        }
    }

    pub(crate) fn cost(&self, ctx0: usize, proba: &Proba) -> i32 {
        let t = self.coeff_type;
        let first = self.first;
        let probas = &proba.coeffs[t];
        let p0 = probas[first][ctx0][0];
        if self.last < 0 {
            return bit_cost(false, p0);
        }
        let costs = proba.costs_by_position(t);
        let coeffs: &[i16; 16] = self.coeffs[..16].try_into().unwrap();
        let abs: [u16; 16] = std::array::from_fn(|i| coeffs[i].unsigned_abs().min(MAX_LEVEL));
        let mut cost = if ctx0 == 0 { bit_cost(true, p0) } else { 0 };
        let mut table = &costs[first][ctx0];
        let last = (self.last as usize).min(15);
        for n in first..last {
            let v = usize::from(abs[n]);
            cost += i32::from(LEVEL_FIXED_COSTS[v]) + i32::from(table[v.min(MAX_VARIABLE_LEVEL)]);
            table = &costs[n + 1][v.min(2)];
        }
        let v = usize::from(abs[last]);
        cost += i32::from(LEVEL_FIXED_COSTS[v]) + i32::from(table[v.min(MAX_VARIABLE_LEVEL)]);
        if last < 15 {
            let b = ENC_BANDS[last + 1] as usize;
            let ctx = if v == 1 { 1 } else { 2 };
            cost += bit_cost(false, probas[b][ctx][0]);
        }
        cost
    }
}

/// The coded bits of all macroblocks, one bit each, emitted once the final probabilities are
/// known. libwebp stores each bit with its probability index (16 bits); that index follows from
/// the bits before it, so `emit_block` walks the token tree again instead.
#[derive(Default)]
pub(crate) struct TokenBuffer {
    pages: Vec<Box<[u64; PAGE_WORDS]>>,
    len: usize,
    /// The bits of the word being filled, stored once it is complete.
    word: u64,
}

const PAGE_WORDS: usize = 1 << 13;

#[inline]
fn token_id(t: usize, b: usize, ctx: usize) -> usize {
    NUM_PROBAS * (ctx + NUM_CTX * (b + NUM_BANDS * t))
}

impl TokenBuffer {
    pub(crate) fn clear(&mut self) {
        self.pages.clear();
        self.len = 0;
        self.word = 0;
    }

    #[inline]
    fn push(&mut self, bit: bool) {
        self.word |= u64::from(bit) << (self.len % 64);
        self.len += 1;
        if self.len % 64 == 0 {
            self.store_word();
        }
    }

    fn store_word(&mut self) {
        let w = self.len / 64 - 1;
        let (page, idx) = (w / PAGE_WORDS, w % PAGE_WORDS);
        if page == self.pages.len() {
            self.pages.push(Box::new([0; PAGE_WORDS]));
        }
        self.pages[page][idx] = self.word;
        self.word = 0;
    }

    #[inline]
    fn add(&mut self, bit: bool, stats: &mut u32) -> bool {
        self.push(bit);
        record_stats(bit, stats);
        bit
    }

    /// Records the coding of `res` in context `ctx`, returning whether it has non-zero levels.
    pub(crate) fn record_coeff_tokens(
        &mut self,
        ctx: usize,
        res: &Residual,
        stats: &mut StatsArray,
    ) -> bool {
        let last = res.last;
        let mut n = res.first;
        let mut s = (n, ctx);
        if !self.add(last >= 0, &mut stats[s.0][s.1][0]) {
            return false;
        }
        while n < 16 {
            let c = res.coeffs[n];
            n += 1;
            let sign = c < 0;
            let v = u32::from(c.unsigned_abs());
            if !self.add(v != 0, &mut stats[s.0][s.1][1]) {
                s = (ENC_BANDS[n] as usize, 0);
                continue;
            }
            if !self.add(v > 1, &mut stats[s.0][s.1][2]) {
                s = (ENC_BANDS[n] as usize, 1);
            } else {
                let st = &mut stats[s.0][s.1];
                if !self.add(v > 4, &mut st[3]) {
                    if self.add(v != 2, &mut st[4]) {
                        self.add(v == 4, &mut st[5]);
                    }
                } else if !self.add(v > 10, &mut st[6]) {
                    if !self.add(v > 6, &mut st[7]) {
                        self.push(v == 6);
                    } else {
                        self.push(v >= 9);
                        self.push(v & 1 == 0);
                    }
                } else {
                    let mut residue = v - 3;
                    let mut mask: u32 = if residue < (8 << 1) {
                        self.add(false, &mut st[8]);
                        self.add(false, &mut st[9]);
                        residue -= 8 << 0;
                        1 << 2
                    } else if residue < (8 << 2) {
                        self.add(false, &mut st[8]);
                        self.add(true, &mut st[9]);
                        residue -= 8 << 1;
                        1 << 3
                    } else if residue < (8 << 3) {
                        self.add(true, &mut st[8]);
                        // libwebp codes this bit with probability 10 but records it in slot 9.
                        self.add(false, &mut st[9]);
                        residue -= 8 << 2;
                        1 << 4
                    } else {
                        self.add(true, &mut st[8]);
                        self.add(true, &mut st[9]);
                        residue -= 8 << 3;
                        1 << 10
                    };
                    while mask != 0 {
                        self.push(residue & mask != 0);
                        mask >>= 1;
                    }
                }
                s = (ENC_BANDS[n] as usize, 2);
            }
            self.push(sign);
            if n == 16 || !self.add(n as i32 <= last, &mut stats[s.0][s.1][0]) {
                return true;
            }
        }
        true
    }

    pub(crate) fn reader(&self) -> TokenReader<'_> {
        TokenReader {
            buffer: self,
            pos: 0,
            word: 0,
        }
    }
}

/// Reads back a `TokenBuffer` in recording order.
pub(crate) struct TokenReader<'a> {
    buffer: &'a TokenBuffer,
    pos: usize,
    /// The unread bits of the current word, next bit lowest.
    word: u64,
}

impl TokenReader<'_> {
    #[inline]
    fn next(&mut self) -> bool {
        if self.pos % 64 == 0 {
            self.word = self.load_word(self.pos / 64);
        }
        self.pos += 1;
        let bit = self.word & 1 != 0;
        self.word >>= 1;
        bit
    }

    #[inline(always)]
    fn put(&mut self, bw: &mut BitWriter, proba: u8) -> bool {
        let bit = self.next();
        bw.put_bit(bit, proba);
        bit
    }

    fn load_word(&self, w: usize) -> u64 {
        if w < self.buffer.len / 64 {
            self.buffer.pages[w / PAGE_WORDS][w % PAGE_WORDS]
        } else {
            self.buffer.word
        }
    }

    /// Codes the next recorded block of `coeff_type` from coefficient `first` in context
    /// `ctx`, following the same tree as `record_coeff_tokens`. Returns whether it had
    /// non-zero levels.
    pub(crate) fn emit_block(
        &mut self,
        bw: &mut BitWriter,
        probas: &[u8],
        coeff_type: usize,
        first: usize,
        ctx: usize,
    ) -> bool {
        let mut n = first;
        let mut base = token_id(coeff_type, n, ctx);
        if !self.put(bw, probas[base]) {
            return false;
        }
        while n < 16 {
            n += 1;
            if !self.put(bw, probas[base + 1]) {
                base = token_id(coeff_type, ENC_BANDS[n] as usize, 0);
                continue;
            }
            if !self.put(bw, probas[base + 2]) {
                base = token_id(coeff_type, ENC_BANDS[n] as usize, 1);
            } else {
                if !self.put(bw, probas[base + 3]) {
                    if self.put(bw, probas[base + 4]) {
                        self.put(bw, probas[base + 5]);
                    }
                } else if !self.put(bw, probas[base + 6]) {
                    if !self.put(bw, probas[base + 7]) {
                        bw.put_bit(self.next(), 159);
                    } else {
                        bw.put_bit(self.next(), 165);
                        bw.put_bit(self.next(), 145);
                    }
                } else {
                    let tab: &[u8] = if !self.put(bw, probas[base + 8]) {
                        if !self.put(bw, probas[base + 9]) {
                            &CAT3
                        } else {
                            &CAT4
                        }
                    } else if !self.put(bw, probas[base + 10]) {
                        &CAT5
                    } else {
                        &CAT6
                    };
                    for &p in tab {
                        bw.put_bit(self.next(), p);
                    }
                }
                base = token_id(coeff_type, ENC_BANDS[n] as usize, 2);
            }
            bw.put_bit(self.next(), 128);
            if n == 16 || !self.put(bw, probas[base]) {
                return true;
            }
        }
        true
    }
}

impl Residual<'_> {
    /// Records the statistics of coding `self` in context `ctx` without coding it.
    pub(crate) fn record(&self, ctx: usize, stats: &mut StatsArray) -> bool {
        let mut n = self.first;
        let mut s = (n, ctx);
        if self.last < 0 {
            record_stats(false, &mut stats[s.0][s.1][0]);
            return false;
        }
        while n as i32 <= self.last {
            record_stats(true, &mut stats[s.0][s.1][0]);
            let mut v;
            loop {
                v = i32::from(self.coeffs[n]);
                n += 1;
                if v != 0 {
                    break;
                }
                record_stats(false, &mut stats[s.0][s.1][1]);
                s = (ENC_BANDS[n] as usize, 0);
            }
            record_stats(true, &mut stats[s.0][s.1][1]);
            if !record_stats(2u32 < (v + 1) as u32, &mut stats[s.0][s.1][2]) {
                s = (ENC_BANDS[n] as usize, 1);
            } else {
                let v = (v.unsigned_abs() as usize).min(MAX_VARIABLE_LEVEL);
                let bits = LEVEL_CODES[v - 1][1];
                let mut pattern = LEVEL_CODES[v - 1][0];
                let mut i = 0;
                loop {
                    pattern >>= 1;
                    if pattern == 0 {
                        break;
                    }
                    let mask = 2 << i;
                    if pattern & 1 != 0 {
                        record_stats(bits & mask != 0, &mut stats[s.0][s.1][3 + i]);
                    }
                    i += 1;
                }
                s = (ENC_BANDS[n] as usize, 2);
            }
        }
        if n < 16 {
            record_stats(false, &mut stats[s.0][s.1][0]);
        }
        true
    }

    /// Codes `self` in context `ctx` with `probas` (those of its coefficient type).
    pub(crate) fn put(&self, bw: &mut BitWriter, ctx: usize, probas: &ProbaArray) -> bool {
        let mut n = self.first;
        let mut p = &probas[n][ctx];
        if !bw.put_bit(self.last >= 0, p[0]) {
            return false;
        }
        while n < 16 {
            let c = self.coeffs[n];
            n += 1;
            let sign = c < 0;
            let mut v = u32::from(c.unsigned_abs());
            if !bw.put_bit(v != 0, p[1]) {
                p = &probas[ENC_BANDS[n] as usize][0];
                continue;
            }
            if !bw.put_bit(v > 1, p[2]) {
                p = &probas[ENC_BANDS[n] as usize][1];
            } else {
                if !bw.put_bit(v > 4, p[3]) {
                    if bw.put_bit(v != 2, p[4]) {
                        bw.put_bit(v == 4, p[5]);
                    }
                } else if !bw.put_bit(v > 10, p[6]) {
                    if !bw.put_bit(v > 6, p[7]) {
                        bw.put_bit(v == 6, 159);
                    } else {
                        bw.put_bit(v >= 9, 165);
                        bw.put_bit(v & 1 == 0, 145);
                    }
                } else {
                    let (mut mask, tab): (u32, &[u8]) = if v < 3 + (8 << 1) {
                        bw.put_bit(false, p[8]);
                        bw.put_bit(false, p[9]);
                        v -= 3 + (8 << 0);
                        (1 << 2, &CAT3)
                    } else if v < 3 + (8 << 2) {
                        bw.put_bit(false, p[8]);
                        bw.put_bit(true, p[9]);
                        v -= 3 + (8 << 1);
                        (1 << 3, &CAT4)
                    } else if v < 3 + (8 << 3) {
                        bw.put_bit(true, p[8]);
                        bw.put_bit(false, p[10]);
                        v -= 3 + (8 << 2);
                        (1 << 4, &CAT5)
                    } else {
                        bw.put_bit(true, p[8]);
                        bw.put_bit(true, p[10]);
                        v -= 3 + (8 << 3);
                        (1 << 10, &CAT6)
                    };
                    let mut k = 0;
                    while mask != 0 {
                        bw.put_bit(v & mask != 0, tab[k]);
                        k += 1;
                        mask >>= 1;
                    }
                }
                p = &probas[ENC_BANDS[n] as usize][2];
            }
            bw.put_bit_uniform(sign);
            if n == 16 || !bw.put_bit(n as i32 <= self.last, p[0]) {
                return true;
            }
        }
        true
    }
}

impl Proba {
    /// Decides whether to code skip flags, returning the bit cost of doing so.
    pub(crate) fn finalize_skip_proba(&mut self, nb_mbs: i32) -> i32 {
        const SKIP_PROBA_THRESHOLD: u8 = 250;
        let nb_events = self.nb_skip;
        self.skip_proba = if nb_mbs != 0 {
            ((nb_mbs - nb_events) as i64 * 255 / nb_mbs as i64) as u8
        } else {
            255
        };
        self.use_skip_proba = self.skip_proba < SKIP_PROBA_THRESHOLD;
        let mut size = 256;
        if self.use_skip_proba {
            size += nb_events * bit_cost(true, self.skip_proba)
                + (nb_mbs - nb_events) * bit_cost(false, self.skip_proba);
            size += 8 * 256;
        }
        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_buffer_reads_back_across_pages_and_clear() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut tokens = TokenBuffer::default();
        for len in [
            0,
            1,
            63,
            64,
            65,
            PAGE_WORDS * 64 + 1,
            2 * PAGE_WORDS * 64 + 77,
        ] {
            tokens.clear();
            let bits: Vec<bool> = (0..len).map(|_| rng.gen()).collect();
            for &b in &bits {
                tokens.push(b);
            }
            let mut reader = tokens.reader();
            assert!(bits.iter().all(|&b| reader.next() == b), "len {len}");
        }
    }
}
