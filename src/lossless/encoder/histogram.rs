//! Symbol histograms and their clustering into the histogram image (libwebp's
//! `enc/histogram_enc.c`).

use super::backward_refs::{PixOrCopy, PixOrCopyMode};
use super::entropy::*;
use super::{NUM_DISTANCE_CODES, NUM_LENGTH_CODES, NUM_LITERAL_CODES};

const NUM_PARTITIONS: usize = 4;
const BIN_SIZE: usize = NUM_PARTITIONS * NUM_PARTITIONS * NUM_PARTITIONS;
const MAX_HISTO_GREEDY: i64 = 100;

const LITERAL: usize = 0;
const RED: usize = 1;
const BLUE: usize = 2;
const ALPHA: usize = 3;
const DISTANCE: usize = 4;

pub(crate) fn histogram_num_codes(palette_code_bits: u32) -> usize {
    NUM_LITERAL_CODES
        + NUM_LENGTH_CODES
        + if palette_code_bits > 0 {
            1 << palette_code_bits
        } else {
            0
        }
}

#[derive(Clone)]
pub(crate) struct Histogram {
    pub(crate) literal: Vec<u32>,
    pub(crate) red: [u32; 256],
    pub(crate) blue: [u32; 256],
    pub(crate) alpha: [u32; 256],
    pub(crate) distance: [u32; NUM_DISTANCE_CODES],
    pub(crate) palette_code_bits: u32,
    trivial_symbol: [u16; 5],
    bit_cost: u64,
    costs: [u64; 5],
    is_used: [bool; 5],
    bin_id: u16,
}

impl Histogram {
    pub(crate) fn new(cache_bits: u32) -> Self {
        Self {
            literal: vec![0; histogram_num_codes(cache_bits)],
            red: [0; 256],
            blue: [0; 256],
            alpha: [0; 256],
            distance: [0; NUM_DISTANCE_CODES],
            palette_code_bits: cache_bits,
            trivial_symbol: [NON_TRIVIAL_SYM; 5],
            bit_cost: 0,
            costs: [0; 5],
            is_used: [true; 5],
            bin_id: 0,
        }
    }

    /// `HistogramClear`: zero counts, default statistics.
    pub(crate) fn clear(&mut self) {
        self.literal.fill(0);
        self.red = [0; 256];
        self.blue = [0; 256];
        self.alpha = [0; 256];
        self.distance = [0; NUM_DISTANCE_CODES];
        self.trivial_symbol = [NON_TRIVIAL_SYM; 5];
        self.is_used = [true; 5];
        self.bit_cost = 0;
        self.costs = [0; 5];
        self.bin_id = 0;
    }

    /// `VP8LHistogramSetClear` on one entry: everything zero, including the statistics.
    fn zero(&mut self) {
        self.clear();
        self.trivial_symbol = [0; 5];
        self.is_used = [false; 5];
    }

    pub(crate) fn population(&self, index: usize) -> &[u32] {
        match index {
            LITERAL => &self.literal,
            RED => &self.red,
            BLUE => &self.blue,
            ALPHA => &self.alpha,
            _ => &self.distance,
        }
    }

    pub(crate) fn population_mut(&mut self, index: usize) -> &mut [u32] {
        match index {
            LITERAL => &mut self.literal,
            RED => &mut self.red,
            BLUE => &mut self.blue,
            ALPHA => &mut self.alpha,
            _ => &mut self.distance,
        }
    }

    #[inline]
    pub(crate) fn add_single(&mut self, v: &PixOrCopy, distance_modifier: Option<u32>) {
        match v.mode {
            PixOrCopyMode::Literal => {
                let argb = v.argb_or_distance;
                self.alpha[(argb >> 24) as usize] += 1;
                self.red[((argb >> 16) & 0xff) as usize] += 1;
                self.literal[((argb >> 8) & 0xff) as usize] += 1;
                self.blue[(argb & 0xff) as usize] += 1;
            }
            PixOrCopyMode::CacheIdx => {
                self.literal[NUM_LITERAL_CODES + NUM_LENGTH_CODES + v.argb_or_distance as usize] +=
                    1;
            }
            PixOrCopyMode::Copy => {
                let (code, _) = prefix_encode_bits(u32::from(v.len));
                self.literal[NUM_LITERAL_CODES + code as usize] += 1;
                let dist = match distance_modifier {
                    None => v.argb_or_distance,
                    Some(xsize) => {
                        super::backward_refs::distance_to_plane_code(xsize, v.argb_or_distance)
                    }
                };
                let (code, _) = prefix_encode_bits(dist);
                self.distance[code as usize] += 1;
            }
        }
    }

    pub(crate) fn store_refs(&mut self, refs: &[PixOrCopy], distance_modifier: Option<u32>) {
        for v in refs {
            self.add_single(v, distance_modifier);
        }
    }

    /// `VP8LHistogramCreate`.
    pub(crate) fn create(&mut self, refs: &[PixOrCopy], palette_code_bits: u32) {
        if self.palette_code_bits != palette_code_bits {
            self.palette_code_bits = palette_code_bits;
            self.literal = vec![0; histogram_num_codes(palette_code_bits)];
        }
        self.clear();
        self.store_refs(refs, None);
    }

    /// `VP8LHistogramEstimateBits`.
    pub(crate) fn estimate_bits(&self) -> u64 {
        let mut cost = 0u64;
        for i in 0..5 {
            cost = cost.wrapping_add(population_cost(self.population(i)).0);
        }
        cost.wrapping_add(
            u64::from(
                extra_cost(&self.literal[NUM_LITERAL_CODES..NUM_LITERAL_CODES + NUM_LENGTH_CODES])
                    .wrapping_add(extra_cost(&self.distance)),
            ) << LOG_2_PRECISION_BITS,
        )
    }

    fn compute_cost(&mut self) {
        for i in 0..5 {
            let (cost, trivial, used) = population_cost(self.population(i));
            self.costs[i] = cost;
            self.trivial_symbol[i] = trivial;
            self.is_used[i] = used;
        }
        self.bit_cost = self.costs[LITERAL]
            .wrapping_add(self.costs[RED])
            .wrapping_add(self.costs[BLUE])
            .wrapping_add(self.costs[ALPHA])
            .wrapping_add(self.costs[DISTANCE]);
    }

    fn update_cost(&mut self, bit_cost: u64, costs: &[u64; 5]) {
        self.bit_cost = bit_cost;
        self.costs = *costs;
    }

    fn is_empty(&self) -> bool {
        !self.is_used.iter().any(|&u| u)
    }
}

fn initial_huffman_cost() -> u64 {
    let huffman_code_of_huffman_code_size = (19u64 * 3) << LOG_2_PRECISION_BITS;
    huffman_code_of_huffman_code_size - div_round(91i64 << LOG_2_PRECISION_BITS, 10) as u64
}

fn final_huffman_cost(stats: &Streaks) -> u64 {
    let retval = initial_huffman_cost();
    let mut retval_extra = (stats.counts[0] as u32)
        .wrapping_mul(1600)
        .wrapping_add(240u32.wrapping_mul(stats.streaks[0][1] as u32));
    retval_extra = retval_extra
        .wrapping_add((stats.counts[1] as u32).wrapping_mul(2640))
        .wrapping_add(720u32.wrapping_mul(stats.streaks[1][1] as u32));
    retval_extra = retval_extra.wrapping_add(1840u32.wrapping_mul(stats.streaks[0][0] as u32));
    retval_extra = retval_extra.wrapping_add(3360u32.wrapping_mul(stats.streaks[1][0] as u32));
    retval.wrapping_add(u64::from(retval_extra) << (LOG_2_PRECISION_BITS - 10))
}

/// `PopulationCost`: (cost, trivial symbol, is used).
fn population_cost(population: &[u32]) -> (u64, u16, bool) {
    let (e, stats) = get_entropy_unrefined(population);
    let trivial = if e.nonzeros == 1 {
        e.nonzero_code as u16
    } else {
        NON_TRIVIAL_SYM
    };
    let used = stats.streaks[1][0] != 0 || stats.streaks[1][1] != 0;
    (
        bits_entropy_refine(&e).wrapping_add(final_huffman_cost(&stats)),
        trivial,
        used,
    )
}

fn get_combined_entropy(h1: &Histogram, h2: &Histogram, index: usize) -> u64 {
    let is_h1_used = h1.is_used[index];
    let is_h2_used = h2.is_used[index];
    let is_trivial = h1.trivial_symbol[index] != NON_TRIVIAL_SYM
        && h1.trivial_symbol[index] == h2.trivial_symbol[index];
    if is_trivial || !is_h1_used || !is_h2_used {
        return if is_h1_used {
            h1.costs[index]
        } else {
            h2.costs[index]
        };
    }
    let (e, stats) = get_combined_entropy_unrefined(h1.population(index), h2.population(index));
    bits_entropy_refine(&e).wrapping_add(final_huffman_cost(&stats))
}

fn saturate_add(a: u64, b: &mut i64) {
    if *b < 0 || (a as i64) <= i64::MAX - *b {
        *b = b.wrapping_add(a as i64);
    } else {
        *b = i64::MAX;
    }
}

/// `GetCombinedHistogramEntropy`: `Some((cost, costs))` when below the threshold.
fn get_combined_histogram_entropy(
    a: &Histogram,
    b: &Histogram,
    cost_threshold_in: i64,
) -> Option<(u64, [u64; 5])> {
    if cost_threshold_in <= 0 {
        return None;
    }
    let cost_threshold = cost_threshold_in as u64;
    let mut cost = 0u64;
    let mut costs = [0u64; 5];
    for i in 0..5 {
        costs[i] = get_combined_entropy(a, b, i);
        cost = cost.wrapping_add(costs[i]);
        if cost >= cost_threshold {
            return None;
        }
    }
    Some((cost, costs))
}

/// `HistogramAdd(h1, hout, hout)`.
fn histogram_add_into(h1: &Histogram, hout: &mut Histogram) {
    for i in 0..5 {
        if h1.is_used[i] {
            let p1 = h1.population(i);
            let used = hout.is_used[i];
            let pout = hout.population_mut(i);
            if used {
                for (o, &a) in pout.iter_mut().zip(p1) {
                    *o = o.wrapping_add(a);
                }
            } else {
                pout.copy_from_slice(p1);
            }
        }
    }
    for i in 0..5 {
        if h1.trivial_symbol[i] != hout.trivial_symbol[i] {
            hout.trivial_symbol[i] = NON_TRIVIAL_SYM;
        }
        hout.is_used[i] = h1.is_used[i] || hout.is_used[i];
    }
}

/// `HistogramAdd(h1, h2, hout)` with three distinct histograms.
fn histogram_add(h1: &Histogram, h2: &Histogram, hout: &mut Histogram) {
    for i in 0..5 {
        let (p1, p2) = (h1.population(i), h2.population(i));
        let pout = hout.population_mut(i);
        match (h1.is_used[i], h2.is_used[i]) {
            (true, true) => {
                for ((o, &a), &b) in pout.iter_mut().zip(p1).zip(p2) {
                    *o = a.wrapping_add(b);
                }
            }
            (true, false) => pout.copy_from_slice(p1),
            (false, true) => pout.copy_from_slice(p2),
            (false, false) => pout.fill(0),
        }
    }
    for i in 0..5 {
        hout.trivial_symbol[i] = if h1.trivial_symbol[i] == h2.trivial_symbol[i] {
            h1.trivial_symbol[i]
        } else {
            NON_TRIVIAL_SYM
        };
        hout.is_used[i] = h1.is_used[i] || h2.is_used[i];
    }
}

fn histogram_add_eval(
    a: &Histogram,
    b: &Histogram,
    out: &mut Histogram,
    mut cost_threshold: i64,
) -> bool {
    let sum_cost = a.bit_cost.wrapping_add(b.bit_cost);
    saturate_add(sum_cost, &mut cost_threshold);
    match get_combined_histogram_entropy(a, b, cost_threshold) {
        None => false,
        Some((bit_cost, costs)) => {
            histogram_add(a, b, out);
            out.update_cost(bit_cost, &costs);
            true
        }
    }
}

fn histogram_add_thresh(a: &Histogram, b: &Histogram, mut cost_threshold: i64) -> Option<i64> {
    saturate_add(a.bit_cost, &mut cost_threshold);
    get_combined_histogram_entropy(a, b, cost_threshold)
        .map(|(cost, _)| cost as i64 - a.bit_cost as i64)
}

fn get_bin_id_for_entropy(min: u64, max: u64, val: u64) -> usize {
    let range = max.wrapping_sub(min);
    if range > 0 {
        let delta = val.wrapping_sub(min);
        ((NUM_PARTITIONS as f64 - 1e-6) * delta as f64 / range as f64) as usize
    } else {
        0
    }
}

fn analyze_entropy_bin(histograms: &mut [Histogram], low_effort: bool) {
    let (mut lmax, mut lmin, mut rmax, mut rmin, mut bmax, mut bmin) =
        (0u64, u64::MAX, 0u64, u64::MAX, 0u64, u64::MAX);
    for h in histograms.iter() {
        lmax = lmax.max(h.costs[LITERAL]);
        lmin = lmin.min(h.costs[LITERAL]);
        rmax = rmax.max(h.costs[RED]);
        rmin = rmin.min(h.costs[RED]);
        bmax = bmax.max(h.costs[BLUE]);
        bmin = bmin.min(h.costs[BLUE]);
    }
    for h in histograms.iter_mut() {
        let mut bin_id = get_bin_id_for_entropy(lmin, lmax, h.costs[LITERAL]);
        if !low_effort {
            bin_id = bin_id * NUM_PARTITIONS + get_bin_id_for_entropy(rmin, rmax, h.costs[RED]);
            bin_id = bin_id * NUM_PARTITIONS + get_bin_id_for_entropy(bmin, bmax, h.costs[BLUE]);
        }
        h.bin_id = bin_id as u16;
    }
}

fn combine_entropy_bin(
    histograms: &mut Vec<Histogram>,
    cur_combo: &mut Histogram,
    num_bins: usize,
    combine_cost_factor: i32,
    low_effort: bool,
) {
    let mut first_of = [-1i32; BIN_SIZE];
    let mut failures = [0u16; BIN_SIZE];
    let _ = num_bins;
    let mut idx = 0;
    while idx < histograms.len() {
        let bin_id = histograms[idx].bin_id as usize;
        let first = first_of[bin_id];
        if first == -1 {
            first_of[bin_id] = idx as i32;
            idx += 1;
        } else if low_effort {
            let first = first as usize;
            let (h_idx, h_first) = pair(histograms, idx, first);
            histogram_add_into(h_idx, h_first);
            histograms.swap_remove(idx);
        } else {
            let first = first as usize;
            let bit_cost = histograms[idx].bit_cost;
            let bit_cost_thresh = -div_round(
                (bit_cost as i64).wrapping_mul(i64::from(combine_cost_factor)),
                100,
            );
            if histogram_add_eval(
                &histograms[first],
                &histograms[idx],
                cur_combo,
                bit_cost_thresh,
            ) {
                let max_combine_failures = 32;
                let nt = |h: &Histogram, i: usize| h.trivial_symbol[i] == NON_TRIVIAL_SYM;
                let mut try_combine =
                    !nt(cur_combo, RED) && !nt(cur_combo, BLUE) && !nt(cur_combo, ALPHA);
                if !try_combine {
                    let hi = &histograms[idx];
                    let hf = &histograms[first];
                    try_combine = (nt(hi, RED) || nt(hi, BLUE) || nt(hi, ALPHA))
                        && (nt(hf, RED) || nt(hf, BLUE) || nt(hf, ALPHA));
                }
                if try_combine || failures[bin_id] >= max_combine_failures {
                    std::mem::swap(cur_combo, &mut histograms[first]);
                    histograms.swap_remove(idx);
                } else {
                    failures[bin_id] += 1;
                    idx += 1;
                }
            } else {
                idx += 1;
            }
        }
    }
    if low_effort {
        for h in histograms.iter_mut() {
            h.compute_cost();
        }
    }
}

/// Mutable `a` and shared `b` at distinct indices.
fn pair<T>(v: &mut [T], shared: usize, exclusive: usize) -> (&T, &mut T) {
    assert_ne!(shared, exclusive);
    if shared < exclusive {
        let (lo, hi) = v.split_at_mut(exclusive);
        (&lo[shared], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(shared);
        (&hi[0], &mut lo[exclusive])
    }
}

fn my_rand(seed: &mut u32) -> u32 {
    *seed = ((u64::from(*seed) * 48271) % 2147483647) as u32;
    *seed
}

#[derive(Clone, Copy, Default)]
struct HistogramPair {
    idx1: i32,
    idx2: i32,
    cost_diff: i64,
    cost_combo: u64,
    costs: [u64; 5],
}

struct HistoQueue {
    queue: Vec<HistogramPair>,
    max_size: usize,
}

impl HistoQueue {
    fn new(max_size: usize) -> Self {
        Self {
            queue: Vec::with_capacity(max_size.min(1 << 16)),
            max_size,
        }
    }

    fn pop_pair(&mut self, i: usize) {
        let last = self.queue.pop().unwrap();
        if i < self.queue.len() {
            self.queue[i] = last;
        }
    }

    fn update_head(&mut self, i: usize) {
        if self.queue[i].cost_diff < self.queue[0].cost_diff {
            self.queue.swap(0, i);
        }
    }

    /// Returns the pair's cost difference, or 0 if not pushed.
    fn push(&mut self, histograms: &[Histogram], idx1: usize, idx2: usize, threshold: i64) -> i64 {
        if self.queue.len() == self.max_size {
            return 0;
        }
        let (idx1, idx2) = if idx1 > idx2 {
            (idx2, idx1)
        } else {
            (idx1, idx2)
        };
        let mut pair = HistogramPair {
            idx1: idx1 as i32,
            idx2: idx2 as i32,
            ..Default::default()
        };
        if !update_pair(&histograms[idx1], &histograms[idx2], threshold, &mut pair) {
            return 0;
        }
        self.queue.push(pair);
        let last = self.queue.len() - 1;
        self.update_head(last);
        pair.cost_diff
    }
}

fn fix_pair(bad_id: i32, good_id: i32, pair: &mut HistogramPair) {
    if pair.idx1 == bad_id {
        pair.idx1 = good_id;
    }
    if pair.idx2 == bad_id {
        pair.idx2 = good_id;
    }
    if pair.idx1 > pair.idx2 {
        std::mem::swap(&mut pair.idx1, &mut pair.idx2);
    }
}

fn update_pair(
    h1: &Histogram,
    h2: &Histogram,
    mut cost_threshold: i64,
    pair: &mut HistogramPair,
) -> bool {
    let sum_cost = h1.bit_cost.wrapping_add(h2.bit_cost) as i64;
    saturate_add(sum_cost as u64, &mut cost_threshold);
    match get_combined_histogram_entropy(h1, h2, cost_threshold) {
        None => false,
        Some((cost, costs)) => {
            pair.cost_combo = cost;
            pair.costs = costs;
            pair.cost_diff = (cost as i64).wrapping_sub(sum_cost);
            true
        }
    }
}

fn merge_into(histograms: &mut [Histogram], src: usize, dst: usize, p: &HistogramPair) {
    let (s, d) = pair(histograms, src, dst);
    histogram_add_into(s, d);
    d.update_cost(p.cost_combo, &p.costs);
}

fn combine_greedy(histograms: &mut Vec<Histogram>) {
    let size = histograms.len();
    let mut q = HistoQueue::new(size * size);
    for i in 0..size {
        for j in i + 1..size {
            q.push(histograms, i, j, 0);
        }
    }
    while !q.queue.is_empty() {
        let head = q.queue[0];
        let (idx1, idx2) = (head.idx1, head.idx2);
        merge_into(histograms, idx2 as usize, idx1 as usize, &head);
        histograms.swap_remove(idx2 as usize);
        let new_size = histograms.len() as i32;
        let mut i = 0;
        while i < q.queue.len() {
            let p = q.queue[i];
            if p.idx1 == idx1 || p.idx2 == idx1 || p.idx1 == idx2 || p.idx2 == idx2 {
                q.pop_pair(i);
            } else {
                fix_pair(new_size, idx2, &mut q.queue[i]);
                q.update_head(i);
                i += 1;
            }
        }
        for i in 0..histograms.len() {
            if i as i32 == idx1 {
                continue;
            }
            q.push(histograms, idx1 as usize, i, 0);
        }
    }
}

/// Returns `do_greedy`.
fn combine_stochastic(histograms: &mut Vec<Histogram>, min_cluster_size: usize) -> bool {
    let mut seed = 1u32;
    let mut tries_with_no_success = 0;
    let outer_iters = histograms.len();
    let num_tries_no_success = outer_iters / 2;
    if histograms.len() < min_cluster_size {
        return true;
    }
    let mut q = HistoQueue::new(9);
    let mut iter = 0;
    while iter < outer_iters && histograms.len() >= min_cluster_size && {
        tries_with_no_success += 1;
        tries_with_no_success < num_tries_no_success
    } {
        let mut best_cost = if q.queue.is_empty() {
            0
        } else {
            q.queue[0].cost_diff
        };
        let size = histograms.len() as u32;
        let rand_range = (size.wrapping_sub(1)).wrapping_mul(size);
        let num_tries = size / 2;
        let mut j = 0;
        while size >= 2 && j < num_tries {
            let tmp = my_rand(&mut seed) % rand_range;
            let idx1 = tmp / (size - 1);
            let mut idx2 = tmp % (size - 1);
            if idx2 >= idx1 {
                idx2 += 1;
            }
            let curr_cost = q.push(histograms, idx1 as usize, idx2 as usize, best_cost);
            if curr_cost < 0 {
                best_cost = curr_cost;
                if q.queue.len() == q.max_size {
                    break;
                }
            }
            j += 1;
        }
        iter += 1;
        if q.queue.is_empty() {
            continue;
        }
        let head = q.queue[0];
        let (best_idx1, best_idx2) = (head.idx1, head.idx2);
        merge_into(histograms, best_idx2 as usize, best_idx1 as usize, &head);
        histograms.swap_remove(best_idx2 as usize);
        let new_size = histograms.len() as i32;
        let mut j = 0;
        while j < q.queue.len() {
            let p = q.queue[j];
            let is_idx1_best = p.idx1 == best_idx1 || p.idx1 == best_idx2;
            let is_idx2_best = p.idx2 == best_idx1 || p.idx2 == best_idx2;
            if is_idx1_best && is_idx2_best {
                q.pop_pair(j);
                continue;
            }
            if is_idx1_best || is_idx2_best {
                fix_pair(best_idx2, best_idx1, &mut q.queue[j]);
                let p = q.queue[j];
                let mut np = p;
                // The removed slot still holds the histogram moved to best_idx2.
                let slot = |i: i32| {
                    if i == new_size {
                        best_idx2 as usize
                    } else {
                        i as usize
                    }
                };
                if !update_pair(
                    &histograms[slot(p.idx1)],
                    &histograms[slot(p.idx2)],
                    0,
                    &mut np,
                ) {
                    q.pop_pair(j);
                    continue;
                }
                q.queue[j] = np;
            }
            fix_pair(new_size, best_idx2, &mut q.queue[j]);
            q.update_head(j);
            j += 1;
        }
        tries_with_no_success = 0;
    }
    histograms.len() <= min_cluster_size
}

/// `HistogramRemap`: the best cluster for each tile, and the clusters rebuilt from them.
fn remap(orig: &[Histogram], orig_used: &[bool], out: &mut [Histogram], symbols: &mut [u32]) {
    let out_size = out.len();
    if out_size > 1 {
        for i in 0..orig.len() {
            if !orig_used[i] {
                symbols[i] = symbols[i - 1];
                continue;
            }
            let mut best_out = 0;
            let mut best_bits = i64::MAX;
            for (k, o) in out.iter().enumerate() {
                if let Some(cur_bits) = histogram_add_thresh(o, &orig[i], best_bits) {
                    best_bits = cur_bits;
                    best_out = k;
                }
            }
            symbols[i] = best_out as u32;
        }
    } else {
        symbols[..orig.len()].fill(0);
    }
    for o in out.iter_mut() {
        o.zero();
    }
    for i in 0..orig.len() {
        if !orig_used[i] {
            continue;
        }
        histogram_add_into(&orig[i], &mut out[symbols[i] as usize]);
    }
}

fn get_combine_cost_factor(histo_size: usize, quality: i32) -> i32 {
    let mut f = 16;
    if quality < 90 {
        if histo_size > 256 {
            f /= 2;
        }
        if histo_size > 512 {
            f /= 2;
        }
        if histo_size > 1024 {
            f /= 2;
        }
        if quality <= 50 {
            f /= 2;
        }
    }
    f
}

/// `VP8LGetHistoImageSymbols`: the clustered histograms, with each tile's cluster in
/// `histogram_symbols`.
pub(crate) fn get_histo_image_symbols(
    xsize: u32,
    ysize: u32,
    refs: &[PixOrCopy],
    quality: i32,
    low_effort: bool,
    histogram_bits: u32,
    cache_bits: u32,
    histogram_symbols: &mut [u32],
) -> Vec<Histogram> {
    let histo_xsize = if histogram_bits > 0 {
        sub_sample_size(xsize, histogram_bits)
    } else {
        1
    };
    let histo_ysize = if histogram_bits > 0 {
        sub_sample_size(ysize, histogram_bits)
    } else {
        1
    };
    let raw_size = (histo_xsize * histo_ysize) as usize;
    let mut orig = vec![Histogram::new(cache_bits); raw_size];
    for h in orig.iter_mut() {
        h.zero();
    }
    {
        let (mut x, mut y) = (0u32, 0u32);
        for v in refs {
            let ix = ((y >> histogram_bits) * histo_xsize + (x >> histogram_bits)) as usize;
            orig[ix].add_single(v, None);
            x += u32::from(v.len);
            while x >= xsize {
                x -= xsize;
                y += 1;
            }
        }
    }
    let mut orig_used = vec![true; raw_size];
    let mut image: Vec<Histogram> = Vec::new();
    for (i, h) in orig.iter_mut().enumerate() {
        h.compute_cost();
        if h.is_empty() {
            orig_used[i] = false;
        } else {
            image.push(h.clone());
        }
    }
    let entropy_combine_num_bins = if low_effort { NUM_PARTITIONS } else { BIN_SIZE };
    let entropy_combine = image.len() > entropy_combine_num_bins * 2 && quality < 100;
    if entropy_combine {
        let combine_cost_factor = get_combine_cost_factor(raw_size, quality);
        analyze_entropy_bin(&mut image, low_effort);
        let mut tmp = Histogram::new(cache_bits);
        combine_entropy_bin(
            &mut image,
            &mut tmp,
            entropy_combine_num_bins,
            combine_cost_factor,
            low_effort,
        );
    }
    if !low_effort || !entropy_combine {
        let q = i64::from(quality);
        let threshold_size =
            (1 + div_round(q * q * q * (MAX_HISTO_GREEDY - 1), 100 * 100 * 100)) as usize;
        if combine_stochastic(&mut image, threshold_size) {
            combine_greedy(&mut image);
        }
    }
    remap(&orig, &orig_used, &mut image, histogram_symbols);
    image
}
