//! Cost-based backward references: a shortest path over literal and copy costs (libwebp's
//! `enc/backward_references_cost_enc.c`).

use super::backward_refs::{
    distance_to_plane_code, BackwardRefs, ColorCache, HashChain, PixOrCopy, MAX_LENGTH,
    MAX_LENGTH_BITS,
};
use super::entropy::{fast_log2, prefix_encode_bits, LOG_2_PRECISION_BITS};
use super::histogram::{histogram_num_codes, Histogram};
use super::{MAX_COLOR_CACHE_BITS, NUM_DISTANCE_CODES, NUM_LENGTH_CODES};

struct CostModel {
    alpha: [u32; 256],
    red: [u32; 256],
    blue: [u32; 256],
    distance: [u32; NUM_DISTANCE_CODES],
    literal: Vec<u32>,
    /// `DivRound(cache_cost * 68, 100)` per cache index.
    cache_hit: [i64; 1 << MAX_COLOR_CACHE_BITS],
}

fn population_to_bit_estimates(population: &[u32], output: &mut [u32]) {
    let mut sum = 0u32;
    let mut nonzeros = 0;
    for &p in population {
        sum = sum.wrapping_add(p);
        if p > 0 {
            nonzeros += 1;
        }
    }
    if nonzeros <= 1 {
        output.fill(0);
    } else {
        let logsum = fast_log2(sum);
        for (o, &p) in output.iter_mut().zip(population) {
            *o = logsum.wrapping_sub(fast_log2(p));
        }
    }
}

impl CostModel {
    fn build(xsize: u32, cache_bits: u32, refs: &BackwardRefs) -> Self {
        let mut histo = Histogram::new(cache_bits);
        histo.store_refs(refs, Some(xsize));
        let mut m = CostModel {
            alpha: [0; 256],
            red: [0; 256],
            blue: [0; 256],
            distance: [0; NUM_DISTANCE_CODES],
            literal: vec![0; histogram_num_codes(cache_bits)],
            cache_hit: [0; 1 << MAX_COLOR_CACHE_BITS],
        };
        population_to_bit_estimates(&histo.literal, &mut m.literal);
        population_to_bit_estimates(&histo.red, &mut m.red);
        population_to_bit_estimates(&histo.blue, &mut m.blue);
        population_to_bit_estimates(&histo.alpha, &mut m.alpha);
        population_to_bit_estimates(&histo.distance, &mut m.distance);
        for ix in 0..m.literal.len() - 256 - NUM_LENGTH_CODES {
            m.cache_hit[ix] = (m.cache_cost(ix as u32) * 68 + 50) / 100;
        }
        m
    }

    #[inline]
    fn literal_cost(&self, v: u32) -> i64 {
        i64::from(self.alpha[(v >> 24) as usize])
            + i64::from(self.red[((v >> 16) & 0xff) as usize])
            + i64::from(self.literal[((v >> 8) & 0xff) as usize])
            + i64::from(self.blue[(v & 0xff) as usize])
    }

    #[inline]
    fn cache_cost(&self, idx: u32) -> i64 {
        i64::from(self.literal[256 + NUM_LENGTH_CODES + idx as usize])
    }

    #[inline]
    fn length_cost(&self, length: u32) -> i64 {
        let (code, extra_bits) = prefix_encode_bits(length);
        i64::from(self.literal[256 + code as usize])
            + (i64::from(extra_bits) << LOG_2_PRECISION_BITS)
    }

    #[inline]
    fn distance_cost(&self, distance: u32) -> i64 {
        let (code, extra_bits) = prefix_encode_bits(distance);
        i64::from(self.distance[code as usize]) + (i64::from(extra_bits) << LOG_2_PRECISION_BITS)
    }
}

/// The cost of coding `color` as a literal or cache index, updating the cache as libwebp's
/// `AddSingleLiteralWithCostModel`. The costs are nonnegative, so `DivRound` is a
/// rounded-up-at-half division.
#[inline(always)]
fn single_literal_cost(color: u32, cache: &mut Option<Box<ColorCache>>, m: &CostModel) -> i64 {
    let ix = cache.as_ref().and_then(|cc| cc.contains(color));
    if let Some(ix) = ix {
        m.cache_hit[ix as usize & ((1 << MAX_COLOR_CACHE_BITS) - 1)]
    } else {
        if let Some(cc) = cache {
            cc.insert(color);
        }
        (m.literal_cost(color) * 82 + 50) / 100
    }
}

const COST_CACHE_INTERVAL_SIZE_MAX: i32 = 500;

/// The costs of pixels `i - 1..i + 2 * MAX_LENGTH` while pixel `i` is processed, the only
/// ones read or written then, in a ring instead of one cost per pixel of the image.
struct CostWindow(Box<[i64; COST_WINDOW]>);

const COST_WINDOW: usize = 1 << 14;
const _: () = assert!(COST_WINDOW > 2 * MAX_LENGTH as usize + 1);

impl core::ops::Index<usize> for CostWindow {
    type Output = i64;
    #[inline(always)]
    fn index(&self, i: usize) -> &i64 {
        &self.0[i & (COST_WINDOW - 1)]
    }
}

impl core::ops::IndexMut<usize> for CostWindow {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut i64 {
        &mut self.0[i & (COST_WINDOW - 1)]
    }
}

/// Interval links are indices into `CostManager::nodes`, `NIL` for none. At most
/// `COST_CACHE_INTERVAL_SIZE_MAX` intervals are live, so `NODES` slots always suffice and an
/// index masked by `NODES - 1` needs no bounds check.
const NIL: u32 = u32::MAX;
const NODES: usize = 512;

#[derive(Clone, Copy)]
struct CostInterval {
    cost: i64,
    start: i32,
    end: i32,
    index: i32,
    previous: u32,
    next: u32,
}

#[derive(Clone, Copy)]
struct Run {
    end: i32,
    cost: i64,
    index: i32,
}

const EMPTY_INTERVAL: CostInterval = CostInterval {
    cost: 0,
    start: 0,
    end: 0,
    index: 0,
    previous: NIL,
    next: NIL,
};

#[derive(Clone, Copy)]
struct CostCacheInterval {
    cost: i64,
    start: i32,
    end: i32,
}

struct CostManager {
    head: u32,
    count: i32,
    cache_intervals: Vec<CostCacheInterval>,
    cost_cache: Vec<i64>,
    costs: CostWindow,
    nodes: Box<[CostInterval; NODES]>,
    /// Head of the free list, linked through `next`.
    free: u32,
    allocated: u32,
}

impl CostManager {
    fn new(pix_count: usize, m: &CostModel) -> Self {
        let cost_cache_size = pix_count.min(MAX_LENGTH as usize);
        let cost_cache: Vec<i64> = (0..cost_cache_size)
            .map(|i| m.length_cost(i as u32))
            .collect();
        let mut cache_intervals = Vec::new();
        let mut cur = CostCacheInterval {
            cost: cost_cache[0],
            start: 0,
            end: 1,
        };
        for (i, &cost_val) in cost_cache.iter().enumerate().skip(1) {
            if cost_val != cur.cost {
                cache_intervals.push(cur);
                cur = CostCacheInterval {
                    cost: cost_val,
                    start: i as i32,
                    end: 0,
                };
            }
            cur.end = i as i32 + 1;
        }
        cache_intervals.push(cur);
        Self {
            head: NIL,
            count: 0,
            cache_intervals,
            cost_cache,
            costs: CostWindow(
                vec![i64::MAX; COST_WINDOW]
                    .into_boxed_slice()
                    .try_into()
                    .unwrap(),
            ),
            nodes: Box::new([EMPTY_INTERVAL; NODES]),
            free: NIL,
            allocated: 0,
        }
    }

    #[inline(always)]
    fn node(&self, i: u32) -> &CostInterval {
        &self.nodes[i as usize & (NODES - 1)]
    }

    #[inline(always)]
    fn node_mut(&mut self, i: u32) -> &mut CostInterval {
        &mut self.nodes[i as usize & (NODES - 1)]
    }

    #[inline]
    fn update_cost(&mut self, dist_array: &mut [u16], i: i32, position: i32, cost: i64) {
        let k = i - position;
        if self.costs[i as usize] > cost {
            self.costs[i as usize] = cost;
            dist_array[i as usize] = (k + 1) as u16;
        }
    }

    fn update_cost_per_interval(
        &mut self,
        dist_array: &mut [u16],
        start: i32,
        end: i32,
        position: i32,
        cost: i64,
    ) {
        for i in start..end {
            self.update_cost(dist_array, i, position, cost);
        }
    }

    #[inline(always)]
    fn connect(&mut self, prev: u32, next: u32) {
        if prev != NIL {
            self.node_mut(prev).next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.node_mut(next).previous = prev;
        }
    }

    #[inline(always)]
    fn pop_interval(&mut self, interval: u32) {
        let CostInterval { previous, next, .. } = *self.node(interval);
        self.connect(previous, next);
        self.node_mut(interval).next = self.free;
        self.free = interval;
        self.count -= 1;
    }

    #[inline]
    fn update_cost_at_index(&mut self, dist_array: &mut [u16], i: i32) {
        let mut current = self.head;
        while current != NIL {
            let c = *self.node(current);
            if c.start > i {
                break;
            }
            if c.end > i {
                self.update_cost(dist_array, i, c.index, c.cost);
            }
            current = c.next;
        }
    }

    /// `UpdateCostAtIndex` with interval cleaning. Until the returned run's `end` no interval
    /// starts or ends, so the live ones reduce to the first of least cost.
    #[inline]
    fn update_cost_at_index_clean(&mut self, dist_array: &mut [u16], i: i32) -> Run {
        let mut run = Run {
            end: i32::MAX,
            cost: i64::MAX,
            index: 0,
        };
        let mut current = self.head;
        while current != NIL {
            let c = *self.node(current);
            if c.start > i {
                run.end = run.end.min(c.start);
                break;
            }
            if c.end <= i {
                self.pop_interval(current);
            } else {
                self.update_cost(dist_array, i, c.index, c.cost);
                run.end = run.end.min(c.end);
                if c.cost < run.cost {
                    run.cost = c.cost;
                    run.index = c.index;
                }
            }
            current = c.next;
        }
        run
    }

    fn position_orphan_interval(&mut self, current: u32, previous: u32) {
        let mut previous = if previous == NIL { self.head } else { previous };
        let cstart = self.node(current).start;
        while previous != NIL && cstart < self.node(previous).start {
            previous = self.node(previous).previous;
        }
        while previous != NIL {
            let n = self.node(previous).next;
            if n != NIL && self.node(n).start < cstart {
                previous = n;
            } else {
                break;
            }
        }
        let next = if previous != NIL {
            self.node(previous).next
        } else {
            self.head
        };
        self.connect(current, next);
        self.connect(previous, current);
    }

    fn insert_interval(
        &mut self,
        dist_array: &mut [u16],
        interval_in: u32,
        cost: i64,
        position: i32,
        start: i32,
        end: i32,
    ) {
        if start >= end {
            return;
        }
        if self.count >= COST_CACHE_INTERVAL_SIZE_MAX {
            self.update_cost_per_interval(dist_array, start, end, position, cost);
            return;
        }
        let idx = if self.free != NIL {
            let i = self.free;
            self.free = self.node(i).next;
            i
        } else {
            self.allocated += 1;
            self.allocated - 1
        };
        *self.node_mut(idx) = CostInterval {
            cost,
            start,
            end,
            index: position,
            previous: NIL,
            next: NIL,
        };
        self.position_orphan_interval(idx, interval_in);
        self.count += 1;
    }

    fn push_interval(
        &mut self,
        dist_array: &mut [u16],
        distance_cost: i64,
        position: i32,
        len: i32,
    ) {
        const SKIP_DISTANCE: i32 = 10;
        if len < SKIP_DISTANCE {
            for j in position..position + len {
                let k = j - position;
                let cost_tmp = distance_cost + self.cost_cache[k as usize];
                if self.costs[j as usize] > cost_tmp {
                    self.costs[j as usize] = cost_tmp;
                    dist_array[j as usize] = (k + 1) as u16;
                }
            }
            return;
        }
        let mut interval = self.head;
        let mut i = 0;
        while i < self.cache_intervals.len() && self.cache_intervals[i].start < len {
            let ci = self.cache_intervals[i];
            let mut start = position + ci.start;
            let end = position + if ci.end > len { len } else { ci.end };
            let cost = distance_cost + ci.cost;
            while interval != NIL {
                let iv = interval;
                let n = *self.node(iv);
                if n.start >= end {
                    break;
                }
                let interval_next = n.next;
                if start >= n.end {
                    interval = interval_next;
                    continue;
                }
                if cost >= n.cost {
                    let start_new = n.end;
                    self.insert_interval(dist_array, iv, cost, position, start, n.start);
                    start = start_new;
                    if start >= end {
                        break;
                    }
                    interval = interval_next;
                    continue;
                }
                if start <= n.start {
                    if n.end <= end {
                        self.pop_interval(iv);
                    } else {
                        self.node_mut(iv).start = end;
                        break;
                    }
                } else if end < n.end {
                    self.node_mut(iv).end = start;
                    self.insert_interval(dist_array, iv, n.cost, n.index, end, n.end);
                    interval = self.node(iv).next;
                    break;
                } else {
                    self.node_mut(iv).end = start;
                }
                interval = interval_next;
            }
            self.insert_interval(dist_array, interval, cost, position, start, end);
            i += 1;
        }
    }
}

fn distance_only(
    xsize: u32,
    ysize: u32,
    argb: &[u32],
    cache_bits: u32,
    hash_chain: &HashChain,
    refs: &BackwardRefs,
    dist_array: &mut [u16],
) {
    let pix_count = (xsize * ysize) as usize;
    let argb = &argb[..pix_count];
    let dist_array = &mut dist_array[..pix_count];
    let offset_length = &hash_chain.offset_length[..pix_count];
    let m = CostModel::build(xsize, cache_bits, refs);
    let mut cache = (cache_bits > 0).then(|| ColorCache::new(cache_bits));
    let mut cm = CostManager::new(pix_count, &m);
    let mut offset_prev: i64 = -1;
    let mut len_prev: i32 = -1;
    let mut offset_cost: i64 = -1;
    let mut first_offset_is_constant = -1;
    let mut reach: i32 = 0;
    let mut run = Run {
        end: 0,
        cost: i64::MAX,
        index: 0,
    };

    let mut prev_cost = single_literal_cost(argb[0], &mut cache, &m);
    cm.costs[0] = prev_cost;
    dist_array[0] = 1;
    for i in 1..pix_count {
        let offset = offset_length[i] >> MAX_LENGTH_BITS;
        let len = (offset_length[i] & ((1 << MAX_LENGTH_BITS) - 1)) as i32;
        let mut cost = cm.costs[i];
        let literal = prev_cost + single_literal_cost(argb[i], &mut cache, &m);
        if cost > literal {
            cost = literal;
            dist_array[i] = 1;
        }
        if len >= 2 {
            if i64::from(offset) != offset_prev {
                let code = distance_to_plane_code(xsize, offset);
                offset_cost = m.distance_cost(code);
                first_offset_is_constant = 1;
                cm.costs[i] = cost;
                cm.push_interval(dist_array, prev_cost + offset_cost, i as i32, len);
                cost = cm.costs[i];
                run.end = 0;
            } else {
                if first_offset_is_constant != 0 {
                    reach = i as i32 - 1 + len_prev - 1;
                    first_offset_is_constant = 0;
                }
                if i as i32 + len - 1 > reach {
                    let mut len_j = 0i32;
                    let mut j = i as i32;
                    while j <= reach {
                        let offset_j = hash_chain.offset(j as usize + 1);
                        len_j = hash_chain.length(j as usize + 1) as i32;
                        if offset_j != offset {
                            len_j = hash_chain.length(j as usize) as i32;
                            break;
                        }
                        j += 1;
                    }
                    cm.costs[i] = cost;
                    cm.update_cost_at_index(dist_array, j - 1);
                    cm.update_cost_at_index(dist_array, j);
                    let c = cm.costs[(j - 1) as usize] + offset_cost;
                    cm.push_interval(dist_array, c, j, len_j);
                    cost = cm.costs[i];
                    run.end = 0;
                    reach = j + len_j - 1;
                }
            }
        }
        if (i as i32) < run.end {
            if cost > run.cost {
                cost = run.cost;
                dist_array[i] = (i as i32 - run.index + 1) as u16;
            }
        } else {
            cm.costs[i] = cost;
            run = cm.update_cost_at_index_clean(dist_array, i as i32);
            cost = cm.costs[i];
        }
        cm.costs[i] = cost;
        cm.costs[i - 1] = i64::MAX;
        prev_cost = cost;
        offset_prev = i64::from(offset);
        len_prev = len;
    }
}

/// `VP8LBackwardReferencesTraceBackwards`: the cheapest path through `hash_chain`'s
/// matches under the cost model of `refs_src`, into `refs_dst`.
pub(crate) fn trace_backwards(
    xsize: u32,
    ysize: u32,
    argb: &[u32],
    cache_bits: u32,
    hash_chain: &HashChain,
    refs_src: &BackwardRefs,
    refs_dst: &mut BackwardRefs,
) {
    let size = (xsize * ysize) as usize;
    let mut dist_array = vec![0u16; size];
    distance_only(
        xsize,
        ysize,
        argb,
        cache_bits,
        hash_chain,
        refs_src,
        &mut dist_array,
    );
    let mut path_start = size;
    let mut cur = size as isize - 1;
    while cur >= 0 {
        let k = dist_array[cur as usize];
        path_start -= 1;
        dist_array[path_start] = k;
        cur -= k as isize;
    }
    let mut cache = (cache_bits > 0).then(|| ColorCache::new(cache_bits));
    refs_dst.clear();
    let mut i = 0usize;
    for &len in &dist_array[path_start..] {
        let len = len as usize;
        if len != 1 {
            let offset = hash_chain.offset(i);
            refs_dst.push(PixOrCopy::copy(offset, len as u32));
            if let Some(cc) = &mut cache {
                for &p in &argb[i..i + len] {
                    cc.insert(p);
                }
            }
            i += len;
        } else {
            let idx = cache.as_ref().and_then(|cc| cc.contains(argb[i]));
            let v = match idx {
                Some(idx) => PixOrCopy::cache_idx(idx),
                None => {
                    if let Some(cc) = &mut cache {
                        cc.insert(argb[i]);
                    }
                    PixOrCopy::literal(argb[i])
                }
            };
            refs_dst.push(v);
            i += 1;
        }
    }
}
