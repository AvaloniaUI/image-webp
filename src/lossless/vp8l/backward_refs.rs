//! LZ77 hash chains, backward references (standard, RLE and box LZ77) and colour cache
//! sizing (libwebp's `enc/backward_references_enc.c`).

use super::cost::trace_backwards;
use super::entropy::prefix_encode;
use super::histogram::Histogram;
use super::tables::PLANE_TO_CODE_LUT;
use super::{MAX_COLOR_CACHE_BITS, NUM_LENGTH_CODES, NUM_LITERAL_CODES};

const HASH_BITS: u32 = 18;
const HASH_SIZE: usize = 1 << HASH_BITS;
pub(crate) const MAX_LENGTH_BITS: u32 = 12;
const WINDOW_SIZE_BITS: u32 = 20;
pub(crate) const MAX_LENGTH: u32 = (1 << MAX_LENGTH_BITS) - 1;
const WINDOW_SIZE: u32 = (1 << WINDOW_SIZE_BITS) - 120;
pub(crate) const MIN_LENGTH: u32 = 4;

pub(crate) const LZ77_STANDARD: u32 = 1;
pub(crate) const LZ77_RLE: u32 = 2;
pub(crate) const LZ77_BOX: u32 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PixOrCopyMode {
    Literal,
    CacheIdx,
    Copy,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PixOrCopy {
    pub(crate) mode: PixOrCopyMode,
    pub(crate) len: u16,
    pub(crate) argb_or_distance: u32,
}

impl PixOrCopy {
    #[inline]
    pub(crate) fn copy(distance: u32, len: u32) -> Self {
        Self {
            mode: PixOrCopyMode::Copy,
            len: len as u16,
            argb_or_distance: distance,
        }
    }
    #[inline]
    pub(crate) fn cache_idx(idx: u32) -> Self {
        Self {
            mode: PixOrCopyMode::CacheIdx,
            len: 1,
            argb_or_distance: idx,
        }
    }
    #[inline]
    pub(crate) fn literal(argb: u32) -> Self {
        Self {
            mode: PixOrCopyMode::Literal,
            len: 1,
            argb_or_distance: argb,
        }
    }
}

pub(crate) type BackwardRefs = Vec<PixOrCopy>;

pub(crate) fn distance_to_plane_code(xsize: u32, dist: u32) -> u32 {
    let (xsize, dist) = (xsize as i32, dist as i32);
    let yoffset = dist / xsize;
    let xoffset = dist - yoffset * xsize;
    if xoffset <= 8 && yoffset < 8 {
        u32::from(PLANE_TO_CODE_LUT[(yoffset * 16 + 8 - xoffset) as usize]) + 1
    } else if xoffset > xsize - 8 && yoffset < 7 {
        u32::from(PLANE_TO_CODE_LUT[((yoffset + 1) * 16 + 8 + (xsize - xoffset)) as usize]) + 1
    } else {
        (dist + 120) as u32
    }
}

#[inline]
fn vector_mismatch(a: &[u32], b: &[u32], length: usize) -> usize {
    let (a, b) = (&a[..length], &b[..length]);
    let mut i = 0;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if <[u32; 4]>::try_from(x).unwrap() != <[u32; 4]>::try_from(y).unwrap() {
            break;
        }
        i += 4;
    }
    while i < length && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Mismatch index of `argb[p1..]` and `argb[p2..]`, 0 if they differ at `best_len_match`.
#[inline]
fn find_match_length(
    argb: &[u32],
    p1: usize,
    p2: usize,
    best_len_match: usize,
    max_limit: usize,
) -> usize {
    if argb[p1 + best_len_match] != argb[p2 + best_len_match] {
        return 0;
    }
    vector_mismatch(&argb[p1..], &argb[p2..], max_limit)
}

const COLOR_CACHE_MASK: usize = (1 << MAX_COLOR_CACHE_BITS) - 1;

#[derive(Clone, Copy)]
pub(crate) struct ColorCache {
    colors: [u32; 1 << MAX_COLOR_CACHE_BITS],
    hash_shift: u32,
}

impl ColorCache {
    pub(crate) fn new(hash_bits: u32) -> Box<Self> {
        Box::new(Self {
            colors: [0; 1 << MAX_COLOR_CACHE_BITS],
            hash_shift: 32 - hash_bits,
        })
    }
    #[inline]
    pub(crate) fn index(&self, argb: u32) -> u32 {
        argb.wrapping_mul(0x1e35a7bd) >> self.hash_shift
    }
    #[inline]
    pub(crate) fn lookup(&self, key: u32) -> u32 {
        self.colors[key as usize & COLOR_CACHE_MASK]
    }
    #[inline]
    pub(crate) fn set(&mut self, key: u32, argb: u32) {
        self.colors[key as usize & COLOR_CACHE_MASK] = argb;
    }
    #[inline]
    pub(crate) fn insert(&mut self, argb: u32) {
        self.set(self.index(argb), argb);
    }
    /// The key if the cache holds `argb`.
    #[inline]
    pub(crate) fn contains(&self, argb: u32) -> Option<u32> {
        let key = self.index(argb);
        (self.lookup(key) == argb).then_some(key)
    }
}

#[inline]
pub(crate) fn hash_pix(argb: u32, shift: u32) -> u32 {
    argb.wrapping_mul(0x1e35a7bd) >> shift
}

/// Best match (offset << 12 | length) at every pixel.
#[derive(Default)]
pub(crate) struct HashChain {
    pub(crate) offset_length: Vec<u32>,
}

impl HashChain {
    #[inline]
    pub(crate) fn offset(&self, pos: usize) -> u32 {
        self.offset_length[pos] >> MAX_LENGTH_BITS
    }
    #[inline]
    pub(crate) fn length(&self, pos: usize) -> u32 {
        self.offset_length[pos] & ((1 << MAX_LENGTH_BITS) - 1)
    }

    pub(crate) fn fill(
        &mut self,
        quality: i32,
        argb: &[u32],
        xsize: u32,
        ysize: u32,
        low_effort: bool,
    ) {
        let size = (xsize * ysize) as usize;
        self.offset_length.clear();
        self.offset_length.resize(size, 0);
        let iter_max = get_max_iters_for_quality(quality);
        let window_size = get_window_size_for_hash_chain(quality, xsize);
        if size <= 2 {
            self.offset_length[0] = 0;
            self.offset_length[size - 1] = 0;
            return;
        }
        let chain = &mut self.offset_length;
        let mut hash_to_first_index = vec![-1i32; HASH_SIZE];
        let mut argb_comp = argb[0] == argb[1];
        let mut pos = 0usize;
        while pos < size - 2 {
            let argb_comp_next = argb[pos + 1] == argb[pos + 2];
            if argb_comp && argb_comp_next {
                let mut tmp = [argb[pos], 0];
                let mut len = 1u32;
                while pos + len as usize + 2 < size && argb[pos + len as usize + 2] == argb[pos] {
                    len += 1;
                }
                if len > MAX_LENGTH {
                    let skip = (len - MAX_LENGTH) as usize;
                    chain[pos..pos + skip].fill(u32::MAX);
                    pos += skip;
                    len = MAX_LENGTH;
                }
                while len > 0 {
                    tmp[1] = len;
                    len -= 1;
                    let hash_code = get_pix_pair_hash64(&tmp) as usize;
                    chain[pos] = hash_to_first_index[hash_code] as u32;
                    hash_to_first_index[hash_code] = pos as i32;
                    pos += 1;
                }
                argb_comp = false;
            } else {
                let hash_code = get_pix_pair_hash64(&argb[pos..pos + 2]) as usize;
                chain[pos] = hash_to_first_index[hash_code] as u32;
                hash_to_first_index[hash_code] = pos as i32;
                pos += 1;
                argb_comp = argb_comp_next;
            }
        }
        chain[pos] = hash_to_first_index[get_pix_pair_hash64(&argb[pos..pos + 2]) as usize] as u32;
        drop(hash_to_first_index);

        let ol = &mut self.offset_length;
        ol[0] = 0;
        ol[size - 1] = 0;
        let mut base_position = size as u32 - 2;
        while base_position > 0 {
            let max_len = max_find_copy_length(size as u32 - 1 - base_position) as usize;
            let bp = base_position as usize;
            let mut iter = iter_max;
            let mut best_length = 0usize;
            let mut best_distance = 0u32;
            let min_pos: i32 = if base_position > window_size {
                (base_position - window_size) as i32
            } else {
                0
            };
            let length_max = max_len.min(256);
            let mut pos = ol[bp] as i32;
            if !low_effort {
                if base_position >= xsize {
                    let curr_length =
                        find_match_length(argb, bp - xsize as usize, bp, best_length, max_len);
                    if curr_length > best_length {
                        best_length = curr_length;
                        best_distance = xsize;
                    }
                    iter -= 1;
                }
                let curr_length = find_match_length(argb, bp - 1, bp, best_length, max_len);
                if curr_length > best_length {
                    best_length = curr_length;
                    best_distance = 1;
                }
                iter -= 1;
                if best_length == MAX_LENGTH as usize {
                    pos = min_pos - 1;
                }
            }
            let mut best_argb = argb[bp + best_length];
            while pos >= min_pos && {
                iter -= 1;
                iter != 0
            } {
                let p = pos as usize;
                if argb[p + best_length] == best_argb {
                    let curr_length = vector_mismatch(&argb[p..], &argb[bp..], max_len);
                    if best_length < curr_length {
                        best_length = curr_length;
                        best_distance = base_position - pos as u32;
                        best_argb = argb[bp + best_length];
                        if best_length >= length_max {
                            break;
                        }
                    }
                }
                pos = ol[p] as i32;
            }
            let mut max_base_position = base_position;
            loop {
                ol[base_position as usize] =
                    (best_distance << MAX_LENGTH_BITS) | best_length as u32;
                base_position -= 1;
                if best_distance == 0 || base_position == 0 {
                    break;
                }
                if base_position < best_distance
                    || argb[(base_position - best_distance) as usize]
                        != argb[base_position as usize]
                {
                    break;
                }
                if best_length == MAX_LENGTH as usize
                    && best_distance != 1
                    && base_position + MAX_LENGTH < max_base_position
                {
                    break;
                }
                if best_length < MAX_LENGTH as usize {
                    best_length += 1;
                    max_base_position = base_position;
                }
            }
        }
    }
}

#[inline]
fn get_pix_pair_hash64(argb: &[u32]) -> u32 {
    let key = argb[1]
        .wrapping_mul(0xc6a4a793)
        .wrapping_add(argb[0].wrapping_mul(0x5bd1e996));
    key >> (32 - HASH_BITS)
}

fn get_max_iters_for_quality(quality: i32) -> i32 {
    8 + (quality * quality) / 128
}

fn get_window_size_for_hash_chain(quality: i32, xsize: u32) -> u32 {
    let max_window_size = if quality > 75 {
        WINDOW_SIZE
    } else if quality > 50 {
        xsize << 8
    } else if quality > 25 {
        xsize << 6
    } else {
        xsize << 4
    };
    max_window_size.min(WINDOW_SIZE)
}

#[inline]
fn max_find_copy_length(len: u32) -> u32 {
    len.min(MAX_LENGTH)
}

#[inline]
fn add_single_literal(pixel: u32, cache: &mut Option<Box<ColorCache>>, refs: &mut BackwardRefs) {
    let v = match cache {
        Some(cc) => {
            let key = cc.index(pixel);
            if cc.lookup(key) == pixel {
                PixOrCopy::cache_idx(key)
            } else {
                cc.set(key, pixel);
                PixOrCopy::literal(pixel)
            }
        }
        None => PixOrCopy::literal(pixel),
    };
    refs.push(v);
}

fn new_cache(cache_bits: u32) -> Option<Box<ColorCache>> {
    (cache_bits > 0).then(|| ColorCache::new(cache_bits))
}

fn backward_references_rle(
    xsize: u32,
    ysize: u32,
    argb: &[u32],
    cache_bits: u32,
    refs: &mut BackwardRefs,
) {
    let pix_count = (xsize * ysize) as usize;
    let xsize = xsize as usize;
    let mut cache = new_cache(cache_bits);
    refs.clear();
    add_single_literal(argb[0], &mut cache, refs);
    let mut i = 1;
    while i < pix_count {
        let max_len = max_find_copy_length((pix_count - i) as u32) as usize;
        let rle_len = find_match_length(argb, i, i - 1, 0, max_len);
        let prev_row_len = if i < xsize {
            0
        } else {
            find_match_length(argb, i, i - xsize, 0, max_len)
        };
        if rle_len >= prev_row_len && rle_len >= MIN_LENGTH as usize {
            refs.push(PixOrCopy::copy(1, rle_len as u32));
            i += rle_len;
        } else if prev_row_len >= MIN_LENGTH as usize {
            refs.push(PixOrCopy::copy(xsize as u32, prev_row_len as u32));
            if let Some(cc) = &mut cache {
                for k in 0..prev_row_len {
                    cc.insert(argb[i + k]);
                }
            }
            i += prev_row_len;
        } else {
            add_single_literal(argb[i], &mut cache, refs);
            i += 1;
        }
    }
}

pub(crate) fn backward_references_lz77(
    xsize: u32,
    ysize: u32,
    argb: &[u32],
    cache_bits: u32,
    hash_chain: &HashChain,
    refs: &mut BackwardRefs,
) {
    let pix_count = (xsize * ysize) as usize;
    let mut cache = new_cache(cache_bits);
    refs.clear();
    let mut i_last_check: isize = -1;
    let mut i = 0usize;
    while i < pix_count {
        let offset = hash_chain.offset(i);
        let mut len = hash_chain.length(i) as usize;
        if len >= MIN_LENGTH as usize {
            let len_ini = len;
            let mut max_reach = 0usize;
            let j_max = if i + len_ini >= pix_count {
                pix_count - 1
            } else {
                i + len_ini
            };
            i_last_check = i_last_check.max(i as isize);
            let mut j = (i_last_check + 1) as usize;
            while j <= j_max {
                let len_j = hash_chain.length(j) as usize;
                let reach = j + if len_j >= MIN_LENGTH as usize {
                    len_j
                } else {
                    1
                };
                if reach > max_reach {
                    len = j - i;
                    max_reach = reach;
                    if max_reach >= pix_count {
                        break;
                    }
                }
                j += 1;
            }
        } else {
            len = 1;
        }
        if len == 1 {
            add_single_literal(argb[i], &mut cache, refs);
        } else {
            refs.push(PixOrCopy::copy(offset, len as u32));
            if let Some(cc) = &mut cache {
                for &p in &argb[i..i + len] {
                    cc.insert(p);
                }
            }
        }
        i += len;
    }
}

const WINDOW_OFFSETS_SIZE_MAX: usize = 32;

fn backward_references_lz77_box(
    xsize: u32,
    ysize: u32,
    argb: &[u32],
    cache_bits: u32,
    hash_chain_best: &HashChain,
    hash_chain: &mut HashChain,
    refs: &mut BackwardRefs,
) {
    let pix_count = (xsize * ysize) as usize;
    let mut window_offsets = [0i32; WINDOW_OFFSETS_SIZE_MAX];
    let mut window_offsets_new = [0i32; WINDOW_OFFSETS_SIZE_MAX];
    let mut window_offsets_size = 0;
    let mut window_offsets_new_size = 0;
    let mut counts = vec![0u16; pix_count];
    let mut best_offset_prev: i32 = -1;
    let mut best_length_prev: i32 = -1;

    if pix_count >= 1 {
        let last = pix_count - 1;
        // counts[pix_count - 1] = 1, counting down.
        counts[last] = 1;
        for i in (0..last).rev() {
            if argb[i] == argb[i + 1] {
                counts[i] = counts[i + 1] + u16::from(u32::from(counts[i + 1]) != MAX_LENGTH);
            } else {
                counts[i] = 1;
            }
        }
    }

    for y in 0..=6i32 {
        for x in -6..=6i32 {
            let offset = y * xsize as i32 + x;
            if offset <= 0 {
                continue;
            }
            let plane_code = distance_to_plane_code(xsize, offset as u32) as usize - 1;
            if plane_code >= WINDOW_OFFSETS_SIZE_MAX {
                continue;
            }
            window_offsets[plane_code] = offset;
        }
    }
    for i in 0..WINDOW_OFFSETS_SIZE_MAX {
        if window_offsets[i] == 0 {
            continue;
        }
        window_offsets[window_offsets_size] = window_offsets[i];
        window_offsets_size += 1;
    }
    for i in 0..window_offsets_size {
        let mut is_reachable = false;
        let mut j = 0;
        while j < window_offsets_size && !is_reachable {
            is_reachable |= window_offsets[i] == window_offsets[j] + 1;
            j += 1;
        }
        if !is_reachable {
            window_offsets_new[window_offsets_new_size] = window_offsets[i];
            window_offsets_new_size += 1;
        }
    }

    hash_chain.offset_length.clear();
    hash_chain.offset_length.resize(pix_count, 0);
    let ol = &mut hash_chain.offset_length;
    for i in 1..pix_count {
        let mut best_length = hash_chain_best.length(i) as i32;
        let mut best_offset: i32 = 0;
        let mut do_compute = true;
        if best_length >= MAX_LENGTH as i32 {
            best_offset = hash_chain_best.offset(i) as i32;
            for &wo in &window_offsets[..window_offsets_size] {
                if best_offset == wo {
                    do_compute = false;
                    break;
                }
            }
        }
        if do_compute {
            let use_prev = best_length_prev > 1 && best_length_prev < MAX_LENGTH as i32;
            let num_ind = if use_prev {
                window_offsets_new_size
            } else {
                window_offsets_size
            };
            best_length = if use_prev { best_length_prev - 1 } else { 0 };
            best_offset = if use_prev { best_offset_prev } else { 0 };
            for ind in 0..num_ind {
                let wo = if use_prev {
                    window_offsets_new[ind]
                } else {
                    window_offsets[ind]
                };
                let mut curr_length: i32 = 0;
                let mut j = i as i32;
                let mut j_offset = i as i32 - wo;
                if j_offset < 0 || argb[j_offset as usize] != argb[i] {
                    continue;
                }
                loop {
                    let counts_j_offset = i32::from(counts[j_offset as usize]);
                    let counts_j = i32::from(counts[j as usize]);
                    if counts_j_offset != counts_j {
                        curr_length += counts_j_offset.min(counts_j);
                        break;
                    }
                    curr_length += counts_j_offset;
                    j_offset += counts_j_offset;
                    j += counts_j_offset;
                    if !(curr_length <= MAX_LENGTH as i32
                        && (j as usize) < pix_count
                        && argb[j_offset as usize] == argb[j as usize])
                    {
                        break;
                    }
                }
                if best_length < curr_length {
                    best_offset = wo;
                    if curr_length >= MAX_LENGTH as i32 {
                        best_length = MAX_LENGTH as i32;
                        break;
                    } else {
                        best_length = curr_length;
                    }
                }
            }
        }
        if best_length <= MIN_LENGTH as i32 {
            ol[i] = 0;
            best_offset_prev = 0;
            best_length_prev = 0;
        } else {
            ol[i] = ((best_offset as u32) << MAX_LENGTH_BITS) | best_length as u32;
            best_offset_prev = best_offset;
            best_length_prev = best_length;
        }
    }
    if pix_count > 0 {
        ol[0] = 0;
    }
    drop(counts);
    backward_references_lz77(xsize, ysize, argb, cache_bits, hash_chain, refs);
}

fn backward_references_2d_locality(xsize: u32, refs: &mut BackwardRefs) {
    for v in refs.iter_mut() {
        if v.mode == PixOrCopyMode::Copy {
            v.argb_or_distance = distance_to_plane_code(xsize, v.argb_or_distance);
        }
    }
}

/// Lowest-entropy colour cache size up to `max_bits` (0 at quality 25 and below).
fn calculate_best_cache_size(
    argb: &[u32],
    quality: i32,
    refs: &BackwardRefs,
    max_bits: u32,
) -> u32 {
    let cache_bits_max = if quality <= 25 { 0 } else { max_bits };
    if cache_bits_max == 0 {
        return 0;
    }
    const CACHE_SIZE: usize = 1 << MAX_COLOR_CACHE_BITS;
    const CACHE_MASK: usize = CACHE_SIZE - 1;
    const CACHE_BASE: usize = NUM_LITERAL_CODES + NUM_LENGTH_CODES;
    let mut histos: Vec<Histogram> = (0..=cache_bits_max).map(Histogram::new).collect();
    let mut hashers = vec![[0u32; CACHE_SIZE]; cache_bits_max as usize + 1];
    let mut pos = 0usize;
    for v in refs {
        if v.mode == PixOrCopyMode::Literal {
            let pix = argb[pos];
            pos += 1;
            let a = (pix >> 24) as usize;
            let r = ((pix >> 16) & 0xff) as usize;
            let g = ((pix >> 8) & 0xff) as usize;
            let b = (pix & 0xff) as usize;
            let mut key = hash_pix(pix, 32 - cache_bits_max) as usize;
            let (h0, hs) = histos.split_first_mut().unwrap();
            h0.blue[b] += 1;
            h0.literal[g] += 1;
            h0.red[r] += 1;
            h0.alpha[a] += 1;
            for (h, colors) in hs.iter_mut().zip(&mut hashers[1..]).rev() {
                let k = key & CACHE_MASK;
                let hit = colors[k] == pix;
                colors[k] = pix;
                let miss = u32::from(!hit);
                h.literal[if hit { CACHE_BASE + k } else { g }] += 1;
                h.blue[b] += miss;
                h.red[r] += miss;
                h.alpha[a] += miss;
                key >>= 1;
            }
        } else {
            let len = usize::from(v.len);
            let (code, _, _) = prefix_encode(len as u32);
            for h in histos.iter_mut() {
                h.literal[NUM_LITERAL_CODES + code as usize] += 1;
            }
            let mut argb_prev = argb[pos] ^ 0xffffffff;
            for &p in &argb[pos..pos + len] {
                if p != argb_prev {
                    let mut key = hash_pix(p, 32 - cache_bits_max) as usize;
                    for colors in hashers[1..].iter_mut().rev() {
                        colors[key & CACHE_MASK] = p;
                        key >>= 1;
                    }
                    argb_prev = p;
                }
            }
            pos += len;
        }
    }
    let mut entropy_min = u64::MAX;
    let mut best = 0;
    for (i, h) in histos.iter().enumerate() {
        let entropy = h.estimate_bits();
        if i == 0 || entropy < entropy_min {
            entropy_min = entropy;
            best = i as u32;
        }
    }
    best
}

fn backward_refs_with_local_cache(argb: &[u32], cache_bits: u32, refs: &mut BackwardRefs) {
    let mut cc = ColorCache::new(cache_bits);
    let mut pixel_index = 0usize;
    for v in refs.iter_mut() {
        if v.mode == PixOrCopyMode::Literal {
            let argb_literal = v.argb_or_distance;
            match cc.contains(argb_literal) {
                Some(ix) => *v = PixOrCopy::cache_idx(ix),
                None => cc.insert(argb_literal),
            }
            pixel_index += 1;
        } else {
            let len = usize::from(v.len);
            let mut prev = argb[pixel_index] ^ 0xffffffff;
            for &p in &argb[pixel_index..pixel_index + len] {
                if p != prev {
                    cc.insert(p);
                    prev = p;
                }
            }
            pixel_index += len;
        }
    }
}

/// Scratch reference lists: `best` (with colour cache), `no_cache` and a temporary.
#[derive(Default)]
pub(crate) struct RefsSet {
    pub(crate) best: BackwardRefs,
    pub(crate) no_cache: BackwardRefs,
    tmp: BackwardRefs,
    hash_chain_box: HashChain,
}

/// `VP8LGetBackwardReferences`: fills `refs.best` (and `refs.no_cache` when `do_no_cache`)
/// and returns the best cache bits.
pub(crate) fn get_backward_references(
    width: u32,
    height: u32,
    argb: &[u32],
    quality: i32,
    low_effort: bool,
    mut lz77_types_to_try: u32,
    cache_bits_max: u32,
    do_no_cache: bool,
    hash_chain: &HashChain,
    refs: &mut RefsSet,
) -> u32 {
    if low_effort {
        backward_references_lz77(width, height, argb, 0, hash_chain, &mut refs.best);
        backward_references_2d_locality(width, &mut refs.best);
        return 0;
    }
    let mut histo = Histogram::new(MAX_COLOR_CACHE_BITS);
    let mut lz77_types_best = [0u32; 2];
    let mut bit_costs_best = [u64::MAX; 2];
    let mut cache_bits_best = 0;
    let RefsSet {
        best,
        no_cache,
        tmp,
        hash_chain_box,
    } = refs;
    let mut lz77_type = 1u32;
    while lz77_types_to_try != 0 {
        if lz77_types_to_try & lz77_type != 0 {
            let mut bit_cost = 0u64;
            match lz77_type {
                LZ77_RLE => backward_references_rle(width, height, argb, 0, tmp),
                LZ77_STANDARD => backward_references_lz77(width, height, argb, 0, hash_chain, tmp),
                LZ77_BOX => backward_references_lz77_box(
                    width,
                    height,
                    argb,
                    0,
                    hash_chain,
                    hash_chain_box,
                    tmp,
                ),
                _ => unreachable!(),
            }
            for i in (0..2).rev() {
                let mut cache_bits = if i == 1 { 0 } else { cache_bits_max };
                if i == 1 && !do_no_cache {
                    continue;
                }
                if i == 0 {
                    cache_bits = calculate_best_cache_size(argb, quality, tmp, cache_bits);
                    if cache_bits > 0 {
                        backward_refs_with_local_cache(argb, cache_bits, tmp);
                    }
                }
                if !(i == 0 && do_no_cache && cache_bits == 0) {
                    histo.create(tmp, cache_bits);
                    bit_cost = histo.estimate_bits();
                }
                if bit_cost < bit_costs_best[i] {
                    if i == 1 {
                        no_cache.clone_from(tmp);
                    } else {
                        std::mem::swap(tmp, best);
                    }
                    bit_costs_best[i] = bit_cost;
                    lz77_types_best[i] = lz77_type;
                    if i == 0 {
                        cache_bits_best = cache_bits;
                    }
                }
            }
        }
        lz77_types_to_try &= !lz77_type;
        lz77_type <<= 1;
    }

    for i in (0..2).rev() {
        if i == 1 && !do_no_cache {
            continue;
        }
        let target: &mut BackwardRefs = if i == 1 { &mut *no_cache } else { &mut *best };
        if (lz77_types_best[i] == LZ77_STANDARD || lz77_types_best[i] == LZ77_BOX) && quality >= 25
        {
            let hc: &HashChain = if lz77_types_best[i] == LZ77_STANDARD {
                hash_chain
            } else {
                &*hash_chain_box
            };
            let cache_bits = if i == 1 { 0 } else { cache_bits_best };
            trace_backwards(width, height, argb, cache_bits, hc, target, tmp);
            histo.create(tmp, cache_bits);
            let bit_cost_trace = histo.estimate_bits();
            if bit_cost_trace < bit_costs_best[i] {
                std::mem::swap(tmp, target);
            }
        }
        backward_references_2d_locality(width, target);
        if i == 1 && lz77_types_best[0] == lz77_types_best[1] && cache_bits_best == 0 {
            best.clone_from(no_cache);
            break;
        }
    }
    cache_bits_best
}
