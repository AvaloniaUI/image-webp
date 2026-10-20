//! Palette detection, ordering and mapping (libwebp's `utils/palette.c` and the palette
//! part of `enc/vp8l_enc.c`).

use super::backward_refs::hash_pix;
use super::entropy::sub_pixels;
use super::Pixel;

pub(crate) const MAX_PALETTE_SIZE: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PaletteSorting {
    SortedDefault,
    MinimizeDelta,
    ModifiedZeng,
}

pub(crate) fn search_color_no_idx(sorted: &[u32], color: u32, num_colors: usize) -> usize {
    let (mut low, mut hi) = (0usize, num_colors);
    if sorted[low] == color {
        return low;
    }
    loop {
        let mid = (low + hi) >> 1;
        if sorted[mid] == color {
            return mid;
        } else if sorted[mid] < color {
            low = mid;
        } else {
            hi = mid;
        }
    }
}

/// Sorted copy of `palette` and the map from sorted to original index.
pub(crate) fn prepare_map_to_palette(palette: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut sorted = palette.to_vec();
    sorted.sort_unstable();
    let mut idx_map = vec![0u32; palette.len()];
    for (i, &c) in palette.iter().enumerate() {
        idx_map[search_color_no_idx(&sorted, c, palette.len())] = i as u32;
    }
    (sorted, idx_map)
}

/// The sorted colours of `argb`, or `None` with more than 256.
pub(crate) fn get_color_palette<P: Pixel>(argb: &[P]) -> Option<Vec<u32>> {
    if let Some(bytes) = P::bytes(argb) {
        let mut seen = [false; 256];
        for &a in bytes {
            seen[usize::from(a)] = true;
        }
        return Some((0..256u32).filter(|&a| seen[a as usize]).map(|a| a << 8).collect());
    }
    const COLOR_HASH_SIZE: usize = MAX_PALETTE_SIZE * 4;
    let mut in_use = vec![false; COLOR_HASH_SIZE];
    let mut colors = vec![0u32; COLOR_HASH_SIZE];
    let mut num_colors = 0;
    let mut last_pix = !argb[0].argb();
    for &p in argb {
        let p = p.argb();
        if p == last_pix {
            continue;
        }
        last_pix = p;
        let mut key = hash_pix(p, 22) as usize;
        loop {
            if !in_use[key] {
                colors[key] = p;
                in_use[key] = true;
                num_colors += 1;
                if num_colors > MAX_PALETTE_SIZE {
                    return None;
                }
                break;
            } else if colors[key] == p {
                break;
            } else {
                key = (key + 1) & (COLOR_HASH_SIZE - 1);
            }
        }
    }
    let mut palette: Vec<u32> = (0..COLOR_HASH_SIZE)
        .filter(|&i| in_use[i])
        .map(|i| colors[i])
        .collect();
    palette.sort_unstable();
    Some(palette)
}

fn palette_component_distance(v: u32) -> u32 {
    if v <= 128 {
        v
    } else {
        256 - v
    }
}

fn palette_color_distance(col1: u32, col2: u32) -> u32 {
    let diff = sub_pixels(col1, col2);
    let mut score = palette_component_distance(diff & 0xff);
    score += palette_component_distance((diff >> 8) & 0xff);
    score += palette_component_distance((diff >> 16) & 0xff);
    score *= 9;
    score + palette_component_distance((diff >> 24) & 0xff)
}

fn palette_has_non_monotonous_deltas(palette: &[u32]) -> bool {
    let mut predict = 0u32;
    let mut sign_found = 0u8;
    for &c in palette {
        let diff = sub_pixels(c, predict);
        let rd = ((diff >> 16) & 0xff) as u8;
        let gd = ((diff >> 8) & 0xff) as u8;
        let bd = (diff & 0xff) as u8;
        if rd != 0 {
            sign_found |= if rd < 0x80 { 1 } else { 2 };
        }
        if gd != 0 {
            sign_found |= if gd < 0x80 { 8 } else { 16 };
        }
        if bd != 0 {
            sign_found |= if bd < 0x80 { 64 } else { 128 };
        }
        predict = c;
    }
    (sign_found & (sign_found << 1)) != 0
}

fn palette_sort_minimize_deltas(palette_sorted: &[u32]) -> Vec<u32> {
    let mut palette = palette_sorted.to_vec();
    if !palette_has_non_monotonous_deltas(palette_sorted) {
        return palette;
    }
    let mut num_colors = palette.len();
    if num_colors > 17 && palette[0] == 0 {
        num_colors -= 1;
        palette.swap(num_colors, 0);
    }
    let mut predict = 0u32;
    for i in 0..num_colors {
        let mut best_ix = i;
        let mut best_score = u32::MAX;
        for k in i..num_colors {
            let cur_score = palette_color_distance(palette[k], predict);
            if best_score > cur_score {
                best_score = cur_score;
                best_ix = k;
            }
        }
        palette.swap(best_ix, i);
        predict = palette[i];
    }
    palette
}

fn cooccurrence_find_max(cooccurrence: &[u32], num_colors: usize) -> (u8, u8) {
    let mut best_sum = 0u32;
    let mut c1 = 0u8;
    for i in 0..num_colors {
        let sum = cooccurrence[i * num_colors..(i + 1) * num_colors]
            .iter()
            .fold(0u32, |a, &b| a.wrapping_add(b));
        if sum > best_sum {
            best_sum = sum;
            c1 = i as u8;
        }
    }
    let mut c2 = 0u8;
    let mut best = 0u32;
    for i in 0..num_colors {
        let v = cooccurrence[c1 as usize * num_colors + i];
        if v > best {
            best = v;
            c2 = i as u8;
        }
    }
    (c1, c2)
}

fn cooccurrence_build<P: Pixel>(argb: &[P], width: usize, palette: &[u32], cooccurrence: &mut [u32]) {
    let num_colors = palette.len();
    let mut lines = vec![0u32; 2 * width];
    let (mut top_off, mut cur_off) = (0usize, width);
    let (sorted, idx_map) = prepare_map_to_palette(palette);
    let mut prev_pix = !argb[0].argb();
    let mut prev_idx = 0u32;
    for (y, row) in argb.chunks_exact(width).enumerate() {
        for (x, &pix) in row.iter().enumerate() {
            let pix = pix.argb();
            if pix != prev_pix {
                prev_idx = idx_map[search_color_no_idx(&sorted, pix, num_colors)];
                prev_pix = pix;
            }
            lines[cur_off + x] = prev_idx;
            if x > 0 && prev_idx != lines[cur_off + x - 1] {
                let left_idx = lines[cur_off + x - 1] as usize;
                cooccurrence[prev_idx as usize * num_colors + left_idx] += 1;
                cooccurrence[left_idx * num_colors + prev_idx as usize] += 1;
            }
            if y > 0 && prev_idx != lines[top_off + x] {
                let top_idx = lines[top_off + x] as usize;
                cooccurrence[prev_idx as usize * num_colors + top_idx] += 1;
                cooccurrence[top_idx * num_colors + prev_idx as usize] += 1;
            }
        }
        std::mem::swap(&mut top_off, &mut cur_off);
    }
}

#[derive(Clone, Copy)]
struct Sum {
    index: u8,
    sum: u32,
}

fn palette_sort_modified_zeng<P: Pixel>(argb: &[P], width: usize, palette_in: &[u32]) -> Vec<u32> {
    let num_colors = palette_in.len() as u32;
    if num_colors <= 1 {
        return palette_in.to_vec();
    }
    let nc = num_colors as usize;
    let mut cooccurrence = vec![0u32; nc * nc];
    cooccurrence_build(argb, width, palette_in, &mut cooccurrence);
    let mut remapping = [0u8; 256];
    let (c1, c2) = cooccurrence_find_max(&cooccurrence, nc);
    remapping[0] = c1;
    remapping[1] = c2;
    let mut first = 0u32;
    let mut last = 1u32;
    let mut num_sums = num_colors - 2;
    let mut sums = [Sum { index: 0, sum: 0 }; 256];
    if num_sums > 0 {
        let mut best = 0usize;
        sums[0] = Sum { index: 0, sum: 0 };
        let mut j = 0;
        for i in 0..nc {
            if i == remapping[0] as usize || i == remapping[1] as usize {
                continue;
            }
            sums[j].index = i as u8;
            sums[j].sum = cooccurrence[i * nc + remapping[0] as usize]
                .wrapping_add(cooccurrence[i * nc + remapping[1] as usize]);
            if sums[j].sum > sums[best].sum {
                best = j;
            }
            j += 1;
        }
        while num_sums > 0 {
            let best_index = sums[best].index as usize;
            let mut delta: i32 = 0;
            let n = (num_colors - num_sums) as i32;
            let mut j = 0u32;
            let ind = first;
            while (ind + j) % num_colors != last + 1 {
                let l_j = remapping[((ind + j) % num_colors) as usize] as usize;
                delta = delta.wrapping_add(
                    (n - 1 - 2 * j as i32).wrapping_mul(cooccurrence[best_index * nc + l_j] as i32),
                );
                j += 1;
            }
            if delta > 0 {
                first = if first == 0 {
                    num_colors - 1
                } else {
                    first - 1
                };
                remapping[first as usize] = best_index as u8;
            } else {
                last += 1;
                remapping[last as usize] = best_index as u8;
            }
            sums[best] = sums[num_sums as usize - 1];
            num_sums -= 1;
            best = 0;
            for i in 0..num_sums as usize {
                sums[i].sum = sums[i]
                    .sum
                    .wrapping_add(cooccurrence[best_index * nc + sums[i].index as usize]);
                if sums[i].sum > sums[best].sum {
                    best = i;
                }
            }
        }
    }
    (0..nc)
        .map(|i| palette_in[remapping[(first as usize + i) % nc] as usize])
        .collect()
}

/// `PaletteSort` of the lexicographically sorted palette.
pub(crate) fn palette_sort<P: Pixel>(
    method: PaletteSorting,
    argb: &[P],
    width: usize,
    palette_sorted: &[u32],
) -> Vec<u32> {
    let num_colors = palette_sorted.len();
    match method {
        PaletteSorting::SortedDefault => {
            if palette_sorted[0] == 0 && num_colors > 17 {
                let mut p = palette_sorted[1..].to_vec();
                p.push(0);
                p
            } else {
                palette_sorted.to_vec()
            }
        }
        PaletteSorting::MinimizeDelta => palette_sort_minimize_deltas(palette_sorted),
        PaletteSorting::ModifiedZeng => palette_sort_modified_zeng(argb, width, palette_sorted),
    }
}

fn bundle_color_map(row: &[u8], xbits: u32, dst: &mut [u32]) {
    if xbits > 0 {
        let bit_depth = 1 << (3 - xbits);
        let mask = (1 << xbits) - 1;
        let mut code = 0xff000000u32;
        for (x, &r) in row.iter().enumerate() {
            let xsub = x & mask;
            if xsub == 0 {
                code = 0xff000000;
            }
            code |= u32::from(r) << (8 + bit_depth * xsub);
            dst[x >> xbits] = code;
        }
    } else {
        for (d, &r) in dst.iter_mut().zip(row) {
            *d = 0xff000000 | (u32::from(r) << 8);
        }
    }
}

/// `ApplyPalette`: replaces each pixel of `src` by its palette index, bundled `xbits`
/// pixels per output pixel.
pub(crate) fn apply_palette<P: Pixel>(
    src: &[P],
    width: usize,
    height: usize,
    dst: &mut [u32],
    dst_stride: usize,
    palette: &[u32],
    xbits: u32,
) {
    let palette_size = palette.len();
    let mut tmp_row = vec![0u8; width];
    if let Some(bytes) = P::bytes(src) {
        let mut index = [0u8; 256];
        for (i, &c) in palette.iter().enumerate() {
            index[(c >> 8 & 0xff) as usize] = i as u8;
        }
        for (y, row) in bytes.chunks_exact(width).take(height).enumerate() {
            for (t, &a) in tmp_row.iter_mut().zip(row) {
                *t = index[usize::from(a)];
            }
            bundle_color_map(&tmp_row, xbits, &mut dst[y * dst_stride..]);
        }
        return;
    }
    let mut run = |index_of: &mut dyn FnMut(u32) -> u32| {
        let mut prev_pix = palette[0];
        let mut prev_idx = 0u32;
        for y in 0..height {
            for x in 0..width {
                let pix = src[y * width + x].argb();
                if pix != prev_pix {
                    prev_idx = index_of(pix);
                    prev_pix = pix;
                }
                tmp_row[x] = prev_idx as u8;
            }
            bundle_color_map(&tmp_row, xbits, &mut dst[y * dst_stride..]);
        }
    };
    if palette_size < 4 {
        run(&mut |c| palette.iter().take(3).position(|&p| p == c).unwrap_or(3) as u32);
        return;
    }
    const INV_BITS: u32 = 11;
    let hashes: [fn(u32) -> u32; 3] = [
        |c| (c >> 8) & 0xff,
        |c| (((c & 0x00ffffff) as u64 * 4222244071) as u32) >> (32 - INV_BITS),
        |c| (((c & 0x00ffffff) as u64 * ((1u64 << 31) - 1)) as u32) >> (32 - INV_BITS),
    ];
    let mut buffer = vec![0xffffu16; 1 << INV_BITS];
    for hash in hashes {
        buffer.fill(0xffff);
        let mut use_lut = true;
        for (j, &p) in palette.iter().enumerate() {
            let ind = hash(p) as usize;
            if buffer[ind] != 0xffff {
                use_lut = false;
                break;
            }
            buffer[ind] = j as u16;
        }
        if use_lut {
            run(&mut |c| u32::from(buffer[hash(c) as usize]));
            return;
        }
    }
    let (sorted, idx_map) = prepare_map_to_palette(palette);
    run(&mut |c| idx_map[search_color_no_idx(&sorted, c, palette_size)]);
}
