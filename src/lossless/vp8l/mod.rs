//! Lossless (VP8L) encoder ported from libwebp 1.6.0, producing the same bitstream as
//! `WebPEncode` with `lossless = 1` for any `method` and `quality` (near-lossless off),
//! and the same `ALPH` chunk as its lossy encoder.
//!
//! libwebp: Copyright 2012 Google Inc. All Rights Reserved. Distributed under libwebp's
//! BSD 3-Clause license and patent grant, reproduced in LICENSE-LIBWEBP.

mod alpha;
mod backward_refs;
mod bit_writer;
mod cost;
mod entropy;
mod histogram;
mod huffman;
mod near_lossless;
#[cfg(target_arch = "aarch64")]
mod neon;
mod palette;
mod predictor;
#[cfg(target_arch = "aarch64")]
use neon as simd;
#[cfg(target_arch = "x86_64")]
mod sse2;
#[cfg(target_arch = "x86_64")]
use sse2 as simd;
mod tables;

pub(crate) use alpha::{encode_alpha, encode_alpha_raw};
#[cfg(test)]
pub(crate) use alpha::encode_alpha_filtered;

use std::borrow::Cow;

use backward_refs::{
    get_backward_references, BackwardRefs, HashChain, PixOrCopyMode, RefsSet, LZ77_BOX, LZ77_RLE,
    LZ77_STANDARD,
};
use bit_writer::BitWriter;
use entropy::*;
use histogram::{get_histo_image_symbols, Histogram};
use huffman::{
    create_compressed_huffman_tree, create_huffman_tree, HuffmanTreeCode, HuffmanTreeToken,
};
use palette::{apply_palette, get_color_palette, palette_sort, PaletteSorting, MAX_PALETTE_SIZE};
use predictor::{
    color_space_transform, optimize_sampling, residual_image, Residuals, MAX_TRANSFORM_BITS,
    MIN_TRANSFORM_BITS,
};

pub(crate) const NUM_LITERAL_CODES: usize = 256;
pub(crate) const NUM_LENGTH_CODES: usize = 24;
pub(crate) const NUM_DISTANCE_CODES: usize = 40;
pub(crate) const MAX_COLOR_CACHE_BITS: u32 = 10;
const CODE_LENGTH_CODES: usize = 19;
const MIN_HUFFMAN_BITS: u32 = 2;
const MAX_HUFFMAN_BITS: u32 = MIN_HUFFMAN_BITS + (1 << 3) - 1;
const MAX_HUFF_IMAGE_SIZE: u32 = 2600;
const MAX_PREDICTOR_IMAGE_SIZE: u32 = 1 << 14;

const PREDICTOR_TRANSFORM: u32 = 0;
const CROSS_COLOR_TRANSFORM: u32 = 1;
const SUBTRACT_GREEN_TRANSFORM: u32 = 2;
const COLOR_INDEXING_TRANSFORM: u32 = 3;

/// The `WebPConfig` fields the lossless encoder reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    pub(crate) quality: f32,
    pub(crate) method: i32,
    pub(crate) exact: bool,
    /// Off restricts the transforms to the ones without spatial prediction.
    pub(crate) use_predictor: bool,
    /// Near-lossless level 0..=100, 100 for none.
    pub(crate) near_lossless: i32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EntropyIx {
    Direct = 0,
    Spatial = 1,
    SubGreen = 2,
    SpatialSubGreen = 3,
    Palette = 4,
    PaletteAndSpatial = 5,
}

const ENTROPY_IXS: [EntropyIx; 6] = [
    EntropyIx::Direct,
    EntropyIx::Spatial,
    EntropyIx::SubGreen,
    EntropyIx::SpatialSubGreen,
    EntropyIx::Palette,
    EntropyIx::PaletteAndSpatial,
];

#[derive(Clone, Copy, Debug)]
struct CrunchSubConfig {
    lz77: u32,
    do_no_cache: bool,
}

#[derive(Clone, Debug)]
struct CrunchConfig {
    entropy_idx: EntropyIx,
    palette_sorting: Option<PaletteSorting>,
    sub_configs: Vec<CrunchSubConfig>,
}

/// A pixel of the picture the encoder is given: an ARGB word, or an alpha plane's byte, which
/// is coded as green.
pub(crate) trait Pixel: Copy + Eq {
    fn argb(self) -> u32;

    /// The alpha plane's bytes, which palette lookups take a byte at a time.
    fn bytes(pic: &[Self]) -> Option<&[u8]>;

    /// The picture as the ARGB words the transforms work in, kept in `argb`.
    fn store_argb(pic: Cow<'_, [Self]>, argb: &mut Vec<u32>);
}

impl Pixel for u32 {
    #[inline(always)]
    fn argb(self) -> u32 {
        self
    }

    fn bytes(_: &[u32]) -> Option<&[u8]> {
        None
    }

    fn store_argb(pic: Cow<'_, [u32]>, argb: &mut Vec<u32>) {
        match pic {
            Cow::Owned(pic) => *argb = pic,
            Cow::Borrowed(pic) => {
                argb.clear();
                argb.extend_from_slice(pic);
            }
        }
    }
}

impl Pixel for u8 {
    #[inline(always)]
    fn argb(self) -> u32 {
        u32::from(self) << 8
    }

    fn bytes(pic: &[u8]) -> Option<&[u8]> {
        Some(pic)
    }

    fn store_argb(pic: Cow<'_, [u8]>, argb: &mut Vec<u32>) {
        argb.clear();
        argb.extend(pic.iter().map(|&a| a.argb()));
    }
}

#[inline]
fn hash_pix_palette(pix: u32) -> u8 {
    (((u64::from(pix) + u64::from(pix >> 19)).wrapping_mul(0x39c5fba7) & 0xffffffff) >> 24) as u8
}

/// `AnalyzeEntropy`: the cheapest transform mode and whether red and blue are always 0.
fn analyze_entropy<P: Pixel>(
    argb: &[P],
    width: u32,
    height: u32,
    use_palette: bool,
    palette_size: usize,
    transform_bits: u32,
) -> (EntropyIx, bool) {
    if use_palette && palette_size <= 16 {
        return (EntropyIx::Palette, true);
    }
    const A: usize = 0;
    const AP: usize = 1;
    const G: usize = 2;
    const GP: usize = 3;
    const R: usize = 4;
    const RP: usize = 5;
    const B: usize = 6;
    const BP: usize = 7;
    const RSG: usize = 8;
    const RPSG: usize = 9;
    const BSG: usize = 10;
    const BPSG: usize = 11;
    const PAL: usize = 12;
    let mut histo = vec![[0u32; 256]; 13];
    let w = width as usize;
    let mut pix_prev = argb[0].argb();
    for y in 0..height as usize {
        let row = &argb[y * w..(y + 1) * w];
        for x in 0..w {
            let pix = row[x].argb();
            let pix_diff = sub_pixels(pix, pix_prev);
            pix_prev = pix;
            if pix_diff == 0 || (y > 0 && pix == argb[(y - 1) * w + x].argb()) {
                continue;
            }
            for (p, [a, r, g, b]) in [(pix, [A, R, G, B]), (pix_diff, [AP, RP, GP, BP])] {
                histo[a][(p >> 24) as usize] += 1;
                histo[r][((p >> 16) & 0xff) as usize] += 1;
                histo[g][((p >> 8) & 0xff) as usize] += 1;
                histo[b][(p & 0xff) as usize] += 1;
            }
            for (p, r, b) in [(pix, RSG, BSG), (pix_diff, RPSG, BPSG)] {
                let green = (p as i32) >> 8;
                histo[r][((((p as i32) >> 16) - green) & 0xff) as usize] += 1;
                histo[b][(((p as i32) - green) & 0xff) as usize] += 1;
            }
            histo[PAL][hash_pix_palette(pix) as usize] += 1;
        }
    }
    for h in [RPSG, BPSG, RP, GP, BP, AP] {
        histo[h][0] += 1;
    }
    let ec: Vec<u64> = histo.iter().map(|h| bits_entropy(h)).collect();
    let mut entropy = [0u64; 5];
    entropy[0] = ec[A] + ec[R] + ec[G] + ec[B];
    entropy[1] = ec[AP] + ec[RP] + ec[GP] + ec[BP];
    entropy[2] = ec[A] + ec[RSG] + ec[G] + ec[BSG];
    entropy[3] = ec[AP] + ec[RPSG] + ec[GP] + ec[BPSG];
    entropy[4] = ec[PAL];
    let tiles = u64::from(sub_sample_size(width, transform_bits))
        * u64::from(sub_sample_size(height, transform_bits));
    entropy[1] += tiles * u64::from(fast_log2(14));
    entropy[3] += tiles * u64::from(fast_log2(24));
    entropy[4] += (palette_size as u64 * 8) << LOG_2_PRECISION_BITS;
    let last = if use_palette { 4 } else { 3 };
    let mut min_ix = 0;
    for k in 1..=last {
        if entropy[min_ix] > entropy[k] {
            min_ix = k;
        }
    }
    const PAIRS: [[usize; 2]; 5] = [[R, B], [RP, BP], [RSG, BSG], [RPSG, BPSG], [R, B]];
    let [rh, bh] = PAIRS[min_ix];
    let red_and_blue_always_zero = (1..256).all(|i| (histo[rh][i] | histo[bh][i]) == 0);
    (ENTROPY_IXS[min_ix], red_and_blue_always_zero)
}

fn clamp_bits(
    width: u32,
    height: u32,
    bits: u32,
    min_bits: u32,
    max_bits: u32,
    image_size_max: u32,
) -> u32 {
    let mut bits = bits.clamp(min_bits, max_bits);
    let mut image_size = sub_sample_size(width, bits) * sub_sample_size(height, bits);
    while bits < max_bits && image_size > image_size_max {
        bits += 1;
        image_size = sub_sample_size(width, bits) * sub_sample_size(height, bits);
    }
    while bits > min_bits && image_size == 1 {
        image_size = sub_sample_size(width, bits - 1) * sub_sample_size(height, bits - 1);
        if image_size != 1 {
            break;
        }
        bits -= 1;
    }
    bits
}

fn clamp_bits_signed(
    width: u32,
    height: u32,
    bits: i32,
    min_bits: u32,
    max_bits: u32,
    image_size_max: u32,
) -> u32 {
    clamp_bits(
        width,
        height,
        bits.max(0) as u32,
        min_bits,
        max_bits,
        image_size_max,
    )
}

fn get_histo_bits(method: i32, use_palette: bool, width: u32, height: u32) -> u32 {
    let histo_bits = (if use_palette { 9 } else { 7 }) - method;
    clamp_bits_signed(
        width,
        height,
        histo_bits,
        MIN_HUFFMAN_BITS,
        MAX_HUFFMAN_BITS,
        MAX_HUFF_IMAGE_SIZE,
    )
}

fn get_transform_bits(method: i32, histo_bits: u32) -> u32 {
    let max_transform_bits = if method < 4 {
        6
    } else if method > 4 {
        4
    } else {
        5
    };
    histo_bits.min(max_transform_bits)
}

struct Encoder {
    config: Config,
    width: u32,
    height: u32,
    histo_bits: u32,
    transform_bits: u32,
    palette_sorted: Vec<u32>,
    use_palette: bool,
    argb: Vec<u32>,
    current_width: u32,
    transform_data: Vec<u32>,
    hash_chain: HashChain,
    refs: RefsSet,
    refs_side: RefsSet,
}

impl Encoder {
    /// `EncoderAnalyze`: the configurations to try and whether red and blue are always 0.
    fn analyze<P: Pixel>(&mut self, argb: &[P]) -> (Vec<CrunchConfig>, bool) {
        let (width, height) = (self.width, self.height);
        let method = self.config.method;
        let quality = self.config.quality;
        let low_effort = method == 0;
        let palette = get_color_palette(argb);
        self.use_palette = palette.is_some();
        self.palette_sorted = palette.unwrap_or_default();
        let palette_size = self.palette_sorted.len();
        self.histo_bits = get_histo_bits(method, self.use_palette, width, height);
        self.transform_bits = get_transform_bits(method, self.histo_bits);
        let mut red_and_blue_always_zero = false;
        let mut configs: Vec<(EntropyIx, Option<PaletteSorting>)> = Vec::new();
        let n_lz77s;
        let mut do_no_cache = false;
        if low_effort {
            configs.push(if self.use_palette {
                (EntropyIx::Palette, Some(PaletteSorting::SortedDefault))
            } else {
                (EntropyIx::SpatialSubGreen, None)
            });
            n_lz77s = 1;
        } else {
            n_lz77s = if palette_size > 0 && palette_size <= 16 {
                2
            } else {
                1
            };
            let (min_entropy_ix, rb_zero) = analyze_entropy(
                argb,
                width,
                height,
                self.use_palette,
                palette_size,
                self.transform_bits,
            );
            red_and_blue_always_zero = rb_zero;
            if method == 6 && quality == 100. {
                do_no_cache = true;
                for ix in ENTROPY_IXS {
                    let is_palette =
                        matches!(ix, EntropyIx::Palette | EntropyIx::PaletteAndSpatial);
                    if is_palette && !self.use_palette {
                        continue;
                    }
                    if is_palette {
                        configs.push((ix, Some(PaletteSorting::MinimizeDelta)));
                        configs.push((ix, Some(PaletteSorting::ModifiedZeng)));
                    } else {
                        configs.push((ix, None));
                    }
                }
            } else {
                configs.push((
                    min_entropy_ix,
                    self.use_palette.then_some(PaletteSorting::MinimizeDelta),
                ));
                if quality >= 75. && method == 5 {
                    do_no_cache = true;
                    if min_entropy_ix == EntropyIx::Palette {
                        configs.push((
                            EntropyIx::PaletteAndSpatial,
                            Some(PaletteSorting::MinimizeDelta),
                        ));
                    }
                }
            }
        }
        if !self.config.use_predictor {
            for c in configs.iter_mut() {
                c.0 = match c.0 {
                    EntropyIx::Spatial => EntropyIx::Direct,
                    EntropyIx::SpatialSubGreen => EntropyIx::SubGreen,
                    EntropyIx::PaletteAndSpatial => EntropyIx::Palette,
                    e => e,
                };
            }
            configs.dedup();
        }
        let crunch = configs
            .into_iter()
            .map(|(entropy_idx, palette_sorting)| CrunchConfig {
                entropy_idx,
                palette_sorting,
                sub_configs: (0..n_lz77s)
                    .map(|j| CrunchSubConfig {
                        lz77: if j == 0 {
                            LZ77_STANDARD | LZ77_RLE
                        } else {
                            LZ77_BOX
                        },
                        do_no_cache,
                    })
                    .collect(),
            })
            .collect();
        (crunch, red_and_blue_always_zero)
    }

    /// `EncodeStreamHook` for all configurations: appends the smallest stream to `bw`.
    /// Takes `pic` to drop or reuse it once the last configuration no longer needs it.
    fn encode_stream<P: Pixel>(&mut self, mut pic: Cow<'_, [P]>, bw: &mut BitWriter) {
        let (configs, red_and_blue_always_zero) = self.analyze(&pic[..]);
        let quality = self.config.quality as i32;
        let low_effort = self.config.method == 0;
        let height = self.height;
        let init = bw.clone();
        let mut best: Option<BitWriter> = None;
        for (idx, config) in configs.iter().enumerate() {
            let last = idx + 1 == configs.len();
            let mut w = if configs.len() > 1 {
                init.clone()
            } else {
                std::mem::take(bw)
            };
            let entropy_idx = config.entropy_idx;
            let use_palette = matches!(
                entropy_idx,
                EntropyIx::Palette | EntropyIx::PaletteAndSpatial
            );
            let use_subtract_green = matches!(
                entropy_idx,
                EntropyIx::SubGreen | EntropyIx::SpatialSubGreen
            );
            let use_predict = matches!(
                entropy_idx,
                EntropyIx::Spatial | EntropyIx::SpatialSubGreen | EntropyIx::PaletteAndSpatial
            );
            let use_cross_color =
                !(low_effort || use_palette) && !red_and_blue_always_zero && use_predict;
            let mut cache_bits = 0;
            if use_palette {
                let palette = palette_sort(
                    config.palette_sorting.unwrap(),
                    &pic[..],
                    self.width as usize,
                    &self.palette_sorted,
                );
                self.encode_palette(&mut w, low_effort, &palette);
                self.map_image_from_palette(&pic[..], &palette);
                if last {
                    pic = Cow::Borrowed(&[]);
                }
                if palette.len() < (1 << MAX_COLOR_CACHE_BITS) {
                    cache_bits = bits_log2_floor(palette.len() as u32) + 1;
                }
            } else {
                let pic = match last {
                    true => std::mem::take(&mut pic),
                    false => Cow::Borrowed(&*pic),
                };
                P::store_argb(pic, &mut self.argb);
                self.current_width = self.width;
                if self.config.near_lossless < 100 && !use_predict {
                    near_lossless::NearLossless::apply(
                        &mut self.argb,
                        self.width as usize,
                        height as usize,
                        self.config.near_lossless,
                    );
                }
            }
            let cw = self.current_width;
            if use_subtract_green {
                w.put_bits(1, 1);
                w.put_bits(SUBTRACT_GREEN_TRANSFORM, 2);
                for p in self.argb.iter_mut() {
                    let argb = *p as i32;
                    let green = (argb >> 8) & 0xff;
                    let new_r = (((argb >> 16) & 0xff) - green) as u32 & 0xff;
                    let new_b = ((argb & 0xff) - green) as u32 & 0xff;
                    *p = (*p & 0xff00ff00) | (new_r << 16) | new_b;
                }
            }
            if use_predict {
                self.apply_predict_filter(
                    &mut w,
                    cw,
                    height,
                    quality,
                    low_effort,
                    use_palette,
                    use_subtract_green,
                );
            }
            if use_cross_color {
                self.apply_cross_color_filter(&mut w, cw, height, quality, low_effort);
            }
            w.put_bits(0, 1);
            encode_image_internal(
                &mut w,
                &self.argb,
                &mut self.hash_chain,
                &mut self.refs,
                &mut self.refs_side,
                cw,
                height,
                quality,
                low_effort,
                config,
                cache_bits,
                self.histo_bits,
            );
            if best
                .as_ref()
                .map_or(true, |b| w.num_bytes() < b.num_bytes())
            {
                best = Some(w);
            }
        }
        *bw = best.unwrap();
    }

    fn transform_data_len(&self, width: u32, height: u32) -> usize {
        (sub_sample_size(width, MIN_TRANSFORM_BITS) * sub_sample_size(height, MIN_TRANSFORM_BITS))
            as usize
    }

    fn apply_predict_filter(
        &mut self,
        bw: &mut BitWriter,
        width: u32,
        height: u32,
        quality: i32,
        low_effort: bool,
        used_palette: bool,
        used_subtract_green: bool,
    ) {
        let method = self.config.method;
        let max_bits = clamp_bits(
            width,
            height,
            self.transform_bits,
            MIN_TRANSFORM_BITS,
            MAX_TRANSFORM_BITS,
            MAX_PREDICTOR_IMAGE_SIZE,
        );
        let min_bits = clamp_bits_signed(
            width,
            height,
            max_bits as i32 - 2 * if method > 4 { method - 4 } else { 0 },
            MIN_TRANSFORM_BITS,
            MAX_TRANSFORM_BITS,
            MAX_PREDICTOR_IMAGE_SIZE,
        );
        let strength = if used_palette {
            100
        } else {
            self.config.near_lossless
        };
        let res = Residuals {
            exact: self.config.exact,
            max_quantization: near_lossless::max_quantization(strength),
            subtract_green: used_subtract_green,
        };
        let n = self.transform_data_len(width, height);
        self.transform_data.clear();
        self.transform_data.resize(n, 0);
        let best_bits = residual_image(
            width as usize,
            height as usize,
            min_bits,
            max_bits,
            low_effort,
            &mut self.argb,
            &mut self.transform_data,
            res,
        );
        bw.put_bits(1, 1);
        bw.put_bits(PREDICTOR_TRANSFORM, 2);
        bw.put_bits(best_bits - MIN_TRANSFORM_BITS, 3);
        let (tw, th) = (
            sub_sample_size(width, best_bits),
            sub_sample_size(height, best_bits),
        );
        encode_image_no_huffman(
            bw,
            &self.transform_data[..(tw * th) as usize],
            &mut self.hash_chain,
            &mut self.refs,
            tw,
            th,
            quality,
            low_effort,
        );
    }

    fn apply_cross_color_filter(
        &mut self,
        bw: &mut BitWriter,
        width: u32,
        height: u32,
        quality: i32,
        low_effort: bool,
    ) {
        let n = self.transform_data_len(width, height);
        self.transform_data.clear();
        self.transform_data.resize(n, 0);
        let best_bits = color_space_transform(
            width as usize,
            height as usize,
            self.transform_bits,
            quality,
            &mut self.argb,
            &mut self.transform_data,
        );
        bw.put_bits(1, 1);
        bw.put_bits(CROSS_COLOR_TRANSFORM, 2);
        bw.put_bits(best_bits - MIN_TRANSFORM_BITS, 3);
        let (tw, th) = (
            sub_sample_size(width, best_bits),
            sub_sample_size(height, best_bits),
        );
        encode_image_no_huffman(
            bw,
            &self.transform_data[..(tw * th) as usize],
            &mut self.hash_chain,
            &mut self.refs,
            tw,
            th,
            quality,
            low_effort,
        );
    }

    fn encode_palette(&mut self, bw: &mut BitWriter, low_effort: bool, palette: &[u32]) {
        let palette_size = palette.len();
        let encoded_size = if palette[palette_size - 1] == 0 && palette_size > 17 {
            palette_size - 1
        } else {
            palette_size
        };
        bw.put_bits(1, 1);
        bw.put_bits(COLOR_INDEXING_TRANSFORM, 2);
        bw.put_bits(encoded_size as u32 - 1, 8);
        let mut tmp = vec![0u32; encoded_size];
        for i in (1..encoded_size).rev() {
            tmp[i] = sub_pixels(palette[i], palette[i - 1]);
        }
        tmp[0] = palette[0];
        encode_image_no_huffman(
            bw,
            &tmp,
            &mut self.hash_chain,
            &mut self.refs,
            encoded_size as u32,
            1,
            20,
            low_effort,
        );
    }

    fn map_image_from_palette<P: Pixel>(&mut self, pic: &[P], palette: &[u32]) {
        let palette_size = palette.len();
        let xbits = if palette_size <= 4 {
            if palette_size <= 2 {
                3
            } else {
                2
            }
        } else if palette_size <= 16 {
            1
        } else {
            0
        };
        self.current_width = sub_sample_size(self.width, xbits);
        let n = (self.current_width * self.height) as usize;
        self.argb.clear();
        self.argb.resize(n, 0);
        apply_palette(
            pic,
            self.width as usize,
            self.height as usize,
            &mut self.argb,
            self.current_width as usize,
            palette,
            xbits,
        );
    }
}

/// `GetHuffBitLengthsAndCodes`: five codes per histogram.
fn get_huff_bit_lengths_and_codes(histograms: &mut [Histogram]) -> Vec<HuffmanTreeCode> {
    let mut codes = vec![HuffmanTreeCode::default(); 5 * histograms.len()];
    for (h, c) in histograms.iter_mut().zip(codes.chunks_exact_mut(5)) {
        create_huffman_tree(&mut h.literal, 15, &mut c[0]);
        create_huffman_tree(&mut h.red, 15, &mut c[1]);
        create_huffman_tree(&mut h.blue, 15, &mut c[2]);
        create_huffman_tree(&mut h.alpha, 15, &mut c[3]);
        create_huffman_tree(&mut h.distance, 15, &mut c[4]);
    }
    codes
}

fn store_huffman_tree_of_huffman_tree_to_bit_mask(bw: &mut BitWriter, code_length_bitdepth: &[u8]) {
    const STORAGE_ORDER: [usize; CODE_LENGTH_CODES] = [
        17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ];
    let mut codes_to_store = CODE_LENGTH_CODES;
    while codes_to_store > 4 {
        if code_length_bitdepth[STORAGE_ORDER[codes_to_store - 1]] != 0 {
            break;
        }
        codes_to_store -= 1;
    }
    bw.put_bits(codes_to_store as u32 - 4, 4);
    for &o in &STORAGE_ORDER[..codes_to_store] {
        bw.put_bits(u32::from(code_length_bitdepth[o]), 3);
    }
}

fn clear_huffman_tree_if_only_one_symbol(code: &mut HuffmanTreeCode) {
    let count = code
        .code_lengths
        .iter()
        .filter(|&&l| l != 0)
        .take(2)
        .count();
    if count > 1 {
        return;
    }
    code.code_lengths.fill(0);
    code.codes.fill(0);
}

fn store_huffman_tree_to_bit_mask(
    bw: &mut BitWriter,
    tokens: &[HuffmanTreeToken],
    code: &HuffmanTreeCode,
) {
    for t in tokens {
        let ix = t.code as usize;
        bw.put_bits(u32::from(code.codes[ix]), u32::from(code.code_lengths[ix]));
        match ix {
            16 => bw.put_bits(u32::from(t.extra_bits), 2),
            17 => bw.put_bits(u32::from(t.extra_bits), 3),
            18 => bw.put_bits(u32::from(t.extra_bits), 7),
            _ => {}
        }
    }
}

fn store_full_huffman_code(
    bw: &mut BitWriter,
    tokens: &mut Vec<HuffmanTreeToken>,
    tree: &HuffmanTreeCode,
) {
    bw.put_bits(0, 1);
    create_compressed_huffman_tree(tree, tokens);
    let num_tokens = tokens.len();
    let mut histogram = [0u32; CODE_LENGTH_CODES];
    for t in tokens.iter() {
        histogram[t.code as usize] += 1;
    }
    let mut huffman_code = HuffmanTreeCode::default();
    create_huffman_tree(&mut histogram, 7, &mut huffman_code);
    store_huffman_tree_of_huffman_tree_to_bit_mask(bw, &huffman_code.code_lengths);
    clear_huffman_tree_if_only_one_symbol(&mut huffman_code);
    let mut trailing_zero_bits = 0u32;
    let mut trimmed_length = num_tokens;
    for t in tokens.iter().rev() {
        let ix = t.code as usize;
        if ix == 0 || ix == 17 || ix == 18 {
            trimmed_length -= 1;
            trailing_zero_bits += u32::from(huffman_code.code_lengths[ix]);
            if ix == 17 {
                trailing_zero_bits += 3;
            } else if ix == 18 {
                trailing_zero_bits += 7;
            }
        } else {
            break;
        }
    }
    let write_trimmed_length = trimmed_length > 1 && trailing_zero_bits > 12;
    let length = if write_trimmed_length {
        trimmed_length
    } else {
        num_tokens
    };
    bw.put_bits(u32::from(write_trimmed_length), 1);
    if write_trimmed_length {
        if trimmed_length == 2 {
            bw.put_bits(0, 3 + 2);
        } else {
            let nbits = bits_log2_floor(trimmed_length as u32 - 2);
            let nbitpairs = nbits / 2 + 1;
            bw.put_bits(nbitpairs - 1, 3);
            bw.put_bits(trimmed_length as u32 - 2, nbitpairs * 2);
        }
    }
    store_huffman_tree_to_bit_mask(bw, &tokens[..length], &huffman_code);
}

fn store_huffman_code(
    bw: &mut BitWriter,
    tokens: &mut Vec<HuffmanTreeToken>,
    code: &HuffmanTreeCode,
) {
    let mut count = 0;
    let mut symbols = [0usize; 2];
    for (i, &l) in code.code_lengths.iter().enumerate() {
        if count >= 3 {
            break;
        }
        if l != 0 {
            if count < 2 {
                symbols[count] = i;
            }
            count += 1;
        }
    }
    if count == 0 {
        bw.put_bits(0x01, 4);
    } else if count <= 2 && symbols[0] < 256 && symbols[1] < 256 {
        bw.put_bits(1, 1);
        bw.put_bits(count as u32 - 1, 1);
        if symbols[0] <= 1 {
            bw.put_bits(0, 1);
            bw.put_bits(symbols[0] as u32, 1);
        } else {
            bw.put_bits(1, 1);
            bw.put_bits(symbols[0] as u32, 8);
        }
        if count == 2 {
            bw.put_bits(symbols[1] as u32, 8);
        }
    } else {
        store_full_huffman_code(bw, tokens, code);
    }
}

fn store_huffman_codes(bw: &mut BitWriter, codes: &mut [HuffmanTreeCode]) {
    let mut tokens = Vec::new();
    for code in codes.iter_mut() {
        store_huffman_code(bw, &mut tokens, code);
        clear_huffman_tree_if_only_one_symbol(code);
    }
}

#[inline]
fn write_huffman_code(bw: &mut BitWriter, code: &HuffmanTreeCode, index: usize) {
    bw.put_bits(
        u32::from(code.codes[index]),
        u32::from(code.code_lengths[index]),
    );
}

fn store_image_to_bit_mask(
    bw: &mut BitWriter,
    width: u32,
    histo_bits: u32,
    refs: &BackwardRefs,
    histogram_symbols: &[u32],
    huffman_codes: &[HuffmanTreeCode],
) {
    let histo_xsize = if histo_bits > 0 {
        sub_sample_size(width, histo_bits)
    } else {
        1
    };
    let tile_mask: u32 = if histo_bits == 0 {
        0
    } else {
        !((1u32 << histo_bits) - 1)
    };
    let (mut x, mut y) = (0u32, 0u32);
    let (mut tile_x, mut tile_y) = (x & tile_mask, y & tile_mask);
    let mut histogram_ix = ((histogram_symbols[0] >> 8) & 0xffff) as usize;
    let mut codes = &huffman_codes[5 * histogram_ix..5 * histogram_ix + 5];
    for v in refs {
        if tile_x != (x & tile_mask) || tile_y != (y & tile_mask) {
            tile_x = x & tile_mask;
            tile_y = y & tile_mask;
            histogram_ix = ((histogram_symbols
                [((y >> histo_bits) * histo_xsize + (x >> histo_bits)) as usize]
                >> 8)
                & 0xffff) as usize;
            codes = &huffman_codes[5 * histogram_ix..5 * histogram_ix + 5];
        }
        match v.mode {
            PixOrCopyMode::Literal => {
                let p = v.argb_or_distance;
                write_huffman_code(bw, &codes[0], ((p >> 8) & 0xff) as usize);
                write_huffman_code(bw, &codes[1], ((p >> 16) & 0xff) as usize);
                write_huffman_code(bw, &codes[2], (p & 0xff) as usize);
                write_huffman_code(bw, &codes[3], (p >> 24) as usize);
            }
            PixOrCopyMode::CacheIdx => {
                write_huffman_code(
                    bw,
                    &codes[0],
                    NUM_LITERAL_CODES + NUM_LENGTH_CODES + v.argb_or_distance as usize,
                );
            }
            PixOrCopyMode::Copy => {
                let (code, n_bits, bits) = prefix_encode(u32::from(v.len));
                let c = &codes[0];
                let depth = u32::from(c.code_lengths[256 + code as usize]);
                let symbol = u32::from(c.codes[256 + code as usize]);
                bw.put_bits((bits << depth) | symbol, depth + n_bits);
                let (code, n_bits, bits) = prefix_encode(v.argb_or_distance);
                write_huffman_code(bw, &codes[4], code as usize);
                bw.put_bits(bits, n_bits);
            }
        }
        x += u32::from(v.len);
        while x >= width {
            x -= width;
            y += 1;
        }
    }
}

/// `EncodeImageNoHuffman`: an image with one set of codes and no colour cache, as used
/// for transform data and the histogram image.
#[allow(clippy::too_many_arguments)]
fn encode_image_no_huffman(
    bw: &mut BitWriter,
    argb: &[u32],
    hash_chain: &mut HashChain,
    refs: &mut RefsSet,
    width: u32,
    height: u32,
    quality: i32,
    low_effort: bool,
) {
    hash_chain.fill(quality, argb, width, height, low_effort);
    let cache_bits = get_backward_references(
        width,
        height,
        argb,
        quality,
        false,
        LZ77_STANDARD | LZ77_RLE,
        0,
        false,
        hash_chain,
        refs,
    );
    let mut histo = Histogram::new(cache_bits);
    histo.store_refs(&refs.best, None);
    let mut codes = get_huff_bit_lengths_and_codes(std::slice::from_mut(&mut histo));
    bw.put_bits(0, 1);
    store_huffman_codes(bw, &mut codes);
    store_image_to_bit_mask(bw, width, 0, &refs.best, &[0], &codes);
}

/// `EncodeImageInternal`: the entropy-coded image, the smallest over the sub-configurations
/// and with or without colour cache.
#[allow(clippy::too_many_arguments)]
fn encode_image_internal(
    bw: &mut BitWriter,
    argb: &[u32],
    hash_chain: &mut HashChain,
    refs: &mut RefsSet,
    refs_side: &mut RefsSet,
    width: u32,
    height: u32,
    quality: i32,
    low_effort: bool,
    config: &CrunchConfig,
    cache_bits: u32,
    histogram_bits_in: u32,
) {
    let xysize = (sub_sample_size(width, histogram_bits_in)
        * sub_sample_size(height, histogram_bits_in)) as usize;
    let mut histogram_argb = vec![0u32; xysize];
    let mut hash_chain_histogram = HashChain::default();
    hash_chain.fill(quality, argb, width, height, low_effort);
    let cache_bits_init = if cache_bits == 0 {
        MAX_COLOR_CACHE_BITS
    } else {
        cache_bits
    };
    let init = std::mem::take(bw);
    let mut best: Option<BitWriter> = None;
    for sub_config in &config.sub_configs {
        let cache_bits_best = get_backward_references(
            width,
            height,
            argb,
            quality,
            low_effort,
            sub_config.lz77,
            cache_bits_init,
            sub_config.do_no_cache,
            hash_chain,
            refs,
        );
        for i_cache in 0..if sub_config.do_no_cache { 2 } else { 1 } {
            let cache_bits_tmp = if i_cache == 0 { cache_bits_best } else { 0 };
            if i_cache == 1 && cache_bits_best == 0 {
                break;
            }
            let mut w = init.clone();
            let mut histogram_bits = histogram_bits_in;
            let refs_i: &BackwardRefs = if i_cache == 0 {
                &refs.best
            } else {
                &refs.no_cache
            };
            let mut histogram_image = get_histo_image_symbols(
                width,
                height,
                refs_i,
                quality,
                low_effort,
                histogram_bits,
                cache_bits_tmp,
                &mut histogram_argb,
            );
            let mut codes = get_huff_bit_lengths_and_codes(&mut histogram_image);
            drop(histogram_image);
            if cache_bits_tmp > 0 {
                w.put_bits(1, 1);
                w.put_bits(cache_bits_tmp, 4);
            } else {
                w.put_bits(0, 1);
            }
            let mut histogram_image_size = 0u32;
            for h in histogram_argb.iter_mut() {
                if *h >= histogram_image_size {
                    histogram_image_size = *h + 1;
                }
                *h <<= 8;
            }
            let write_histogram_image = histogram_image_size > 1;
            w.put_bits(u32::from(write_histogram_image), 1);
            if write_histogram_image {
                histogram_bits = optimize_sampling(
                    &mut histogram_argb,
                    width,
                    height,
                    histogram_bits_in,
                    MAX_HUFFMAN_BITS,
                );
                w.put_bits(histogram_bits - 2, 3);
                let (hw, hh) = (
                    sub_sample_size(width, histogram_bits),
                    sub_sample_size(height, histogram_bits),
                );
                encode_image_no_huffman(
                    &mut w,
                    &histogram_argb[..(hw * hh) as usize],
                    &mut hash_chain_histogram,
                    refs_side,
                    hw,
                    hh,
                    quality,
                    low_effort,
                );
            }
            store_huffman_codes(&mut w, &mut codes[..5 * histogram_image_size as usize]);
            store_image_to_bit_mask(
                &mut w,
                width,
                histogram_bits,
                refs_i,
                &histogram_argb,
                &codes,
            );
            if best
                .as_ref()
                .map_or(true, |b| w.num_bytes() < b.num_bytes())
            {
                best = Some(w);
            }
        }
    }
    *bw = best.unwrap();
}

/// `VP8LEncodeStream`: the transforms and entropy-coded image of `argb`, appended to `bw`.
fn encode_stream<P: Pixel>(
    config: Config,
    pic: Cow<'_, [P]>,
    width: u32,
    height: u32,
    bw: &mut BitWriter,
) {
    let mut enc = Encoder {
        config,
        width,
        height,
        histo_bits: 0,
        transform_bits: 0,
        palette_sorted: Vec::new(),
        use_palette: false,
        argb: Vec::new(),
        current_width: width,
        transform_data: Vec::new(),
        hash_chain: HashChain::default(),
        refs: RefsSet::default(),
        refs_side: RefsSet::default(),
    };
    let _ = MAX_PALETTE_SIZE;
    enc.encode_stream(pic, bw);
}

/// `VP8LEncodeImage` minus the RIFF header: the `VP8L` chunk payload of an ARGB image.
pub(crate) fn encode_argb(config: Config, mut argb: Vec<u32>, width: u32, height: u32) -> Vec<u8> {
    if !config.exact {
        for p in argb.iter_mut() {
            if *p >> 24 == 0 {
                *p = 0;
            }
        }
    }
    let has_alpha = argb.iter().any(|&p| p >> 24 != 0xff);
    let mut bw = BitWriter::with_capacity(argb.len());
    bw.put_bits(width - 1, 14);
    bw.put_bits(height - 1, 14);
    bw.put_bits(u32::from(has_alpha), 1);
    bw.put_bits(0, 3);
    encode_stream(config, Cow::Owned(argb), width, height, &mut bw);
    let mut out = vec![0x2f];
    out.extend_from_slice(&bw.finish());
    out
}

/// The `VP8L` payload of `data` in `color`, as `WebPEncode` with `lossless = 1`.
pub(crate) fn encode_lossless(
    data: &[u8],
    width: u32,
    height: u32,
    color: crate::ColorType,
    config: Config,
) -> Vec<u8> {
    use crate::ColorType;
    let argb: Vec<u32> = match color {
        ColorType::L8 => data
            .iter()
            .map(|&l| 0xff000000 | u32::from(l) * 0x010101)
            .collect(),
        ColorType::La8 => data
            .chunks_exact(2)
            .map(|p| u32::from(p[1]) << 24 | u32::from(p[0]) * 0x010101)
            .collect(),
        ColorType::Rgb8 => data
            .chunks_exact(3)
            .map(|p| 0xff000000 | u32::from(p[0]) << 16 | u32::from(p[1]) << 8 | u32::from(p[2]))
            .collect(),
        ColorType::Rgba8 => data
            .chunks_exact(4)
            .map(|p| {
                u32::from(p[3]) << 24
                    | u32::from(p[0]) << 16
                    | u32::from(p[1]) << 8
                    | u32::from(p[2])
            })
            .collect(),
    };
    encode_argb(config, argb, width, height)
}

/// `EncodeLossless` of alpha_enc.c: the green-only image of `alpha`, without header.
fn encode_alpha_plane(
    alpha: &[u8],
    width: u32,
    height: u32,
    effort_level: i32,
    use_quality_100: bool,
) -> Vec<u8> {
    encode_green(Cow::Borrowed(alpha), width, height, effort_level, use_quality_100)
}

fn encode_green<P: Pixel>(
    pic: Cow<'_, [P]>,
    width: u32,
    height: u32,
    effort_level: i32,
    use_quality_100: bool,
) -> Vec<u8> {
    let config = Config {
        quality: if use_quality_100 && effort_level == 6 {
            100.
        } else {
            8. * effort_level as f32
        },
        method: effort_level,
        exact: true,
        use_predictor: true,
        near_lossless: 100,
    };
    let mut bw = BitWriter::with_capacity(pic.len() >> 3);
    encode_stream(config, pic, width, height, &mut bw);
    bw.finish()
}

pub(crate) fn bench_alpha_plane(alpha: &[u8], width: u32, height: u32, method: u8) -> usize {
    encode_alpha_plane(alpha, width, height, i32::from(method), true).len()
}

#[cfg(test)]
mod tests {
    use super::backward_refs::{distance_to_plane_code, PixOrCopy};
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// A stream of one code group written from chosen references and codes: literals, copies
    /// of the row above and four-pixel copies from half a million to a million pixels back,
    /// whose length and distance symbols get 15-bit codes from a geometric histogram, so the
    /// distance's 18 extra bits follow two long codes, past one fill of the decoder's window.
    /// The alpha plane coded from its bytes is the stream coded from the green-only ARGB
    /// image libwebp builds, at every effort, for planes that take each transform choice.
    #[test]
    fn alpha_plane_from_bytes_matches_green_argb() {
        let mut rng = StdRng::seed_from_u64(11);
        let (w, h) = (67usize, 41usize);
        let mut planes: Vec<Vec<u8>> = vec![
            vec![0; w * h],
            (0..w * h).map(|i| if (i % w) * 3 > w + (i / w) { 255 } else { 0 }).collect(),
            (0..w * h).map(|i| [0, 64, 128, 255][(i / 7 + i % w / 9) % 4]).collect(),
            (0..w * h).map(|i| ((i % w) * 255 / w) as u8).collect(),
            (0..w * h).map(|i| ((i % w) * 2 + (i / w) * 3) as u8).collect(),
            (0..w * h).map(|_| rng.gen()).collect(),
            (0..w * h).map(|_| rng.gen_range(0..12u8) * 20).collect(),
            (0..w * h).map(|_| rng.gen_range(100..=255u8)).collect(),
        ];
        let blobs: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i32 - 30, (i / w) as i32 - 20);
                (255 - (x * x + y * y).min(255)) as u8
            })
            .collect();
        planes.push(blobs);
        for (k, plane) in planes.iter().enumerate() {
            for (pw, ph) in [(w, h), (1, w * h), (w * h, 1)] {
                for effort in 0..=6 {
                    for q100 in [true, false] {
                        let green: Vec<u32> = plane.iter().map(|&a| u32::from(a) << 8).collect();
                        let want = encode_green(Cow::Owned(green), pw as u32, ph as u32, effort, q100);
                        let got = encode_alpha_plane(plane, pw as u32, ph as u32, effort, q100);
                        assert!(got == want, "plane {k} {pw}x{ph} effort {effort} q100 {q100}");
                    }
                }
            }
        }
    }

    #[test]
    fn far_copies_after_long_codes_decode() {
        let (width, height) = (2048u32, 600u32);
        let total = (width * height) as usize;
        let mut rng = StdRng::seed_from_u64(3);
        let mut argb: Vec<u32> = Vec::with_capacity(total);
        let mut refs: BackwardRefs = Vec::new();
        let mut far = 0;
        while argb.len() < total {
            let at = argb.len();
            let left = total - at;
            let (dist, len) = if at >= (1 << 19) + 1000 && rng.gen_range(0..300) == 0 && left >= 4 {
                far += 1;
                (rng.gen_range((1 << 19) + 1000..((1 << 20) - 120).min(at)), 4)
            } else if at >= width as usize && rng.gen_range(0..50) == 0 && left >= 10 {
                (width as usize, 10)
            } else {
                let green = rng.gen::<u32>().trailing_zeros().min(13);
                let p = 0xff000000 | rng.gen_range(0..4u32) << 16 | green << 8;
                argb.push(p);
                refs.push(PixOrCopy::literal(p));
                continue;
            };
            for i in 0..len {
                argb.push(argb[at - dist + i]);
            }
            refs.push(PixOrCopy::copy(distance_to_plane_code(width, dist as u32), len as u32));
        }
        assert!(far > 1000, "{far} far copies");
        let mut histo = Histogram::new(0);
        histo.store_refs(&refs, None);
        for k in 0..14 {
            histo.literal[k] = 1 << (14 - k);
            histo.distance[k] = 1 << (14 - k);
        }
        histo.literal[256 + 3] = 1;
        histo.literal[256 + 7] = 1 << 4;
        histo.distance[38] = 1;
        histo.distance[39] = 1;
        let mut codes = get_huff_bit_lengths_and_codes(std::slice::from_mut(&mut histo));
        assert_eq!((codes[0].code_lengths[256 + 3], codes[4].code_lengths[38]), (15, 15));
        let mut bw = BitWriter::with_capacity(total);
        bw.put_bits(width - 1, 14);
        bw.put_bits(height - 1, 14);
        bw.put_bits(0, 1);
        bw.put_bits(0, 3);
        bw.put_bits(0, 1);
        bw.put_bits(0, 1);
        bw.put_bits(0, 1);
        store_huffman_codes(&mut bw, &mut codes);
        store_image_to_bit_mask(&mut bw, width, 0, &refs, &[0], &codes);
        let mut payload = vec![0x2f];
        payload.extend_from_slice(&bw.finish());
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(12 + payload.len() as u32 + payload.len() as u32 % 2).to_le_bytes());
        data.extend_from_slice(b"WEBPVP8L");
        data.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        data.extend_from_slice(&payload);
        if payload.len() % 2 == 1 {
            data.push(0);
        }
        let headers = crate::headers::decoding(&data).unwrap();
        let mut decoder = crate::FrameDecoder::new(&data, &headers, None, true).unwrap();
        let mut rgba = Vec::with_capacity(total * 4);
        while let Some((_, rows)) = decoder.next_band().unwrap() {
            rgba.extend_from_slice(rows);
        }
        let expected: Vec<u8> = argb
            .iter()
            .flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8, (p >> 24) as u8])
            .collect();
        let wrong = rgba.chunks(4).zip(expected.chunks(4)).filter(|(a, b)| a != b).count();
        assert_eq!(wrong, 0, "{wrong} wrong pixels");
    }
}
