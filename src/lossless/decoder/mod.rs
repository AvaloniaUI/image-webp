//! Decoding of lossless WebP images
//!
//! [Lossless spec](https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification)

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) use avx2::{dispatch_alpha, rgb_from_rgba};
mod huffman;
#[cfg(target_arch = "aarch64")]
mod neon;
mod reverse_transform;
#[cfg(target_arch = "x86_64")]
mod sse2;
#[cfg(target_arch = "aarch64")]
use neon as simd;
#[cfg(target_arch = "x86_64")]
use sse2 as simd;

use std::io::{self, BufRead};
use std::mem;

use super::{CODE_LENGTH_CODES, CODE_LENGTH_CODE_ORDER, DISTANCE_MAP};
use crate::decoder::DecodingError;
use huffman::HuffmanTree;
use reverse_transform::{
    apply_color_indexing_transform, apply_color_transform, apply_predictor_transform,
    apply_subtract_green_transform, TransformType,
};

const GREEN: usize = 0;
const RED: usize = 1;
const BLUE: usize = 2;
const ALPHA: usize = 3;
const DIST: usize = 4;

const HUFFMAN_CODES_PER_META_CODE: usize = 5;

type HuffmanCodeGroup = [HuffmanTree; HUFFMAN_CODES_PER_META_CODE];

const ALPHABET_SIZE: [u16; HUFFMAN_CODES_PER_META_CODE] = [256 + 24, 256, 256, 256, 40];

#[inline]
pub(crate) fn subsample_size(size: u16, bits: u8) -> u16 {
    ((u32::from(size) + (1u32 << bits) - 1) >> bits)
        .try_into()
        .unwrap()
}

const NUM_TRANSFORM_TYPES: usize = 4;

//Decodes lossless WebP images
#[derive(Debug)]
pub(crate) struct LosslessDecoder<R> {
    bit_reader: BitReader<R>,
    transforms: [Option<TransformType>; NUM_TRANSFORM_TYPES],
    transform_order: Vec<u8>,
    width: u16,
    height: u16,
}

impl<R: BufRead> LosslessDecoder<R> {
    /// Create a new decoder
    pub(crate) const fn new(r: R) -> Self {
        Self {
            bit_reader: BitReader::new(r),
            transforms: [None, None, None, None],
            transform_order: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Reads the VP8L header of a `width`×`height` frame, failing where it gives another size,
    /// and its transforms and codes; the RGBA pixels follow a band at a time.
    pub(crate) fn frame(mut self, width: u32, height: u32) -> Result<LosslessBands<R>, DecodingError> {
        let signature = self.bit_reader.read_bits::<u8>(8)?;
        if signature != 0x2f {
            return Err(DecodingError::LosslessSignatureInvalid(signature));
        }

        self.width = self.bit_reader.read_bits::<u16>(14)? + 1;
        self.height = self.bit_reader.read_bits::<u16>(14)? + 1;
        if u32::from(self.width) != width || u32::from(self.height) != height {
            return Err(DecodingError::InconsistentImageSizes);
        }

        let _alpha_used = self.bit_reader.read_bits::<u8>(1)?;
        let version_num = self.bit_reader.read_bits::<u8>(3)?;
        if version_num != 0 {
            return Err(DecodingError::VersionNumberInvalid(version_num));
        }

        let xsize = self.read_transforms()?;
        let color_cache = self.read_color_cache()?.map(ColorCache::new);
        let huffman_info = self.read_huffman_codes(true, xsize, self.height, color_cache)?;
        Ok(LosslessBands::new(self, xsize, huffman_info))
    }

    /// Reads the headerless stream of an ALPH chunk, the green channel of a `width`×`height`
    /// image, up to its pixels, which [`AlphaBands`] decodes a band of rows at a time.
    pub(crate) fn alpha(mut self, width: u16, height: u16) -> Result<AlphaBands<R>, DecodingError> {
        self.width = width;
        self.height = height;
        let xsize = self.read_transforms()?;
        let color_cache = self.read_color_cache()?.map(ColorCache::new);
        let huffman_info = self.read_huffman_codes(true, xsize, height, color_cache)?;
        let indices_only = self.transform_order == [3]
            && huffman_info.color_cache.is_none()
            && huffman_info.huffman_code_groups.iter().all(|g| g[RED].is_single_node() && g[BLUE].is_single_node() && g[ALPHA].is_single_node());
        if !indices_only {
            return Ok(AlphaBands::Argb {
                bands: LosslessBands::new(self, xsize, huffman_info),
                greens: Vec::new(),
                first: 0,
            });
        }
        let Some(TransformType::ColorIndexingTransform { table_size, table_data }) = &self.transforms[3] else {
            unreachable!()
        };
        let mut greens = [0u8; 256];
        for (g, c) in greens.iter_mut().zip(table_data.chunks_exact(4)) {
            *g = c[1];
        }
        let bits = match table_size {
            ..=2 => 3,
            3..=4 => 2,
            5..=16 => 1,
            _ => 0,
        };
        let packed = usize::from(xsize) * usize::from(height);
        let band = usize::from(xsize) * BAND_ROWS.min(usize::from(height));
        Ok(AlphaBands::Indices(Box::new(AlphaIndices {
            decoder: self,
            huffman: huffman_info,
            window: Window::new(packed, band),
            index: 0,
            greens,
            bits,
            xsize: usize::from(xsize),
        })))
    }

    /// A transform's or entropy image's `xsize`×`ysize` RGBA pixels, decoded whole.
    fn decode_image_stream(&mut self, xsize: u16, ysize: u16) -> Result<Vec<u8>, DecodingError> {
        let color_cache = self.read_color_cache()?.map(ColorCache::new);
        let huffman_info = self.read_huffman_codes(false, xsize, ysize, color_cache)?;
        let pixels = usize::from(xsize) * usize::from(ysize);
        let mut state = PixelState::new(huffman_info, xsize, pixels);
        let mut window = Window::new(pixels, pixels);
        self.decode_pixels(&mut state, &mut window, pixels)?;
        Ok(window.data)
    }

    /// Reads transforms and their data from the bitstream
    fn read_transforms(&mut self) -> Result<u16, DecodingError> {
        let mut xsize = self.width;

        while self.bit_reader.read_bits::<u8>(1)? == 1 {
            let transform_type_val = self.bit_reader.read_bits::<u8>(2)?;

            if self.transforms[usize::from(transform_type_val)].is_some() {
                //can only have one of each transform, error
                return Err(DecodingError::TransformError);
            }

            self.transform_order.push(transform_type_val);

            let transform_type = match transform_type_val {
                0 => {
                    //predictor

                    let size_bits = self.bit_reader.read_bits::<u8>(3)? + 2;

                    let block_xsize = subsample_size(xsize, size_bits);
                    let block_ysize = subsample_size(self.height, size_bits);

                    let predictor_data = self.decode_image_stream(block_xsize, block_ysize)?;

                    TransformType::PredictorTransform {
                        size_bits,
                        predictor_data,
                    }
                }
                1 => {
                    //color transform

                    let size_bits = self.bit_reader.read_bits::<u8>(3)? + 2;

                    let block_xsize = subsample_size(xsize, size_bits);
                    let block_ysize = subsample_size(self.height, size_bits);

                    let transform_data = self.decode_image_stream(block_xsize, block_ysize)?;

                    TransformType::ColorTransform {
                        size_bits,
                        transform_data,
                    }
                }
                2 => {
                    //subtract green

                    TransformType::SubtractGreen
                }
                3 => {
                    let color_table_size = self.bit_reader.read_bits::<u16>(8)? + 1;

                    let mut color_map = self.decode_image_stream(color_table_size, 1)?;

                    let bits = if color_table_size <= 2 {
                        3
                    } else if color_table_size <= 4 {
                        2
                    } else if color_table_size <= 16 {
                        1
                    } else {
                        0
                    };
                    xsize = subsample_size(xsize, bits);

                    Self::adjust_color_map(&mut color_map);

                    TransformType::ColorIndexingTransform {
                        table_size: color_table_size,
                        table_data: color_map,
                    }
                }
                _ => unreachable!(),
            };

            self.transforms[usize::from(transform_type_val)] = Some(transform_type);
        }

        Ok(xsize)
    }

    /// Adjusts the color map since it's subtraction coded
    fn adjust_color_map(color_map: &mut [u8]) {
        for i in 4..color_map.len() {
            color_map[i] = color_map[i].wrapping_add(color_map[i - 4]);
        }
    }

    /// Reads huffman codes associated with an image
    #[inline(never)]
    fn read_huffman_codes(
        &mut self,
        read_meta: bool,
        xsize: u16,
        ysize: u16,
        color_cache: Option<ColorCache>,
    ) -> Result<HuffmanInfo, DecodingError> {
        let mut num_huff_groups = 1u32;

        let mut huffman_bits = 0;
        let mut huffman_xsize = 1;
        let mut huffman_ysize = 1;
        let mut entropy_image = Vec::new();

        if read_meta && self.bit_reader.read_bits::<u8>(1)? == 1 {
            //meta huffman codes
            huffman_bits = self.bit_reader.read_bits::<u8>(3)? + 2;
            huffman_xsize = subsample_size(xsize, huffman_bits);
            huffman_ysize = subsample_size(ysize, huffman_bits);

            let data = self.decode_image_stream(huffman_xsize, huffman_ysize)?;

            entropy_image = data
                .chunks_exact(4)
                .map(|pixel| {
                    let meta_huff_code = (u16::from(pixel[0]) << 8) | u16::from(pixel[1]);
                    if u32::from(meta_huff_code) >= num_huff_groups {
                        num_huff_groups = u32::from(meta_huff_code) + 1;
                    }
                    meta_huff_code
                })
                .collect::<Vec<u16>>();
        }

        let mut hufftree_groups = Vec::new();

        for _i in 0..num_huff_groups {
            let mut group: HuffmanCodeGroup = Default::default();
            for j in 0..HUFFMAN_CODES_PER_META_CODE {
                let mut alphabet_size = ALPHABET_SIZE[j];
                if j == 0 {
                    if let Some(color_cache) = color_cache.as_ref() {
                        alphabet_size += 1 << color_cache.color_cache_bits;
                    }
                }

                let tree = self.read_huffman_code(alphabet_size)?;
                group[j] = tree;
            }
            hufftree_groups.push(group);
        }

        let huffman_mask = if huffman_bits == 0 {
            !0
        } else {
            (1 << huffman_bits) - 1
        };

        let info = HuffmanInfo {
            xsize: huffman_xsize,
            _ysize: huffman_ysize,
            color_cache,
            image: entropy_image,
            bits: huffman_bits,
            mask: huffman_mask,
            huffman_code_groups: hufftree_groups,
        };

        Ok(info)
    }

    /// Decodes and returns a single huffman tree
    fn read_huffman_code(&mut self, alphabet_size: u16) -> Result<HuffmanTree, DecodingError> {
        let simple = self.bit_reader.read_bits::<u8>(1)? == 1;

        if simple {
            let num_symbols = self.bit_reader.read_bits::<u8>(1)? + 1;

            let is_first_8bits = self.bit_reader.read_bits::<u8>(1)?;
            let zero_symbol = self.bit_reader.read_bits::<u16>(1 + 7 * is_first_8bits)?;
            let one_symbol = match num_symbols {
                2 => Some(self.bit_reader.read_bits::<u16>(8)?),
                _ => None,
            };
            // As libwebp: symbols past the alphabet are left out, a repeated one counts once.
            let mut symbols = [Some(zero_symbol), one_symbol].into_iter().flatten().filter(|&s| s < alphabet_size);
            match (symbols.next(), symbols.next()) {
                (None, _) => Err(DecodingError::BitStreamError),
                (Some(a), Some(b)) if a != b => Ok(HuffmanTree::build_two_node(a.min(b), a.max(b))),
                (Some(a), _) => Ok(HuffmanTree::build_single_node(a)),
            }
        } else {
            let mut code_length_code_lengths = vec![0; CODE_LENGTH_CODES];

            let num_code_lengths = 4 + self.bit_reader.read_bits::<usize>(4)?;
            for i in 0..num_code_lengths {
                code_length_code_lengths[CODE_LENGTH_CODE_ORDER[i]] =
                    self.bit_reader.read_bits(3)?;
            }

            let new_code_lengths =
                self.read_huffman_code_lengths(code_length_code_lengths, alphabet_size)?;

            HuffmanTree::build_implicit(new_code_lengths)
        }
    }

    /// Reads huffman code lengths
    fn read_huffman_code_lengths(
        &mut self,
        code_length_code_lengths: Vec<u16>,
        num_symbols: u16,
    ) -> Result<Vec<u16>, DecodingError> {
        let table = HuffmanTree::build_implicit(code_length_code_lengths)?;

        let mut max_symbol = if self.bit_reader.read_bits::<u8>(1)? == 1 {
            let length_nbits = 2 + 2 * self.bit_reader.read_bits::<u8>(3)?;
            let max_minus_two = self.bit_reader.read_bits::<u16>(length_nbits)?;
            if max_minus_two > num_symbols - 2 {
                return Err(DecodingError::BitStreamError);
            }
            2 + max_minus_two
        } else {
            num_symbols
        };

        let mut code_lengths = vec![0; usize::from(num_symbols)];
        let mut prev_code_len = 8; //default code length

        let mut symbol = 0;
        while symbol < num_symbols {
            if max_symbol == 0 {
                break;
            }
            max_symbol -= 1;

            self.bit_reader.fill()?;
            let code_len = table.read_symbol(&mut self.bit_reader)?;

            if code_len < 16 {
                code_lengths[usize::from(symbol)] = code_len;
                symbol += 1;
                if code_len != 0 {
                    prev_code_len = code_len;
                }
            } else {
                let use_prev = code_len == 16;
                let slot = code_len - 16;
                let extra_bits = match slot {
                    0 => 2,
                    1 => 3,
                    2 => 7,
                    _ => return Err(DecodingError::BitStreamError),
                };
                let repeat_offset = match slot {
                    0 | 1 => 3,
                    2 => 11,
                    _ => return Err(DecodingError::BitStreamError),
                };

                let mut repeat = self.bit_reader.read_bits::<u16>(extra_bits)? + repeat_offset;

                if symbol + repeat > num_symbols {
                    return Err(DecodingError::BitStreamError);
                }

                let length = if use_prev { prev_code_len } else { 0 };
                while repeat > 0 {
                    repeat -= 1;
                    code_lengths[usize::from(symbol)] = length;
                    symbol += 1;
                }
            }
        }

        Ok(code_lengths)
    }

    /// Decodes pixels into `window` until pixel `until` is decoded, as libwebp's
    /// `DecodeImageData` does, a backward reference or a run reaching past it.
    fn decode_pixels(&mut self, st: &mut PixelState, window: &mut Window, until: usize) -> Result<(), DecodingError> {
        window.make_room(st.index, (until + MAX_COPY).min(st.num_values));
        let (base, last) = (window.base, window.base + window.cap);
        self.decode_pixels_into(st, &mut window.data, base, last, until)
    }

    /// [`decode_pixels`](Self::decode_pixels) into `data`, which holds pixels `base..last`.
    fn decode_pixels_into(
        &mut self,
        st: &mut PixelState,
        data: &mut [u8],
        base: usize,
        last: usize,
        until: usize,
    ) -> Result<(), DecodingError> {
        let width = usize::from(st.width);
        let num_values = st.num_values;
        let huffman_info = &mut st.huffman;
        let mut tree = &huffman_info.huffman_code_groups[st.huff_index];
        let mut index = st.index;
        let mut next_block_start = st.next_block_start;

        while index < until {
            if self.bit_reader.bit_pos < LBITS && !self.bit_reader.eos {
                let at_entry = index;
                let fast = Fast { width, num_values, base, last, until };
                (index, next_block_start) =
                    fast.run(&mut self.bit_reader, huffman_info, &mut st.huff_index, data, index, next_block_start)?;
                tree = &huffman_info.huffman_code_groups[st.huff_index];
                if index >= until || index != at_entry {
                    continue;
                }
            }
            self.bit_reader.fill()?;
            let at = (index - base) * 4;

            if index >= next_block_start {
                let x = index % width;
                let y = index / width;
                next_block_start = (x | usize::from(huffman_info.mask)).min(width - 1) + y * width + 1;

                st.huff_index = huffman_info.get_huff_index(x as u16, y as u16);
                tree = &huffman_info.huffman_code_groups[st.huff_index];

                // Fast path: If all the codes each contain only a single
                // symbol, then the pixel data isn't written to the bitstream
                // and we can just fill the output buffer with the symbol
                // directly.
                if tree[..4].iter().all(|t| t.is_single_node()) {
                    let code = tree[GREEN].read_symbol(&mut self.bit_reader)?;
                    if code < 256 {
                        let n = if huffman_info.bits == 0 {
                            num_values - index
                        } else {
                            next_block_start - index
                        }
                        .min(until - index);

                        let red = tree[RED].read_symbol(&mut self.bit_reader)?;
                        let blue = tree[BLUE].read_symbol(&mut self.bit_reader)?;
                        let alpha = tree[ALPHA].read_symbol(&mut self.bit_reader)?;
                        let value = u32::from_le_bytes([red as u8, code as u8, blue as u8, alpha as u8]);

                        for px in data[at..at + n * 4].chunks_exact_mut(4) {
                            px.copy_from_slice(&value.to_le_bytes());
                        }

                        if let Some(color_cache) = huffman_info.color_cache.as_mut() {
                            color_cache.insert(value);
                        }

                        index += n;
                        continue;
                    }
                }
            }

            let code = tree[GREEN].read_symbol(&mut self.bit_reader)?;

            //check code
            if code < 256 {
                //literal, so just use huffman codes and read as argb
                // The end of the stream is checked once for red, blue and alpha: past it the
                // reads give garbage and the check fails as the first read past it would.
                let red = tree[RED].read_symbol_unchecked(&mut self.bit_reader) as u8;
                // Blue and alpha take up to 30 bits, which the window still holds below this; a
                // fill there cannot reach the end of the stream, so skipping it changes nothing.
                if self.bit_reader.bit_pos > LBITS - 30 {
                    self.bit_reader.fill()?;
                }
                let blue = tree[BLUE].read_symbol_unchecked(&mut self.bit_reader) as u8;
                let alpha = tree[ALPHA].read_symbol_unchecked(&mut self.bit_reader) as u8;
                self.bit_reader.check()?;

                let pixel = u32::from_le_bytes([red, code as u8, blue, alpha]);
                data[at..at + 4].copy_from_slice(&pixel.to_le_bytes());
                if let Some(color_cache) = huffman_info.color_cache.as_mut() {
                    color_cache.insert(pixel);
                }
                index += 1;
            } else if code < 256 + 24 {
                //backward reference, so go back and use that to add image data
                let length_symbol = code - 256;
                let length = Self::get_copy_distance(&mut self.bit_reader, length_symbol)?;

                let dist_symbol = tree[DIST].read_symbol(&mut self.bit_reader)?;
                // Two 15-bit codes and 18 extra bits outrun one fill, as libwebp refills here.
                self.bit_reader.fill()?;
                let dist_code = Self::get_copy_distance(&mut self.bit_reader, dist_symbol)?;
                let dist = Self::plane_code_to_distance(st.width, dist_code);

                if index < dist || num_values - index < length {
                    return Err(DecodingError::BitStreamError);
                }

                if dist == 1 {
                    let value: [u8; 4] = data[at - 4..][..4].try_into().unwrap();
                    for px in data[at..at + length * 4].chunks_exact_mut(4) {
                        px.copy_from_slice(&value);
                    }
                } else {
                    if index + length + 3 <= last {
                        let start = at - dist * 4;
                        data.copy_within(start..start + 16, at);

                        if length > 4 || dist < 4 {
                            for i in (0..length * 4).step_by((dist * 4).min(16)).skip(1) {
                                data.copy_within(start + i..start + i + 16, at + i);
                            }
                        }
                    } else {
                        for i in at..at + length * 4 {
                            data[i] = data[i - dist * 4];
                        }
                    }

                    if let Some(color_cache) = huffman_info.color_cache.as_mut() {
                        for pixel in data[at..][..length * 4].chunks_exact(4) {
                            color_cache.insert(u32::from_le_bytes(pixel.try_into().unwrap()));
                        }
                    }
                }
                index += length;
            } else {
                //color cache, so use previously stored pixels to get this pixel
                let color_cache = huffman_info
                    .color_cache
                    .as_mut()
                    .ok_or(DecodingError::BitStreamError)?;
                let color = color_cache.lookup((code - 280).into());
                data[at..][..4].copy_from_slice(&color.to_le_bytes());
                color_cache.insert(color);
                index += 1;

                if index < next_block_start {
                    self.bit_reader.fill()?;
                    if let Some((bits, code)) = tree[GREEN].peek_symbol(&self.bit_reader) {
                        if code >= 280 {
                            self.bit_reader.consume(bits)?;
                            let color = color_cache.lookup((code - 280).into());
                            data[at + 4..][..4].copy_from_slice(&color.to_le_bytes());
                            color_cache.insert(color);
                            index += 1;
                        }
                    }
                }
            }
        }

        st.index = index;
        st.next_block_start = next_block_start;
        Ok(())
    }

    /// libwebp's `DecodeAlphaData` up to index `until`: green codes only, a byte each, the end of
    /// the stream checked after each code, and an end inside the last code accepted.
    fn decode_alpha_8b(
        &mut self,
        width: u16,
        height: u16,
        huffman_info: &HuffmanInfo,
        window: &mut Window<1>,
        index: &mut usize,
        until: usize,
    ) -> Result<(), DecodingError> {
        let end = usize::from(width) * usize::from(height);
        window.make_room(*index, (until + MAX_COPY).min(end));
        let base = window.base;
        let data = &mut window.data[..];
        let mut pos = *index;
        while pos < until {
            if self.bit_reader.bit_pos < LBITS && !self.bit_reader.eos {
                let at_entry = pos;
                pos = alpha_fast(&mut self.bit_reader, huffman_info, usize::from(width), end, data, base, pos, until)?;
                if pos >= until || pos != at_entry {
                    continue;
                }
            }
            let (x, y) = (pos % usize::from(width), pos / usize::from(width));
            let tree = &huffman_info.huffman_code_groups[huffman_info.get_huff_index(x as u16, y as u16)];
            self.bit_reader.fill()?;
            let code = tree[GREEN].read_symbol_unchecked(&mut self.bit_reader);
            if code < 256 {
                data[pos - base] = code as u8;
                pos += 1;
            } else if code < 256 + 24 {
                let length = Self::copy_distance_unchecked(&mut self.bit_reader, code - 256)?;
                let dist_symbol = tree[DIST].read_symbol_unchecked(&mut self.bit_reader);
                self.bit_reader.fill()?;
                let dist_code = Self::copy_distance_unchecked(&mut self.bit_reader, dist_symbol)?;
                let dist = Self::plane_code_to_distance(width, dist_code);
                if pos < dist || end - pos < length {
                    return Err(DecodingError::BitStreamError);
                }
                for i in pos - base..pos - base + length {
                    data[i] = data[i - dist];
                }
                pos += length;
            } else {
                return Err(DecodingError::BitStreamError);
            }
            if self.bit_reader.is_eos()? {
                self.bit_reader.eos = true;
                break;
            }
        }
        *index = pos;
        if self.bit_reader.eos && pos < end {
            return Err(DecodingError::BitStreamError);
        }
        Ok(())
    }

    /// `GetCopyDistance` with libwebp's reads past the end.
    fn copy_distance_unchecked(bit_reader: &mut BitReader<R>, prefix_code: u16) -> Result<usize, DecodingError> {
        if prefix_code < 4 {
            return Ok(usize::from(prefix_code + 1));
        }
        let extra_bits = ((prefix_code - 2) >> 1) as u8;
        let offset = (2 + (usize::from(prefix_code) & 1)) << extra_bits;
        Ok(offset + bit_reader.read_bits_raw(extra_bits)? as usize + 1)
    }

    /// Reads color cache data from the bitstream
    fn read_color_cache(&mut self) -> Result<Option<u8>, DecodingError> {
        if self.bit_reader.read_bits::<u8>(1)? == 1 {
            let code_bits = self.bit_reader.read_bits::<u8>(4)?;

            if !(1..=11).contains(&code_bits) {
                return Err(DecodingError::InvalidColorCacheBits(code_bits));
            }

            Ok(Some(code_bits))
        } else {
            Ok(None)
        }
    }

    /// Gets the copy distance from the prefix code and bitstream
    fn get_copy_distance(
        bit_reader: &mut BitReader<R>,
        prefix_code: u16,
    ) -> Result<usize, DecodingError> {
        if prefix_code < 4 {
            return Ok(usize::from(prefix_code + 1));
        }
        let extra_bits: u8 = ((prefix_code - 2) >> 1).try_into().unwrap();
        let offset = (2 + (usize::from(prefix_code) & 1)) << extra_bits;

        let bits = bit_reader.read_bits::<u32>(extra_bits)? as usize;
        Ok(offset + bits + 1)
    }

    /// Gets distance to pixel
    fn plane_code_to_distance(xsize: u16, plane_code: usize) -> usize {
        if plane_code > 120 {
            plane_code - 120
        } else {
            let (xoffset, yoffset) = DISTANCE_MAP[plane_code - 1];

            let dist = i32::from(xoffset) + i32::from(yoffset) * i32::from(xsize);
            if dist < 1 {
                return 1;
            }
            dist.try_into().unwrap()
        }
    }
}

/// The pixel loop with the bits in locals, which the pixel stores could alias and keep in memory
/// otherwise. It stops 16 bytes short of the end of the read-ahead, so every fill is one load and
/// no code runs past the stream: the bits and the end of the stream stay libwebp's. It also stops
/// at `until`, and ahead of a group of single-symbol codes, which the caller fills as runs.
struct Fast {
    width: usize,
    num_values: usize,
    base: usize,
    last: usize,
    until: usize,
}

/// `VP8LFillBitWindow` with at least eight bytes left.
#[inline(always)]
fn fill(buf: &[u8], val: &mut u64, pos: &mut u32, at: &mut usize) {
    let w = u64::from_le_bytes(buf[*at..*at + 8].try_into().unwrap());
    let shift = *pos & !7;
    *val = (*val >> shift) | (w << 1) << (LBITS - 1 - shift);
    *at += (shift / 8) as usize;
    *pos &= 7;
}

/// `GetCopyDistance`, up to 18 extra bits.
#[inline(always)]
fn prefix(val: u64, pos: &mut u32, symbol: u16) -> usize {
    if symbol < 4 {
        return usize::from(symbol) + 1;
    }
    let extra = u32::from(symbol - 2) >> 1;
    let offset = (2 + (usize::from(symbol) & 1)) << extra;
    let bits = (val >> *pos) as usize & ((1 << extra) - 1);
    *pos += extra;
    offset + bits + 1
}

/// [`decode_alpha_8b`](LosslessDecoder::decode_alpha_8b) with the bits in locals, short of the
/// end of the read-ahead as [`Fast`], where no code reaches the end of the stream. Two literals
/// are read per fill: the second is only looked up, and taken if it is a literal too.
#[allow(clippy::too_many_arguments)]
fn alpha_fast<R: BufRead>(
    br: &mut BitReader<R>,
    huffman: &HuffmanInfo,
    width: usize,
    end: usize,
    data: &mut [u8],
    base: usize,
    mut index: usize,
    until: usize,
) -> Result<usize, DecodingError> {
    let (mut val, mut pos, mut at) = (br.val, br.bit_pos, br.at);
    let buf = &br.buf[..];
    let Some(at_limit) = buf.len().checked_sub(17) else {
        return Ok(index);
    };
    let mut group = &huffman.huffman_code_groups[0];
    let mut stop = index;
    let result = loop {
        if index >= stop || at > at_limit {
            if index >= until || at > at_limit {
                break Ok(());
            }
            let (x, y) = (index % width, index / width);
            group = &huffman.huffman_code_groups[huffman.get_huff_index(x as u16, y as u16)];
            stop = ((x | usize::from(huffman.mask)).min(width - 1) + y * width + 1).min(until);
        }
        fill(buf, &mut val, &mut pos, &mut at);
        let (code, n) = group[GREEN].symbol(val >> pos);
        pos += n;
        if code < 256 {
            data[index - base] = code as u8;
            index += 1;
            if index < stop {
                let (code, n) = group[GREEN].symbol(val >> pos);
                if code < 256 {
                    pos += n;
                    data[index - base] = code as u8;
                    index += 1;
                }
            }
        } else if code < 256 + 24 {
            let length = prefix(val, &mut pos, code - 256);
            let (dist_symbol, n) = group[DIST].symbol(val >> pos);
            pos += n;
            fill(buf, &mut val, &mut pos, &mut at);
            let dist_code = prefix(val, &mut pos, dist_symbol);
            let dist = LosslessDecoder::<R>::plane_code_to_distance(width as u16, dist_code);
            if index < dist || end - index < length {
                break Err(DecodingError::BitStreamError);
            }
            let o = index - base;
            if dist == 1 {
                let value = data[o - 1];
                data[o..o + length].fill(value);
            } else if dist >= length {
                data.copy_within(o - dist..o - dist + length, o);
            } else {
                for i in o..o + length {
                    data[i] = data[i - dist];
                }
            }
            index += length;
        } else {
            break Err(DecodingError::BitStreamError);
        }
    };
    (br.val, br.bit_pos, br.at) = (val, pos, at);
    result.map(|()| index)
}

impl Fast {
    #[inline(always)]
    fn run<R: BufRead>(
        &self,
        br: &mut BitReader<R>,
        huffman: &mut HuffmanInfo,
        huff_index: &mut usize,
        data: &mut [u8],
        mut index: usize,
        mut next_block_start: usize,
    ) -> Result<(usize, usize), DecodingError> {
        let HuffmanInfo { xsize, color_cache, image, bits, mask, huffman_code_groups, .. } = huffman;
        let mut group = &huffman_code_groups[*huff_index];
        let (mut val, mut pos, mut at) = (br.val, br.bit_pos, br.at);
        let buf = &br.buf[..];
        let Some(at_limit) = buf.len().checked_sub(17) else {
            return Ok((index, next_block_start));
        };
        let mut stop = next_block_start.min(self.until);
        let result = loop {
            if index >= stop || at > at_limit {
                if index >= self.until || at > at_limit {
                    break Ok(());
                }
                let (x, y) = (index % self.width, index / self.width);
                let group_index = if *bits == 0 {
                    0
                } else {
                    usize::from(image[(y >> *bits) * usize::from(*xsize) + (x >> *bits)])
                };
                let tree = &huffman_code_groups[group_index];
                if tree[..4].iter().all(|t| t.is_single_node()) {
                    break Ok(());
                }
                next_block_start = (x | usize::from(*mask)).min(self.width - 1) + y * self.width + 1;
                stop = next_block_start.min(self.until);
                *huff_index = group_index;
                group = tree;
            }
            fill(buf, &mut val, &mut pos, &mut at);
            let (code, n) = group[GREEN].symbol(val >> pos);
            pos += n;
            let o = (index - self.base) * 4;
            if code < 256 {
                let (red, n) = group[RED].symbol(val >> pos);
                pos += n;
                // Blue and alpha take up to 30 bits, and the next fill wants fewer than 64 read.
                if pos > LBITS - 31 {
                    fill(buf, &mut val, &mut pos, &mut at);
                }
                let (blue, n) = group[BLUE].symbol(val >> pos);
                pos += n;
                let (alpha, n) = group[ALPHA].symbol(val >> pos);
                pos += n;
                let pixel = u32::from_le_bytes([red as u8, code as u8, blue as u8, alpha as u8]);
                data[o..o + 4].copy_from_slice(&pixel.to_le_bytes());
                if let Some(cache) = color_cache.as_mut() {
                    cache.insert(pixel);
                }
                index += 1;
            } else if code < 256 + 24 {
                match self.copy::<R>(buf, group, data, color_cache, index, code, (val, pos, at)) {
                    Ok((length, bits)) => {
                        (val, pos, at) = bits;
                        index += length;
                    }
                    Err(e) => break Err(e),
                }
            } else {
                let Some(cache) = color_cache.as_mut() else {
                    break Err(DecodingError::BitStreamError);
                };
                let color = cache.lookup(usize::from(code - 280));
                data[o..o + 4].copy_from_slice(&color.to_le_bytes());
                cache.insert(color);
                index += 1;
            }
        };
        (br.val, br.bit_pos, br.at) = (val, pos, at);
        result.map(|()| (index, next_block_start))
    }

    /// A backward reference of length code `code`: its length, and the bits after it. Out of line,
    /// so the literal loop keeps its registers.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn copy<R: BufRead>(
        &self,
        buf: &[u8],
        group: &HuffmanCodeGroup,
        data: &mut [u8],
        color_cache: &mut Option<ColorCache>,
        index: usize,
        code: u16,
        (mut val, mut pos, mut at): (u64, u32, usize),
    ) -> Result<(usize, (u64, u32, usize)), DecodingError> {
        let length = prefix(val, &mut pos, code - 256);
        let (dist_symbol, n) = group[DIST].symbol(val >> pos);
        pos += n;
        fill(buf, &mut val, &mut pos, &mut at);
        let dist_code = prefix(val, &mut pos, dist_symbol);
        let dist = LosslessDecoder::<R>::plane_code_to_distance(self.width as u16, dist_code);
        if index < dist || self.num_values - index < length {
            return Err(DecodingError::BitStreamError);
        }
        let o = (index - self.base) * 4;
        if dist == 1 {
            let value: [u8; 4] = data[o - 4..][..4].try_into().unwrap();
            for px in data[o..o + length * 4].chunks_exact_mut(4) {
                px.copy_from_slice(&value);
            }
        } else {
            if index + length + 3 <= self.last {
                let start = o - dist * 4;
                data.copy_within(start..start + 16, o);
                if length > 4 || dist < 4 {
                    for i in (0..length * 4).step_by((dist * 4).min(16)).skip(1) {
                        data.copy_within(start + i..start + i + 16, o + i);
                    }
                }
            } else {
                for i in o..o + length * 4 {
                    data[i] = data[i - dist * 4];
                }
            }
            if let Some(cache) = color_cache.as_mut() {
                cache.insert_all(&data[o..][..length * 4]);
            }
        }
        Ok((length, (val, pos, at)))
    }
}

#[derive(Debug, Clone)]
/// The entropy-decoded pixels a backward reference can still reach: libwebp's distance codes
/// reach `WINDOW - 120` pixels back, the longest run `MAX_COPY` pixels on. Pixel `i` sits at
/// `i - base`; the last `WINDOW` pixels slide to the front when the next band would not fit.
struct Window<const B: usize = 4> {
    data: Vec<u8>,
    base: usize,
    cap: usize,
}

const WINDOW: usize = 1 << 20;
const MAX_COPY: usize = 4096;

impl<const B: usize> Window<B> {
    /// Allocated by the first [`make_room`](Self::make_room), which a whole-frame decode never calls.
    fn new(num_values: usize, band: usize) -> Self {
        let cap = num_values.min(WINDOW + WINDOW.max(band + MAX_COPY));
        Self { data: Vec::new(), base: 0, cap }
    }

    fn make_room(&mut self, index: usize, need: usize) {
        if self.data.is_empty() {
            self.data = vec![0; self.cap * B];
        }
        if need <= self.base + self.cap || index < WINDOW {
            return;
        }
        let keep = index - WINDOW;
        self.data.copy_within((keep - self.base) * B..(index - self.base) * B, 0);
        self.base = keep;
    }

    fn rows(&self, from: usize, to: usize) -> &[u8] {
        &self.data[(from - self.base) * B..(to - self.base) * B]
    }
}

/// Where a pixel decode stands between bands.
struct PixelState {
    huffman: HuffmanInfo,
    width: u16,
    num_values: usize,
    index: usize,
    next_block_start: usize,
    huff_index: usize,
}

impl PixelState {
    fn new(huffman: HuffmanInfo, width: u16, num_values: usize) -> Self {
        let huff_index = huffman.get_huff_index(0, 0);
        Self { huffman, width, num_values, index: 0, next_block_start: 0, huff_index }
    }
}

/// The main image of a VP8L stream decoded a band of rows at a time as libwebp's `ProcessRows`
/// hands them on: the entropy-coded pixels kept in a [`Window`], the inverse transforms run over
/// each band in a cache holding the band and, before it, the previous row the predictor needs.
pub(crate) struct LosslessBands<R> {
    decoder: LosslessDecoder<R>,
    state: PixelState,
    window: Window,
    cache: Vec<u8>,
    top: Vec<u8>,
    row: u16,
}

pub(crate) const BAND_ROWS: usize = 16;

impl<R: BufRead> LosslessBands<R> {
    fn new(decoder: LosslessDecoder<R>, xsize: u16, huffman: HuffmanInfo) -> Self {
        let (width, height) = (usize::from(decoder.width), usize::from(decoder.height));
        let pixels = usize::from(xsize) * height;
        let band = usize::from(xsize) * BAND_ROWS.min(height);
        Self {
            decoder,
            state: PixelState::new(huffman, xsize, pixels),
            window: Window::new(pixels, band),
            cache: vec![0; width * 4 * (1 + BAND_ROWS.min(height))],
            top: vec![0; width * 4],
            row: 0,
        }
    }

    /// The next band's first row and its RGBA rows, or `None` after the last.
    pub(crate) fn next_band(&mut self) -> Result<Option<(u16, &[u8])>, DecodingError> {
        let (width, height) = (usize::from(self.decoder.width), usize::from(self.decoder.height));
        let y0 = usize::from(self.row);
        if y0 >= height {
            return Ok(None);
        }
        let rows = BAND_ROWS.min(height - y0);
        let xsize = usize::from(self.state.width);
        self.decoder.decode_pixels(&mut self.state, &mut self.window, (y0 + rows) * xsize)?;
        let head = width * 4;
        self.cache[head..head + rows * xsize * 4].copy_from_slice(self.window.rows(y0 * xsize, (y0 + rows) * xsize));
        self.transform(y0, rows);
        self.row += rows as u16;
        Ok(Some((y0 as u16, &self.cache[head..head + rows * width * 4])))
    }

    /// Whether no band has been decoded yet.
    pub(crate) fn at_start(&self) -> bool {
        self.row == 0
    }

    /// Decodes the whole frame into `out`, its `width`×`height` RGBA pixels, and undoes the
    /// transforms in place there, as upstream image-webp does: no window and no band copies.
    /// Only before the first band.
    pub(crate) fn read_into(&mut self, out: &mut [u8]) -> Result<(), DecodingError> {
        debug_assert_eq!(self.row, 0);
        let (full, height) = (usize::from(self.decoder.width), usize::from(self.decoder.height));
        let pixels = self.state.num_values;
        self.decoder.decode_pixels_into(&mut self.state, &mut out[..pixels * 4], 0, pixels, pixels)?;
        let mut width = usize::from(self.state.width);
        for &t in self.decoder.transform_order.iter().rev() {
            let stride = width * 4;
            let image = &mut out[..height * stride];
            match self.decoder.transforms[usize::from(t)].as_ref().unwrap() {
                TransformType::PredictorTransform { size_bits, predictor_data } => {
                    // Row 0 has no row above it in `out`; it goes through the cache behind one.
                    let first = &mut self.cache[..2 * stride];
                    first[stride..].copy_from_slice(&image[..stride]);
                    apply_predictor_transform(first, width as u16, 0, 1, *size_bits, predictor_data);
                    image[..stride].copy_from_slice(&first[stride..]);
                    if height > 1 {
                        apply_predictor_transform(image, width as u16, 1, height - 1, *size_bits, predictor_data);
                    }
                }
                TransformType::ColorTransform { size_bits, transform_data } => {
                    apply_color_transform(image, width as u16, 0, *size_bits, transform_data);
                }
                TransformType::SubtractGreen => apply_subtract_green_transform(image),
                TransformType::ColorIndexingTransform { table_size, table_data } => {
                    width = full;
                    apply_color_indexing_transform(&mut out[..height * full * 4], full as u16, height as u16, *table_size, table_data);
                }
            }
        }
        self.row = height as u16;
        Ok(())
    }

    fn transform(&mut self, y0: usize, rows: usize) {
        let full = usize::from(self.decoder.width);
        let head = full * 4;
        let mut width = usize::from(self.state.width);
        for &t in self.decoder.transform_order.iter().rev() {
            let stride = width * 4;
            match self.decoder.transforms[usize::from(t)].as_ref().unwrap() {
                TransformType::PredictorTransform { size_bits, predictor_data } => {
                    let band = &mut self.cache[head - stride..head + rows * stride];
                    band[..stride].copy_from_slice(&self.top[..stride]);
                    apply_predictor_transform(band, width as u16, y0, rows, *size_bits, predictor_data);
                    self.top[..stride].copy_from_slice(&band[rows * stride..]);
                }
                TransformType::ColorTransform { size_bits, transform_data } => {
                    apply_color_transform(&mut self.cache[head..head + rows * stride], width as u16, y0, *size_bits, transform_data);
                }
                TransformType::SubtractGreen => apply_subtract_green_transform(&mut self.cache[head..head + rows * stride]),
                TransformType::ColorIndexingTransform { table_size, table_data } => {
                    width = full;
                    apply_color_indexing_transform(&mut self.cache[head..head + rows * full * 4], full as u16, rows as u16, *table_size, table_data);
                }
            }
        }
    }
}

/// The pixels of an ALPH chunk's lossless stream, decoded a band of rows at a time: as RGBA
/// whose green is kept, or as palette indices a byte at a time, as libwebp decodes them.
pub(crate) enum AlphaBands<R> {
    Argb {
        bands: LosslessBands<R>,
        greens: Vec<u8>,
        first: usize,
    },
    Indices(Box<AlphaIndices<R>>),
}

pub(crate) struct AlphaIndices<R> {
    decoder: LosslessDecoder<R>,
    huffman: HuffmanInfo,
    window: Window<1>,
    index: usize,
    greens: [u8; 256],
    bits: u8,
    xsize: usize,
}

impl<R: BufRead> AlphaBands<R> {
    /// Row `y`'s green bytes into `row`, the rows read top to bottom.
    pub(crate) fn read_row(&mut self, y: usize, row: &mut [u8]) -> Result<(), DecodingError> {
        let w = row.len();
        match self {
            AlphaBands::Argb { bands, greens, first } => {
                if y >= *first + greens.len() / w {
                    let (y0, band) = bands.next_band()?.ok_or(DecodingError::BitStreamError)?;
                    greens.resize(band.len() / 4, 0);
                    #[cfg(target_arch = "x86_64")]
                    let done = avx2::extract_green(band, greens);
                    #[cfg(not(target_arch = "x86_64"))]
                    let done = 0;
                    for (g, p) in greens[done..].iter_mut().zip(band[done * 4..].as_chunks::<4>().0) {
                        *g = p[1];
                    }
                    *first = usize::from(y0);
                }
                row.copy_from_slice(&greens[(y - *first) * w..][..w]);
            }
            AlphaBands::Indices(a) => {
                let (xsize, height) = (a.xsize, usize::from(a.decoder.height));
                if a.index < (y + 1) * xsize {
                    let until = (y + BAND_ROWS).min(height) * xsize;
                    let (width, huffman) = (xsize as u16, &a.huffman);
                    a.decoder.decode_alpha_8b(width, height as u16, huffman, &mut a.window, &mut a.index, until)?;
                }
                let packed = a.window.rows(y * xsize, (y + 1) * xsize);
                let (per_byte, mask) = (1 << a.bits, (1u16 << (8 >> a.bits)) - 1);
                for (x, g) in row.iter_mut().enumerate() {
                    let index = (u16::from(packed[x / per_byte]) >> ((x % per_byte) << (3 - a.bits))) & mask;
                    *g = a.greens[usize::from(index)];
                }
            }
        }
        Ok(())
    }
}

/// The whole-plane ALPH decode banded decoding replaced, kept as the reference it is held to.
#[cfg(test)]
impl<R: BufRead> LosslessDecoder<R> {
    pub(crate) fn decode_alpha_whole(mut self, width: u16, height: u16) -> Result<Vec<u8>, DecodingError> {
        self.width = width;
        self.height = height;
        let xsize = self.read_transforms()?;
        let color_cache = self.read_color_cache()?.map(ColorCache::new);
        let huffman_info = self.read_huffman_codes(true, xsize, height, color_cache)?;
        let pixels = usize::from(width) * usize::from(height);
        let indices_only = self.transform_order == [3]
            && huffman_info.color_cache.is_none()
            && huffman_info.huffman_code_groups.iter().all(|g| g[RED].is_single_node() && g[BLUE].is_single_node() && g[ALPHA].is_single_node());
        if !indices_only {
            let mut bands = LosslessBands::new(self, xsize, huffman_info);
            let mut plane = Vec::with_capacity(pixels);
            while let Some((_, rows)) = bands.next_band()? {
                plane.extend(rows.chunks_exact(4).map(|p| p[1]));
            }
            return Ok(plane);
        }
        let mut plane = vec![0; pixels];
        self.decode_alpha_8b_whole(xsize, height, &huffman_info, &mut plane)?;
        let Some(TransformType::ColorIndexingTransform { table_size, table_data }) = &self.transforms[3] else {
            unreachable!()
        };
        let mut greens = [0u8; 256];
        for (g, c) in greens.iter_mut().zip(table_data.chunks_exact(4)) {
            *g = c[1];
        }
        let bits = match table_size {
            ..=2 => 3,
            3..=4 => 2,
            5..=16 => 1,
            _ => 0,
        };
        let (w, xsize) = (usize::from(width), usize::from(xsize));
        let (per_byte, mask) = (1 << bits, (1u16 << (8 >> bits)) - 1);
        // Indices expand in place from the end: a row's pixels never land before its indices.
        for y in (0..usize::from(height)).rev() {
            for x in (0..w).rev() {
                let packed = plane[y * xsize + x / per_byte];
                let index = (u16::from(packed) >> ((x % per_byte) << (3 - bits))) & mask;
                plane[y * w + x] = greens[usize::from(index)];
            }
        }
        Ok(plane)
    }

    fn decode_alpha_8b_whole(
        &mut self,
        width: u16,
        height: u16,
        huffman_info: &HuffmanInfo,
        data: &mut [u8],
    ) -> Result<(), DecodingError> {
        let end = usize::from(width) * usize::from(height);
        let mut pos = 0;
        while pos < end {
            let (x, y) = (pos % usize::from(width), pos / usize::from(width));
            let tree = &huffman_info.huffman_code_groups[huffman_info.get_huff_index(x as u16, y as u16)];
            self.bit_reader.fill()?;
            let code = tree[GREEN].read_symbol_unchecked(&mut self.bit_reader);
            if code < 256 {
                data[pos] = code as u8;
                pos += 1;
            } else if code < 256 + 24 {
                let length = Self::copy_distance_unchecked(&mut self.bit_reader, code - 256)?;
                let dist_symbol = tree[DIST].read_symbol_unchecked(&mut self.bit_reader);
                self.bit_reader.fill()?;
                let dist_code = Self::copy_distance_unchecked(&mut self.bit_reader, dist_symbol)?;
                let dist = Self::plane_code_to_distance(width, dist_code);
                if pos < dist || end - pos < length {
                    return Err(DecodingError::BitStreamError);
                }
                for i in pos..pos + length {
                    data[i] = data[i - dist];
                }
                pos += length;
            } else {
                return Err(DecodingError::BitStreamError);
            }
            if self.bit_reader.is_eos()? {
                self.bit_reader.eos = true;
                break;
            }
        }
        if pos < end {
            return Err(DecodingError::BitStreamError);
        }
        Ok(())
    }
}

struct HuffmanInfo {
    xsize: u16,
    _ysize: u16,
    color_cache: Option<ColorCache>,
    image: Vec<u16>,
    bits: u8,
    mask: u16,
    huffman_code_groups: Vec<HuffmanCodeGroup>,
}

impl HuffmanInfo {
    fn get_huff_index(&self, x: u16, y: u16) -> usize {
        if self.bits == 0 {
            return 0;
        }
        let position =
            usize::from(y >> self.bits) * usize::from(self.xsize) + usize::from(x >> self.bits);
        let meta_huff_code: usize = usize::from(self.image[position]);
        meta_huff_code
    }
}

/// The colours of a colour cache as RGBA words, hashed as libwebp hashes its ARGB words; held
/// at the largest size, 11 bits, so an index masked to it needs no bounds check.
#[derive(Debug, Clone)]
struct ColorCache {
    color_cache_bits: u8,
    color_cache: Box<[u32; 1 << 11]>,
}

impl ColorCache {
    fn new(bits: u8) -> Self {
        Self { color_cache_bits: bits, color_cache: Box::new([0; 1 << 11]) }
    }

    #[inline(always)]
    fn insert(&mut self, rgba: u32) {
        let argb = rgba.swap_bytes().rotate_right(8);
        let index = (0x1e35a7bdu32.wrapping_mul(argb)) >> (32 - self.color_cache_bits);
        self.color_cache[index as usize & ((1 << 11) - 1)] = rgba;
    }

    /// Inserts the RGBA pixels `pixels` in order.
    #[inline(always)]
    fn insert_all(&mut self, pixels: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        let pixels = if pixels.len() >= 32 {
            &pixels[avx2::color_cache_insert(&mut self.color_cache, self.color_cache_bits, pixels) * 4..]
        } else {
            pixels
        };
        for pixel in pixels.as_chunks::<4>().0 {
            self.insert(u32::from_le_bytes(*pixel));
        }
    }

    #[inline(always)]
    fn lookup(&self, index: usize) -> u32 {
        self.color_cache[index & ((1 << 11) - 1)]
    }
}

/// libwebp's VP8L bit reader: a 64-bit window `val` over the stream, read from `bit_pos`, with
/// bytes shifted in at the top. Past the end it keeps libwebp's values (reads wrap around the
/// window, and set `eos`), since a stream may end inside its last code. The stream is read a
/// [`READ_AHEAD`] window at a time.
#[derive(Debug, Clone)]
pub(crate) struct BitReader<R> {
    reader: R,
    buf: Vec<u8>,
    /// Where the bytes of `buf` not shifted in yet start.
    at: usize,
    val: u64,
    bit_pos: u32,
    eos: bool,
    started: bool,
}

const LBITS: u32 = 64;
const WBITS: u32 = 32;
const READ_AHEAD: usize = 64 * 1024;

impl<R: BufRead> BitReader<R> {
    const fn new(reader: R) -> Self {
        Self {
            reader,
            buf: Vec::new(),
            at: 0,
            val: 0,
            // An empty window: the first fill goes the slow way, which starts the stream.
            bit_pos: LBITS,
            eos: false,
            started: false,
        }
    }

    /// The stream's bytes not shifted in yet, empty at its end.
    #[inline]
    fn bytes(&mut self) -> Result<&[u8], DecodingError> {
        if self.at == self.buf.len() {
            self.read_ahead()?;
        }
        Ok(&self.buf[self.at..])
    }

    #[cold]
    fn read_ahead(&mut self) -> Result<(), DecodingError> {
        self.buf.resize(READ_AHEAD, 0);
        let mut n = 0;
        while n < READ_AHEAD {
            match self.reader.read(&mut self.buf[n..]) {
                Ok(0) => break,
                Ok(k) => n += k,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.into()),
            }
        }
        self.buf.truncate(n);
        self.at = 0;
        Ok(())
    }

    fn start(&mut self) -> Result<(), DecodingError> {
        self.started = true;
        self.bit_pos = 0;
        for i in 0..8 {
            let Some(&b) = self.bytes()?.first() else { break };
            self.val |= u64::from(b) << (8 * i);
            self.at += 1;
        }
        Ok(())
    }

    fn exhausted(&mut self) -> Result<bool, DecodingError> {
        Ok(self.bytes()?.is_empty())
    }

    /// `VP8LIsEndOfStream`.
    pub(crate) fn is_eos(&mut self) -> Result<bool, DecodingError> {
        Ok(self.eos || (self.bit_pos > LBITS && self.exhausted()?))
    }

    fn shift_bytes(&mut self) -> Result<(), DecodingError> {
        while self.bit_pos >= 8 {
            let Some(&b) = self.bytes()?.first() else { break };
            self.val = (self.val >> 8) | u64::from(b) << (LBITS - 8);
            self.at += 1;
            self.bit_pos -= 8;
        }
        if self.is_eos()? {
            self.eos = true;
            self.bit_pos = 0;
        }
        Ok(())
    }

    /// `VP8LFillBitWindow`. Short of the end of the read-ahead it shifts in every whole byte read
    /// with one load: `val` holds the eight bytes before `at` either way, so the bits read and
    /// the state at the end of the stream come out as libwebp's.
    #[inline(always)]
    pub(crate) fn fill(&mut self) -> Result<(), DecodingError> {
        if self.bit_pos < LBITS && self.buf.len() - self.at > 8 {
            let w = u64::from_le_bytes(self.buf[self.at..self.at + 8].try_into().unwrap());
            let shift = self.bit_pos & !7;
            // Branch-free: shifting by 64 in two steps leaves nothing of `w` where no byte is read.
            self.val = (self.val >> shift) | (w << 1) << (LBITS - 1 - shift);
            self.at += (shift / 8) as usize;
            self.bit_pos &= 7;
            return Ok(());
        }
        self.fill_slow()
    }

    #[cold]
    #[inline(never)]
    fn fill_slow(&mut self) -> Result<(), DecodingError> {
        if !self.started {
            self.start()?;
        }
        if self.bit_pos >= WBITS {
            let buf = self.bytes()?;
            // libwebp loads four bytes at once only with more than eight left; the window
            // comes out the same either way short of the end.
            if buf.len() > 8 {
                let w = u32::from_le_bytes(buf[..4].try_into().unwrap());
                self.at += 4;
                self.val = (self.val >> WBITS) | u64::from(w) << (LBITS - WBITS);
                self.bit_pos -= WBITS;
            } else {
                self.shift_bytes()?;
            }
        }
        Ok(())
    }

    /// `VP8LPrefetchBits`, all 64 bits of it.
    pub(crate) fn peek_full(&self) -> u64 {
        self.val >> (self.bit_pos & (LBITS - 1))
    }

    pub(crate) fn peek(&self, num: u8) -> u64 {
        self.peek_full() & ((1 << num) - 1)
    }

    /// The bits a code of up to 15 bits is looked up with: libwebp reads a longer code's bits
    /// past the eighth from a second window read, which wraps differently past the end.
    pub(crate) fn peek_code(&self) -> u64 {
        let v = self.peek_full();
        if self.bit_pos + 15 <= LBITS {
            return v;
        }
        let second = self.val >> ((self.bit_pos + 8) & (LBITS - 1));
        (v & 0xff) | second << 8
    }

    /// `VP8LSetBitPos`: no check, as libwebp checks for the end where it decides.
    pub(crate) fn advance(&mut self, num: u8) {
        self.bit_pos += u32::from(num);
    }

    /// Advances, failing where that runs past the end of the stream.
    #[inline(always)]
    pub(crate) fn consume(&mut self, num: u8) -> Result<(), DecodingError> {
        self.advance(num);
        self.check()
    }

    /// Fails where the bits read so far run past the end of the stream.
    #[inline(always)]
    pub(crate) fn check(&mut self) -> Result<(), DecodingError> {
        if self.eos || self.bit_pos > LBITS {
            return self.check_end();
        }
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn check_end(&mut self) -> Result<(), DecodingError> {
        if self.is_eos()? {
            return Err(DecodingError::BitStreamError);
        }
        Ok(())
    }

    /// `VP8LReadBits`: zero once at the end. Short of the end its byte shifts are one fill.
    #[inline(always)]
    pub(crate) fn read_bits_raw(&mut self, num: u8) -> Result<u32, DecodingError> {
        let bit_pos = self.bit_pos + u32::from(num);
        if !self.eos && num <= 24 && bit_pos < LBITS && self.buf.len() - self.at > 8 {
            let v = self.peek(num) as u32;
            self.bit_pos = bit_pos;
            self.fill()?;
            return Ok(v);
        }
        self.read_bits_slow(num)
    }

    #[cold]
    #[inline(never)]
    fn read_bits_slow(&mut self, num: u8) -> Result<u32, DecodingError> {
        if !self.started {
            self.start()?;
        }
        if self.eos || num > 24 {
            self.eos = true;
            self.bit_pos = 0;
            return Ok(0);
        }
        let v = self.peek(num) as u32;
        self.bit_pos += u32::from(num);
        self.shift_bytes()?;
        Ok(v)
    }

    /// Reads `num` bits, failing where that runs past the end of the stream.
    pub(crate) fn read_bits<T: TryFrom<u32>>(&mut self, num: u8) -> Result<T, DecodingError> {
        debug_assert!(num as usize <= 8 * mem::size_of::<T>());
        let value = self.read_bits_raw(num)?;
        if self.eos {
            return Err(DecodingError::BitStreamError);
        }
        value.try_into().map_err(|_| DecodingError::BitStreamError)
    }
}

#[cfg(test)]
mod test {

    use std::io::Cursor;

    use super::BitReader;

    #[test]
    fn bit_read_test() {
        //10011100 01000001 11100001
        let mut bit_reader = BitReader::new(Cursor::new(vec![0x9C, 0x41, 0xE1]));

        assert_eq!(bit_reader.read_bits::<u8>(3).unwrap(), 4); //100
        assert_eq!(bit_reader.read_bits::<u8>(2).unwrap(), 3); //11
        assert_eq!(bit_reader.read_bits::<u8>(6).unwrap(), 12); //001100
        assert_eq!(bit_reader.read_bits::<u16>(10).unwrap(), 40); //0000101000
        assert_eq!(bit_reader.read_bits::<u8>(3).unwrap(), 7); //111
    }

    #[test]
    fn bit_read_error_test() {
        //01101010
        let mut bit_reader = BitReader::new(Cursor::new(vec![0x6A]));

        assert_eq!(bit_reader.read_bits::<u8>(3).unwrap(), 2); //010
        assert_eq!(bit_reader.read_bits::<u8>(5).unwrap(), 13); //01101
        // As libwebp, zeros up to 64 bits, then the end.
        assert_eq!(bit_reader.read_bits::<u32>(24).unwrap(), 0);
        assert_eq!(bit_reader.read_bits::<u32>(24).unwrap(), 0);
        assert_eq!(bit_reader.read_bits::<u8>(8).unwrap(), 0);
        assert!(bit_reader.read_bits::<u8>(1).is_err());
    }
}
