use byteorder_lite::{LittleEndian, ReadBytesExt};
use std::default::Default;
use std::io::{self, Read, Seek, SeekFrom};

use crate::ahead::{Ahead, Produce, Runner};
use crate::decoder::{DecodingError, UpsamplingMethod};
use crate::ReadSeek;
use common::*;
use prediction::*;

use arithmetic_decoder::{by_position, ArithmeticDecoder, ModeProbs, PlaneProbs};

mod arithmetic_decoder;
mod common;
mod loop_filter;
#[cfg(all(decoder_simd, target_arch = "aarch64"))]
mod neon;
mod prediction;
#[cfg(all(decoder_simd, any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32")))]
mod sse2;
#[cfg(all(decoder_simd, target_arch = "aarch64"))]
use neon as simd;
#[cfg(decoder_simd)]
pub(crate) use simd::{unfilter_gradient, unfilter_horizontal};
#[cfg(all(decoder_simd, any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32")))]
use sse2 as simd;
mod transform;
#[cfg(feature = "encoder")]
pub(crate) mod encoder;
mod yuv;
pub(crate) use yuv::fill_row_yuv;
#[cfg(feature = "encoder")]
pub(crate) use yuv::upsample_chroma_row;

#[derive(Default, Clone, Copy)]
struct MacroBlock {
    bpred: [IntraMode; 16],
    luma_mode: LumaMode,
    chroma_mode: ChromaMode,
    segmentid: u8,
    coeffs_skipped: bool,
    non_zero_dct: bool,
}

/// Info required from a previously decoded macro block in future
/// For the top macroblocks this will be the bottom values, for the left macroblock the right values
#[derive(Default, Clone, Copy)]
struct PreviousMacroBlock {
    bpred: [IntraMode; 4],
    // complexity is laid out like: y2,y,y,y,y,u,u,v,v
    complexity: [u8; 9],
}

/// A Representation of the last decoded video frame
#[derive(Default, Debug, Clone)]
pub struct Frame {
    /// The width of the luma plane
    pub width: u16,

    /// The height of the luma plane
    pub height: u16,

    /// The luma plane of the frame
    pub ybuf: Vec<u8>,

    /// The blue plane of the frame
    pub ubuf: Vec<u8>,

    /// The red plane of the frame
    pub vbuf: Vec<u8>,

    pub(crate) version: u8,

    /// Indicates whether this frame is a keyframe
    // TODO: Remove in the next API breaking release
    pub keyframe: bool,

    /// Indicates whether this frame is intended for display
    pub for_display: bool,

    // Section 9.2
    /// The pixel type of the frame as defined by Section 9.2
    /// of the VP8 Specification
    pub pixel_type: u8,

    // Section 9.4 and 15
    pub(crate) filter_type: bool, //if true uses simple filter // if false uses normal filter
    pub(crate) filter_level: u8,
    pub(crate) sharpness_level: u8,
}

impl Frame {



    /// Gets the buffer size
    #[must_use]
    pub fn get_buf_size(&self) -> usize {
        self.ybuf.len() * 3
    }
}

/// VP8 Decoder
///
/// Only decodes keyframes
pub struct Vp8Decoder<'a> {
    /// The data the bitstream is read from.
    source: Box<dyn ReadSeek + Send + 'a>,
    b: ArithmeticDecoder,

    mbwidth: u16,
    mbheight: u16,
    macroblocks: Vec<MacroBlock>,

    frame: Frame,

    segments_enabled: bool,
    segments_update_map: bool,
    segment: [Segment; MAX_SEGMENTS],

    loop_filter_adjustments_enabled: bool,
    ref_delta: [i32; 4],
    mode_delta: [i32; 4],

    partitions: [ArithmeticDecoder; 8],
    num_partitions: u8,

    segment_probs: [Prob; 3],
    token_probs: Box<TokenProbTables>,
    pos_probs: Box<[PlaneProbs; 4]>,

    // Section 9.11
    prob_skip_false: Option<Prob>,
    /// The modes of the macroblock row, read ahead of its residuals as libwebp reads them.
    mode_row: Vec<MacroBlock>,

    top: Vec<PreviousMacroBlock>,
    left: PreviousMacroBlock,

    // The bottom rows of the previous macroblock row, used for predictions
    // See Section 12
    top_border_y: Vec<u8>,
    top_border_u: Vec<u8>,
    top_border_v: Vec<u8>,

    // The macroblock being reconstructed with its borders, kept from one macroblock to the
    // next for the left border.
    ws_y: [u8; LUMA_BLOCK_SIZE],
    ws_u: [u8; CHROMA_BLOCK_SIZE],
    ws_v: [u8; CHROMA_BLOCK_SIZE],

    // Luma and chroma row held in the first row of ybuf and u/vbuf; with `band_rows`,
    // the buffers hold only a band of rows instead of the whole frame.
    y_base: usize,
    uv_base: usize,
    band_rows: bool,
}

impl<'a> Vp8Decoder<'a> {
    fn new() -> Self {
        let f = Frame::default();
        let s = Segment::default();

        Self {
            source: Box::new(io::Cursor::new(&[][..])),
            b: ArithmeticDecoder::new(),

            mbwidth: 0,
            mbheight: 0,
            macroblocks: Vec::new(),

            frame: f,
            segments_enabled: false,
            segments_update_map: false,
            segment: [s; MAX_SEGMENTS],

            loop_filter_adjustments_enabled: false,
            ref_delta: [0; 4],
            mode_delta: [0; 4],

            partitions: [
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
            ],

            num_partitions: 1,

            segment_probs: [255; 3],
            token_probs: Box::new(COEFF_PROBS),
            pos_probs: by_position(&COEFF_PROBS),

            // Section 9.11
            prob_skip_false: None,
            mode_row: Vec::new(),

            top: Vec::new(),
            left: PreviousMacroBlock::default(),

            top_border_y: Vec::new(),
            top_border_u: Vec::new(),
            top_border_v: Vec::new(),

            ws_y: [0; LUMA_BLOCK_SIZE],
            ws_u: [0; CHROMA_BLOCK_SIZE],
            ws_v: [0; CHROMA_BLOCK_SIZE],

            y_base: 0,
            uv_base: 0,
            band_rows: false,
        }
    }

    fn update_token_probabilities(&mut self) -> Result<(), DecodingError> {
        let mut res = self.b.start_accumulated_result();
        for (i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                for (k, ks) in js.iter().enumerate() {
                    for (t, prob) in ks.iter().enumerate().take(NUM_DCT_TOKENS - 1) {
                        if self.b.read_bool(*prob).or_accumulate(&mut res) {
                            let v = self.b.read_literal(8).or_accumulate(&mut res);
                            self.token_probs[i][j][k][t] = v;
                        }
                    }
                }
            }
        }
        self.b.check(res, ())
    }

    /// Partitions as libwebp splits the data `start..end`: sizes past the data are cut to it,
    /// and the last partition takes the rest of the data, which must not be empty.
    fn init_partitions(&mut self, n: usize, start: u64, end: u64) -> Result<(), DecodingError> {
        let last = n - 1;
        if end - start < 3 * last as u64 {
            return Err(DecodingError::NotEnoughInitData);
        }
        let mut sizes = [0; 21];
        self.source.seek(SeekFrom::Start(start))?;
        self.source.read_exact(&mut sizes[..3 * last])?;
        let mut at = start + 3 * last as u64;
        for (i, s) in sizes[..3 * last].chunks_exact(3).enumerate() {
            let size = u64::from(s[0]) | u64::from(s[1]) << 8 | u64::from(s[2]) << 16;
            let part_end = (at + size).min(end);
            self.partitions[i].init_window(&mut *self.source, at, part_end)?;
            at = part_end;
        }
        if at == end {
            return Err(DecodingError::NotEnoughInitData);
        }
        self.partitions[last].init_window(&mut *self.source, at, end)?;
        Ok(())
    }

    fn read_quantization_indices(&mut self) -> Result<(), DecodingError> {
        fn dc_quant(index: i32) -> i16 {
            DC_QUANT[index.clamp(0, 127) as usize]
        }

        fn ac_quant(index: i32) -> i16 {
            AC_QUANT[index.clamp(0, 127) as usize]
        }

        let mut res = self.b.start_accumulated_result();

        let yac_abs = self.b.read_literal(7).or_accumulate(&mut res);
        let ydc_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let y2dc_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let y2ac_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let uvdc_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let uvac_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);

        let n = if self.segments_enabled {
            MAX_SEGMENTS
        } else {
            1
        };
        for i in 0usize..n {
            let base = i32::from(if self.segments_enabled {
                if self.segment[i].delta_values {
                    i16::from(self.segment[i].quantizer_level) + i16::from(yac_abs)
                } else {
                    i16::from(self.segment[i].quantizer_level)
                }
            } else {
                i16::from(yac_abs)
            });

            self.segment[i].ydc = dc_quant(base + ydc_delta);
            self.segment[i].yac = ac_quant(base);

            self.segment[i].y2dc = dc_quant(base + y2dc_delta) * 2;
            // The intermediate result (max`284*155`) can be larger than the `i16` range.
            self.segment[i].y2ac = (i32::from(ac_quant(base + y2ac_delta)) * 155 / 100) as i16;

            self.segment[i].uvdc = dc_quant(base + uvdc_delta);
            self.segment[i].uvac = ac_quant(base + uvac_delta);

            if self.segment[i].y2ac < 8 {
                self.segment[i].y2ac = 8;
            }

            if self.segment[i].uvdc > 132 {
                self.segment[i].uvdc = 132;
            }
        }

        self.b.check(res, ())
    }

    fn read_loop_filter_adjustments(&mut self) -> Result<(), DecodingError> {
        let mut res = self.b.start_accumulated_result();

        if self.b.read_flag().or_accumulate(&mut res) {
            for i in 0usize..4 {
                self.ref_delta[i] = self.b.read_optional_signed_value(6).or_accumulate(&mut res);
            }

            for i in 0usize..4 {
                self.mode_delta[i] = self.b.read_optional_signed_value(6).or_accumulate(&mut res);
            }
        }

        self.b.check(res, ())
    }

    fn read_segment_updates(&mut self) -> Result<(), DecodingError> {
        let mut res = self.b.start_accumulated_result();

        // Section 9.3
        self.segments_update_map = self.b.read_flag().or_accumulate(&mut res);
        let update_segment_feature_data = self.b.read_flag().or_accumulate(&mut res);

        if update_segment_feature_data {
            let segment_feature_mode = self.b.read_flag().or_accumulate(&mut res);

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].delta_values = !segment_feature_mode;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].quantizer_level =
                    self.b.read_optional_signed_value(7).or_accumulate(&mut res) as i8;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].loopfilter_level =
                    self.b.read_optional_signed_value(6).or_accumulate(&mut res) as i8;
            }
        }

        if self.segments_update_map {
            for i in 0usize..3 {
                let update = self.b.read_flag().or_accumulate(&mut res);

                let prob = if update {
                    self.b.read_literal(8).or_accumulate(&mut res)
                } else {
                    255
                };
                self.segment_probs[i] = prob;
            }
        }

        self.b.check(res, ())
    }

    /// Reads the frame header of the VP8 bitstream at `start` of the source, which runs to
    /// the end of the data at `end`, and splits the rest into partitions.
    fn read_frame_header(&mut self, start: u64, end: u64) -> Result<(), DecodingError> {
        let mut header = [0; 10];
        let have = (end - start).min(10) as usize;
        self.source.seek(SeekFrom::Start(start))?;
        self.source.read_exact(&mut header[..have])?;
        let mut r = &header[..have];
        let tag = r.read_u24::<LittleEndian>()?;

        let keyframe = tag & 1 == 0;
        if !keyframe {
            return Err(DecodingError::UnsupportedFeature(
                "Non-keyframe frames".to_owned(),
            ));
        }
        self.frame.keyframe = true;

        self.frame.version = ((tag >> 1) & 7) as u8;
        self.frame.for_display = (tag >> 4) & 1 != 0;
        if self.frame.version > 3 || !self.frame.for_display {
            return Err(DecodingError::UnsupportedFeature("VP8 profile or hidden frame".to_owned()));
        }

        let first_partition_size = tag >> 5;

        let mut tag = [0u8; 3];
        r.read_exact(&mut tag)?;

        if tag != [0x9d, 0x01, 0x2a] {
            return Err(DecodingError::Vp8MagicInvalid(tag));
        }

        let w = r.read_u16::<LittleEndian>()?;
        let h = r.read_u16::<LittleEndian>()?;

        self.frame.width = w & 0x3FFF;
        self.frame.height = h & 0x3FFF;

        self.mbwidth = self.frame.width.div_ceil(16);
        self.mbheight = self.frame.height.div_ceil(16);

        // defaults are intra mode DC and complexity 0
        self.top = vec![PreviousMacroBlock::default(); self.mbwidth.into()];
        self.left = PreviousMacroBlock::default();

        let (luma_rows, chroma_rows) = if self.band_rows {
            (BAND_KEEP_Y + 16, BAND_KEEP_UV + 8)
        } else {
            (
                usize::from(self.mbheight) * 16,
                usize::from(self.mbheight) * 8,
            )
        };
        self.frame.ybuf = vec![0u8; usize::from(self.mbwidth) * 16 * luma_rows];
        self.frame.ubuf = vec![0u8; usize::from(self.mbwidth) * 8 * chroma_rows];
        self.frame.vbuf = vec![0u8; usize::from(self.mbwidth) * 8 * chroma_rows];

        self.top_border_y = vec![127u8; self.frame.width as usize + 4 + 16];

        // 8 pixels per macroblock
        self.top_border_u = vec![127u8; 8 * self.mbwidth as usize];

        self.top_border_v = vec![127u8; 8 * self.mbwidth as usize];

        let size = u64::from(first_partition_size);
        let first_start = start + 10;
        if end - first_start < size {
            return Err(DecodingError::IoError(io::ErrorKind::UnexpectedEof.into()));
        }
        let mut first = vec![0; size as usize];
        self.source.seek(SeekFrom::Start(first_start))?;
        self.source.read_exact(&mut first)?;
        self.b.init(&first);

        let mut res = self.b.start_accumulated_result();
        let color_space = self.b.read_literal(1).or_accumulate(&mut res);
        self.frame.pixel_type = self.b.read_literal(1).or_accumulate(&mut res);

        // libwebp reads and ignores the colour space bit.
        let _ = color_space;

        self.segments_enabled = self.b.read_flag().or_accumulate(&mut res);
        if self.segments_enabled {
            self.read_segment_updates()?;
        }

        self.frame.filter_type = self.b.read_flag().or_accumulate(&mut res);
        self.frame.filter_level = self.b.read_literal(6).or_accumulate(&mut res);
        self.frame.sharpness_level = self.b.read_literal(3).or_accumulate(&mut res);

        self.loop_filter_adjustments_enabled = self.b.read_flag().or_accumulate(&mut res);
        if self.loop_filter_adjustments_enabled {
            self.read_loop_filter_adjustments()?;
        }

        let num_partitions = 1 << self.b.read_literal(2).or_accumulate(&mut res) as usize;
        self.b.check(res, ())?;

        self.num_partitions = num_partitions as u8;
        self.init_partitions(num_partitions, first_start + size, end)?;

        self.read_quantization_indices()?;

        // Refresh entropy probs ?????
        let _ = self.b.read_literal(1);

        self.update_token_probabilities()?;
        self.pos_probs = by_position(&self.token_probs);

        let mut res = self.b.start_accumulated_result();
        let mb_no_skip_coeff = self.b.read_literal(1).or_accumulate(&mut res);
        self.prob_skip_false = if mb_no_skip_coeff == 1 {
            Some(self.b.read_literal(8).or_accumulate(&mut res))
        } else {
            None
        };
        self.b.check(res, ())?;
        Ok(())
    }

    #[inline(always)]
    fn read_macroblock_header(&mut self, mbx: usize) -> Result<MacroBlock, DecodingError> {
        if mbx == 0 {
            self.read_mode_row()?;
        }
        Ok(self.mode_row[mbx])
    }

    /// libwebp's `VP8ParseIntraModeRow`.
    fn read_mode_row(&mut self) -> Result<(), DecodingError> {
        let probs = ModeProbs {
            segment: (self.segments_enabled && self.segments_update_map).then_some(self.segment_probs),
            skip: self.prob_skip_false,
        };
        self.mode_row.clear();
        for top in &mut self.top {
            let m = self.b.read_modes(probs, &mut top.bpred, &mut self.left.bpred)?;
            self.mode_row.push(MacroBlock {
                bpred: m.bpred,
                luma_mode: m.luma,
                chroma_mode: m.chroma,
                segmentid: m.segment,
                coeffs_skipped: m.skip,
                ..MacroBlock::default()
            });
        }
        Ok(())
    }

    fn intra_predict_luma(
        &mut self,
        mbx: usize,
        mby: usize,
        mb: &MacroBlock,
        resdata: &mut [i16; 384],
        kinds: &[u8; 24],
    ) {
        let stride = 1usize + 16 + 4;
        let mw = self.mbwidth as usize;
        let ws = &mut self.ws_y;
        update_border_luma(ws, mbx, mby, mw, &self.top_border_y);

        match mb.luma_mode {
            LumaMode::V => predict_vpred(ws, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(ws, 16, 1, 1, stride),
            LumaMode::TM => predict_tm::<16>(ws, 1, 1, stride),
            LumaMode::DC => predict_dc::<16>(ws, stride, mby != 0, mbx != 0),
            LumaMode::B => predict_4x4(ws, &mb.bpred, resdata, kinds),
        }

        if mb.luma_mode != LumaMode::B {
            for y in 0usize..4 {
                for x in 0usize..4 {
                    let i = x + y * 4;
                    let pos = (1 + y * 4) * stride + 1 + x * 4;
                    transform::add_coeffs(ws, &mut resdata[i * 16..][..16], kinds[i], pos, stride);
                }
            }
        }
    }

    /// Copies the reconstructed luma of macroblock `mbx`, `mby` out of the workspace, apart
    /// from its prediction so that the stores of the last transforms have left.
    fn store_luma(&mut self, mbx: usize, mby: usize) {
        let stride = LUMA_STRIDE;
        let mw = self.mbwidth as usize;
        let ws = &self.ws_y;
        self.top_border_y[mbx * 16..][..16].copy_from_slice(&ws[16 * stride + 1..][..16]);

        let lw = mw * 16;
        let base = (mby * 16).wrapping_sub(self.y_base) * lw + mbx * 16;
        let dst = &mut self.frame.ybuf[base..][..15 * lw + 16];
        for y in 0usize..16 {
            dst[y * lw..][..16].copy_from_slice(&ws[(1 + y) * stride + 1..][..16]);
        }
    }

    fn intra_predict_chroma(
        &mut self,
        mbx: usize,
        mby: usize,
        mb: &MacroBlock,
        resdata: &mut [i16; 384],
        kinds: &[u8; 24],
    ) {
        let stride = 1usize + 8;

        let mw = self.mbwidth as usize;

        let (uws, vws) = (&mut self.ws_u, &mut self.ws_v);
        update_border_chroma(uws, mbx, mby, &self.top_border_u);
        update_border_chroma(vws, mbx, mby, &self.top_border_v);

        match mb.chroma_mode {
            ChromaMode::DC => {
                predict_dc::<8>(uws, stride, mby != 0, mbx != 0);
                predict_dc::<8>(vws, stride, mby != 0, mbx != 0);
            }
            ChromaMode::V => {
                predict_vpred(uws, 8, 1, 1, stride);
                predict_vpred(vws, 8, 1, 1, stride);
            }
            ChromaMode::H => {
                predict_hpred(uws, 8, 1, 1, stride);
                predict_hpred(vws, 8, 1, 1, stride);
            }
            ChromaMode::TM => {
                predict_tm::<8>(uws, 1, 1, stride);
                predict_tm::<8>(vws, 1, 1, stride);
            }
        }

        let pos = stride + 1;
        transform::add_coeffs_uv(uws, &mut resdata[16 * 16..][..64], &kinds[16..20], pos, stride);
        transform::add_coeffs_uv(vws, &mut resdata[20 * 16..][..64], &kinds[20..24], pos, stride);

        self.top_border_u[mbx * 8..][..8].copy_from_slice(&uws[8 * stride + 1..][..8]);
        self.top_border_v[mbx * 8..][..8].copy_from_slice(&vws[8 * stride + 1..][..8]);

        let cw = mw * 8;
        let base = (mby * 8).wrapping_sub(self.uv_base) * cw + mbx * 8;
        for (buf, ws) in [
            (&mut self.frame.ubuf, &self.ws_u),
            (&mut self.frame.vbuf, &self.ws_v),
        ] {
            let dst = &mut buf[base..][..7 * cw + 8];
            for y in 0usize..8 {
                dst[y * cw..][..8].copy_from_slice(&ws[(1 + y) * stride + 1..][..8]);
            }
        }
    }

    /// Reads the coefficients of a macroblock into `blocks` (zero on entry) and the kind of
    /// inverse transform each block needs into `kinds`.
    #[inline(always)]
    fn read_residual_data(
        &mut self,
        mb: &mut MacroBlock,
        mbx: usize,
        p: usize,
        blocks: &mut [i16; 384],
        kinds: &mut [u8; 24],
    ) -> Result<(), DecodingError> {
        let seg = &self.segment[mb.segmentid as usize];
        let probs = &self.pos_probs;
        let d = &mut self.partitions[p];
        let mut top = self.top[mbx].complexity;
        let mut lefts = self.left.complexity;
        let mut plane = if mb.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        if plane == Plane::Y2 {
            let complexity = usize::from(top[0] + lefts[0]);
            let mut block = [0i16; 16];
            let dq = [i32::from(seg.y2dc), i32::from(seg.y2ac)];
            let n = d.read_coeffs(&probs[plane as usize], complexity, 0, dq, &mut block)?;

            lefts[0] = u8::from(n > 0);
            top[0] = u8::from(n > 0);

            let mut block = block.map(i32::from);
            transform::iwht4x4(&mut block);

            for k in 0usize..16 {
                blocks[16 * k] = block[k] as i16;
            }

            plane = Plane::YCoeff1;
        }
        let first = usize::from(plane == Plane::YCoeff1);
        let dq = [i32::from(seg.ydc), i32::from(seg.yac)];

        for y in 0usize..4 {
            let mut left = lefts[y + 1];
            for x in 0usize..4 {
                let i = x + y * 4;
                let block: &mut [i16; 16] = (&mut blocks[i * 16..][..16]).try_into().unwrap();

                let complexity = usize::from(top[x + 1] + left);
                let n = d.read_coeffs(&probs[plane as usize], complexity, first, dq, block)?;
                kinds[i] = transform::luma_kind(n, block);

                left = u8::from(n > first);
                top[x + 1] = left;
            }

            lefts[y + 1] = left;
        }

        let dq = [i32::from(seg.uvdc), i32::from(seg.uvac)];
        let probs = &probs[Plane::Chroma as usize];
        for &j in &[5usize, 7usize] {
            for y in 0usize..2 {
                let mut left = lefts[y + j];

                for x in 0usize..2 {
                    let i = x + y * 2 + if j == 5 { 16 } else { 20 };
                    let block: &mut [i16; 16] = (&mut blocks[i * 16..][..16]).try_into().unwrap();

                    let complexity = usize::from(top[x + j] + left);
                    let n = d.read_coeffs(probs, complexity, 0, dq, block)?;
                    kinds[i] = transform::kind(n, block[0]);

                    left = u8::from(n > 0);
                    top[x + j] = left;
                }

                lefts[y + j] = left;
            }
        }

        self.top[mbx].complexity = top;
        self.left.complexity = lefts;
        mb.non_zero_dct = kinds.iter().any(|&k| k != transform::NO_COEFFS);
        Ok(())
    }

    /// Does loop filtering on the macroblock
    fn loop_filter(&mut self, mbx: usize, mby: usize, mb: &MacroBlock) {
        let f = self.mb_filter(mb);
        let mut planes = PlaneRefs {
            y: &mut self.frame.ybuf,
            u: &mut self.frame.ubuf,
            v: &mut self.frame.vbuf,
            y_base: self.y_base,
            uv_base: self.uv_base,
            luma_w: self.mbwidth as usize * 16,
            chroma_w: self.mbwidth as usize * 8,
            filter_type: self.frame.filter_type,
        };
        filter_macroblock(&mut planes, mbx, mby, f);
    }

    fn mb_filter(&self, mb: &MacroBlock) -> MbFilter {
        let (level, interior, hev) = self.calculate_filter_parameters(mb);
        // we skip subblock filtering if the coding mode isn't B_PRED and there's no DCT coefficient coded
        let subblocks = mb.luma_mode == LumaMode::B || (!mb.coeffs_skipped && mb.non_zero_dct);
        MbFilter {
            level,
            interior,
            hev,
            subblocks,
        }
    }

    //return values are the filter level, interior limit and hev threshold
    fn calculate_filter_parameters(&self, macroblock: &MacroBlock) -> (u8, u8, u8) {
        let segment = self.segment[macroblock.segmentid as usize];
        let mut filter_level = i32::from(self.frame.filter_level);

        // if frame level filter level is 0, we must skip loop filter
        if filter_level == 0 {
            return (0, 0, 0);
        }

        if self.segments_enabled {
            if segment.delta_values {
                filter_level += i32::from(segment.loopfilter_level);
            } else {
                filter_level = i32::from(segment.loopfilter_level);
            }
        }

        if self.loop_filter_adjustments_enabled {
            filter_level += self.ref_delta[0];
            if macroblock.luma_mode == LumaMode::B {
                filter_level += self.mode_delta[0];
            }
        }

        let filter_level = filter_level.clamp(0, 63) as u8;

        //interior limit
        let mut interior_limit = filter_level;

        if self.frame.sharpness_level > 0 {
            interior_limit >>= if self.frame.sharpness_level > 4 { 2 } else { 1 };

            if interior_limit > 9 - self.frame.sharpness_level {
                interior_limit = 9 - self.frame.sharpness_level;
            }
        }

        if interior_limit == 0 {
            interior_limit = 1;
        }

        // high edge variance threshold
        let hev_threshold = if filter_level >= 40 {
            2
        } else if filter_level >= 15 {
            1
        } else {
            0
        };

        (filter_level, interior_limit, hev_threshold)
    }

    /// Decodes the frame of a VP8 bitstream read from `r` to its end.
    pub fn decode_frame<R: Read>(mut r: R) -> Result<Frame, DecodingError> {
        let mut data = Vec::new();
        r.read_to_end(&mut data)?;
        let end = data.len() as u64;
        let mut decoder = Vp8Decoder::new();
        decoder.source = Box::new(io::Cursor::new(data));
        decoder.read_frame_header(0, end)?;
        decoder.decode_frame_()
    }

    fn decode_frame_(mut self) -> Result<Frame, DecodingError> {
        let mut row = Vec::with_capacity(usize::from(self.mbwidth));
        for mby in 0..self.mbheight as usize {
            self.decode_mb_row(mby, &mut row, None)?;
            self.macroblocks.extend_from_slice(&row);
        }
        for mby in 0..self.mbheight as usize {
            for mbx in 0..self.mbwidth as usize {
                let mb = self.macroblocks[mby * self.mbwidth as usize + mbx];
                self.loop_filter(mbx, mby, &mb);
            }
        }
        Ok(self.frame)
    }

    /// Reconstructs macroblock row `mby` without loop filtering, its macroblocks into `row`, read
    /// here or taken from `parsed`, which read them ahead.
    fn decode_mb_row(
        &mut self,
        mby: usize,
        row: &mut Vec<MacroBlock>,
        mut parsed: Option<&mut ParsedAhead>,
    ) -> Result<(), DecodingError> {
        let p = mby % self.num_partitions as usize;
        self.left = PreviousMacroBlock::default();
        row.clear();
        let mut blocks = [0i16; 384];
        for mbx in 0..self.mbwidth as usize {
            let mut kinds = [transform::NO_COEFFS; 24];
            let mb = match parsed.as_deref_mut() {
                Some(parsed) => parsed.next(&mut blocks, &mut kinds)?,
                None => self.parse_mb(mbx, p, &mut blocks, &mut kinds)?,
            };
            self.intra_predict_luma(mbx, mby, &mb, &mut blocks, &kinds);
            self.intra_predict_chroma(mbx, mby, &mb, &mut blocks, &kinds);
            self.store_luma(mbx, mby);
            row.push(mb);
        }
        Ok(())
    }

    /// Reads the modes of macroblock `mbx` of the row partition `p` codes, and its coefficients
    /// into `blocks` (zero on entry) and the kind of transform each block needs into `kinds`.
    #[inline(always)]
    fn parse_mb(
        &mut self,
        mbx: usize,
        p: usize,
        blocks: &mut [i16; 384],
        kinds: &mut [u8; 24],
    ) -> Result<MacroBlock, DecodingError> {
        let mb = self.read_macroblock_header(mbx)?;
        self.parse_residuals(mb, mbx, p, blocks, kinds)
    }

    /// [`parse_mb`](Self::parse_mb) of a macroblock whose modes are read.
    #[inline(always)]
    fn parse_residuals(
        &mut self,
        mut mb: MacroBlock,
        mbx: usize,
        p: usize,
        blocks: &mut [i16; 384],
        kinds: &mut [u8; 24],
    ) -> Result<MacroBlock, DecodingError> {
        if !mb.coeffs_skipped {
            self.partitions[p].top_up(&mut *self.source)?;
            self.read_residual_data(&mut mb, mbx, p, blocks, kinds)?;
        } else if self.partitions[p].is_past_eof() {
            // libwebp checks the token partition after every macroblock, one that reads
            // nothing from it included.
            return Err(DecodingError::BitStreamError);
        } else {
            let top = &mut self.top[mbx].complexity;
            if mb.luma_mode != LumaMode::B {
                self.left.complexity[0] = 0;
                top[0] = 0;
            }
            self.left.complexity[1..].fill(0);
            top[1..].fill(0);
        }
        Ok(mb)
    }

    /// Moves what reading the macroblocks needs into a decoder of its own, which reads them
    /// ahead while this one reconstructs them.
    fn split_parser(&mut self) -> Vp8Decoder<'a> {
        let mut p = Vp8Decoder::new();
        std::mem::swap(&mut p.source, &mut self.source);
        std::mem::swap(&mut p.b, &mut self.b);
        std::mem::swap(&mut p.partitions, &mut self.partitions);
        std::mem::swap(&mut p.pos_probs, &mut self.pos_probs);
        std::mem::swap(&mut p.top, &mut self.top);
        p.mbwidth = self.mbwidth;
        p.mbheight = self.mbheight;
        p.num_partitions = self.num_partitions;
        p.segments_enabled = self.segments_enabled;
        p.segments_update_map = self.segments_update_map;
        p.segment = self.segment;
        p.segment_probs = self.segment_probs;
        p.prob_skip_false = self.prob_skip_false;
        p.left = self.left;
        p
    }

    /// Moves what reading the modes of the macroblocks needs into a decoder of its own, which
    /// reads them ahead of their residuals.
    fn split_modes(&mut self) -> Vp8Decoder<'a> {
        let mut m = Vp8Decoder::new();
        std::mem::swap(&mut m.b, &mut self.b);
        m.top = self.top.clone();
        m.mbwidth = self.mbwidth;
        m.mbheight = self.mbheight;
        m.segments_enabled = self.segments_enabled;
        m.segments_update_map = self.segments_update_map;
        m.segment_probs = self.segment_probs;
        m.prob_skip_false = self.prob_skip_false;
        m
    }
}

/// Macroblocks read ahead, in raster order: their modes and kinds of transform, the coefficients
/// of their blocks with any (only the DC of a DC-only block), and the error the read stopped at.
#[derive(Default)]
struct Parsed {
    mbs: Vec<(MacroBlock, [u8; 24])>,
    coeffs: Vec<i16>,
    error: Option<DecodingError>,
}

/// The bytes a [`Parsed`] unit holds: half of them for macroblocks, half for coefficients.
const PARSED_UNIT_BYTES: usize = 16 << 10;
const PARSED_MBS: usize = PARSED_UNIT_BYTES / 2 / std::mem::size_of::<(MacroBlock, [u8; 24])>();
const PARSED_COEFFS: usize = PARSED_UNIT_BYTES / 2 / 2;

impl Parsed {
    /// Empties the unit, sized for [`PARSED_UNIT_BYTES`] so that it never grows.
    fn reset(&mut self) {
        self.mbs.clear();
        self.coeffs.clear();
        self.error = None;
        self.mbs.reserve_exact(PARSED_MBS);
        self.coeffs.reserve_exact(PARSED_COEFFS);
    }

    /// Whether the unit holds another macroblock whatever its coefficients.
    fn has_room(&self) -> bool {
        self.mbs.len() < PARSED_MBS && self.coeffs.len() + 384 <= PARSED_COEFFS
    }

    /// Adds a macroblock, moving its coefficients out of `blocks`, which it clears as the
    /// reconstruction does.
    fn push(&mut self, mb: MacroBlock, kinds: &[u8; 24], blocks: &mut [i16; 384]) {
        for (block, &kind) in blocks.as_chunks_mut::<16>().0.iter_mut().zip(kinds) {
            match kind {
                transform::NO_COEFFS => {}
                transform::DC_ONLY => {
                    self.coeffs.push(block[0]);
                    block[0] = 0;
                }
                _ => {
                    self.coeffs.extend_from_slice(block);
                    block.fill(0);
                }
            }
        }
        self.mbs.push((mb, *kinds));
    }
}

/// Reads the macroblocks of a frame ahead of their reconstruction, a [`Parsed`] unit at a time.
struct Parser {
    dec: Vp8Decoder<'static>,
    /// The runner and units ahead a [`ModeRows`] reads the modes with once the parser starts,
    /// after the parser has its helper.
    modes_on: Option<(Runner, usize)>,
    modes: Option<Box<ModesAhead>>,
    mbx: usize,
    mby: usize,
    done: bool,
}

impl Produce for Parser {
    type Unit = Parsed;

    fn produce(&mut self, spare: Option<Parsed>) -> Option<Parsed> {
        let (mbwidth, mbheight) = (usize::from(self.dec.mbwidth), usize::from(self.dec.mbheight));
        if self.done || self.mby == mbheight {
            return None;
        }
        if let Some((runner, ahead)) = self.modes_on.take() {
            let modes = ModeRows { dec: self.dec.split_modes(), mby: 0, rows: mode_unit_rows(mbwidth), done: false };
            let ahead = Ahead::new(modes, runner, ahead);
            self.modes = Some(Box::new(ModesAhead { ahead, unit: ModeUnit::default(), at: 0, width: mbwidth }));
        }
        let modes = self.modes.as_deref_mut()?;
        let mut unit = spare.unwrap_or_default();
        unit.reset();
        let mut blocks = [0i16; 384];
        while self.mby < mbheight && unit.has_room() {
            if self.mbx == 0 {
                self.dec.left = PreviousMacroBlock::default();
                if let Err(e) = modes.next_row(&mut self.dec.mode_row) {
                    unit.error = Some(e);
                    self.done = true;
                    break;
                }
            }
            let p = self.mby % usize::from(self.dec.num_partitions);
            let mut kinds = [transform::NO_COEFFS; 24];
            let mb = self.dec.mode_row[self.mbx];
            match self.dec.parse_residuals(mb, self.mbx, p, &mut blocks, &mut kinds) {
                Ok(mb) => unit.push(mb, &kinds, &mut blocks),
                Err(e) => {
                    unit.error = Some(e);
                    self.done = true;
                    break;
                }
            }
            self.mbx += 1;
            if self.mbx == mbwidth {
                self.mbx = 0;
                self.mby += 1;
            }
        }
        Some(unit)
    }
}

/// Reads the modes of macroblock rows ahead of their residuals, `rows` at a time.
struct ModeRows {
    dec: Vp8Decoder<'static>,
    mby: usize,
    rows: usize,
    done: bool,
}

/// The modes of macroblock rows, and the error the read stopped at.
#[derive(Default)]
struct ModeUnit {
    mbs: Vec<MacroBlock>,
    error: Option<DecodingError>,
}

/// About how many bytes of modes a [`ModeUnit`] holds: at least one row.
const MODE_BYTES: usize = 8 << 10;

fn mode_unit_rows(mbwidth: usize) -> usize {
    (MODE_BYTES / (mbwidth * std::mem::size_of::<MacroBlock>())).max(1)
}

impl Produce for ModeRows {
    type Unit = ModeUnit;

    fn produce(&mut self, spare: Option<ModeUnit>) -> Option<ModeUnit> {
        let mbheight = usize::from(self.dec.mbheight);
        if self.done || self.mby == mbheight {
            return None;
        }
        let mut unit = spare.unwrap_or_default();
        unit.mbs.clear();
        unit.mbs.reserve_exact(self.rows * usize::from(self.dec.mbwidth));
        unit.error = None;
        for _ in 0..self.rows.min(mbheight - self.mby) {
            self.dec.left = PreviousMacroBlock::default();
            if let Err(e) = self.dec.read_mode_row() {
                unit.error = Some(e);
                self.done = true;
                break;
            }
            unit.mbs.extend_from_slice(&self.dec.mode_row);
            self.mby += 1;
        }
        Some(unit)
    }
}

/// The mode rows a [`ModeRows`] reads ahead, handed out one at a time.
struct ModesAhead {
    ahead: Ahead<ModeRows>,
    unit: ModeUnit,
    at: usize,
    width: usize,
}

impl ModesAhead {
    /// The modes of the next row into `row`; the error the read stopped at once it gets there.
    fn next_row(&mut self, row: &mut Vec<MacroBlock>) -> Result<(), DecodingError> {
        while self.at == self.unit.mbs.len() {
            if let Some(e) = self.unit.error.take() {
                return Err(e);
            }
            let spent = std::mem::take(&mut self.unit);
            self.unit = self.ahead.next(Some(spent)).ok_or(DecodingError::BitStreamError)?;
            self.at = 0;
        }
        row.clear();
        row.extend_from_slice(&self.unit.mbs[self.at..][..self.width]);
        self.at += self.width;
        Ok(())
    }
}

/// The macroblocks a [`Parser`] reads ahead, handed out one at a time.
struct ParsedAhead {
    ahead: Ahead<Parser>,
    unit: Parsed,
    at: usize,
    pos: usize,
}

impl ParsedAhead {
    /// The next macroblock, its coefficients into `blocks` (zero on entry) and kinds of
    /// transform into `kinds`; the error the read stopped at once it gets there.
    fn next(&mut self, blocks: &mut [i16; 384], kinds: &mut [u8; 24]) -> Result<MacroBlock, DecodingError> {
        while self.at == self.unit.mbs.len() {
            if let Some(e) = self.unit.error.take() {
                return Err(e);
            }
            let spent = std::mem::take(&mut self.unit);
            self.unit = self.ahead.next(Some(spent)).ok_or(DecodingError::BitStreamError)?;
            self.at = 0;
            self.pos = 0;
        }
        let (mb, k) = self.unit.mbs[self.at];
        self.at += 1;
        *kinds = k;
        let coeffs = &self.unit.coeffs;
        for (block, &kind) in blocks.as_chunks_mut::<16>().0.iter_mut().zip(&k) {
            match kind {
                transform::NO_COEFFS => {}
                transform::DC_ONLY => {
                    block[0] = coeffs[self.pos];
                    self.pos += 1;
                }
                _ => {
                    block.copy_from_slice(&coeffs[self.pos..self.pos + 16]);
                    self.pos += 16;
                }
            }
        }
        Ok(mb)
    }
}

/// Decodes a frame a macroblock row at a time into bands of finished rows, top to bottom, as
/// RGB (`BPP` 3) or RGBA (`BPP` 4, alpha 255). Holds about 23 rows of YUV and RGB instead of
/// the whole frame.
pub(crate) struct BandDecoder<'a, const BPP: usize> {
    dec: Vp8Decoder<'a>,
    upsampling: UpsamplingMethod,
    row: Vec<MacroBlock>,
    mby: usize,
    emitted: usize,
    rgb: Vec<u8>,
    u_full: Vec<u8>,
    v_full: Vec<u8>,
    uv_emitted: usize,
    bypass_filter: bool,
    /// The macroblocks read ahead on helpers, where they are.
    parsed: Option<Box<ParsedAhead>>,
}

/// A plane of a frame's pixels.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Channel {
    Y,
    U,
    V,
}

impl<'a, const BPP: usize> BandDecoder<'a, BPP> {
    /// Reads the frame header of the VP8 bitstream at `start` of `source`, a bitstream that
    /// runs to the end of the data at `end`.
    pub(crate) fn new(source: Box<dyn ReadSeek + Send + 'a>, start: u64, end: u64, upsampling: UpsamplingMethod) -> Result<Self, DecodingError> {
        let mut dec = Vp8Decoder::new();
        dec.source = source;
        dec.band_rows = true;
        dec.read_frame_header(start, end)?;
        let width = usize::from(dec.frame.width);
        let mut rgb = vec![0u8; (BAND_KEEP_Y + 16) * width * BPP];
        if BPP == 4 {
            for px in rgb.chunks_exact_mut(4) {
                px[3] = 255;
            }
        }
        Ok(Self {
            row: Vec::with_capacity(usize::from(dec.mbwidth)),
            dec,
            upsampling,
            mby: 0,
            emitted: 0,
            rgb,
            u_full: vec![0; width],
            v_full: vec![0; width],
            uv_emitted: 0,
            bypass_filter: false,
            parsed: None,
        })
    }

    /// Leaves out the loop filter, as libwebp does for a decode scaled well down.
    pub(crate) fn bypass_filter(&mut self) {
        self.bypass_filter = true;
    }

    /// Decodes the next macroblock row and hands the rows of each plane it finished to `f`,
    /// top to bottom; the luma rows final so far, or `None` after the last row.
    pub(crate) fn next_planes(
        &mut self,
        mut f: impl FnMut(Channel, &[u8]),
    ) -> Result<Option<usize>, DecodingError> {
        let dec = &mut self.dec;
        let mbheight = usize::from(dec.mbheight);
        if self.mby == mbheight {
            return Ok(None);
        }
        let width = usize::from(dec.frame.width);
        let height = usize::from(dec.frame.height);
        let luma_w = usize::from(dec.mbwidth) * 16;
        let chroma_w = luma_w / 2;
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let mby = self.mby;
        self.mby += 1;
        self.decode_row(mby)?;
        let dec = &self.dec;
        let last = mby + 1 == mbheight;
        let end = if last { height } else { (mby * 16 + 16 - BAND_KEEP_Y).min(height) };
        let uv_end = if last { chroma_height } else { end.div_ceil(2).min(chroma_height) };
        for y in self.emitted..end {
            f(Channel::Y, &dec.frame.ybuf[y.wrapping_sub(dec.y_base) * luma_w..][..width]);
        }
        for c in self.uv_emitted..uv_end {
            f(Channel::U, &dec.frame.ubuf[c.wrapping_sub(dec.uv_base) * chroma_w..][..chroma_width]);
        }
        for c in self.uv_emitted..uv_end {
            f(Channel::V, &dec.frame.vbuf[c.wrapping_sub(dec.uv_base) * chroma_w..][..chroma_width]);
        }
        self.emitted = self.emitted.max(end);
        self.uv_emitted = self.uv_emitted.max(uv_end);
        Ok(Some(self.emitted))
    }

    fn decode_row(&mut self, mby: usize) -> Result<(), DecodingError> {
        let dec = &mut self.dec;
        let luma_w = usize::from(dec.mbwidth) * 16;
        let chroma_w = luma_w / 2;
        if mby > 0 {
            dec.frame.ybuf.copy_within(16 * luma_w.., 0);
            dec.frame.ubuf.copy_within(8 * chroma_w.., 0);
            dec.frame.vbuf.copy_within(8 * chroma_w.., 0);
        }
        dec.y_base = (mby * 16).wrapping_sub(BAND_KEEP_Y);
        dec.uv_base = (mby * 8).wrapping_sub(BAND_KEEP_UV);
        dec.decode_mb_row(mby, &mut self.row, self.parsed.as_deref_mut())?;
        if !self.bypass_filter {
            for (mbx, mb) in self.row.iter().enumerate() {
                dec.loop_filter(mbx, mby, mb);
            }
        }
        Ok(())
    }

    pub(crate) fn dimensions(&self) -> (u16, u16) {
        (self.dec.frame.width, self.dec.frame.height)
    }

    /// Whether the frame has macroblocks enough to read ahead.
    fn reads_ahead(&self) -> bool {
        usize::from(self.dec.mbwidth) * usize::from(self.dec.mbheight) >= MIN_AHEAD_MBS
    }

    /// What reading `ahead` units of macroblocks and of their modes ahead holds beyond decoding
    /// them in place, 0 where the frame is too small to: per part the units ahead, the one being
    /// read and the one taken.
    pub(crate) fn ahead_bytes(&self, ahead: usize) -> usize {
        if !self.reads_ahead() {
            return 0;
        }
        let mbwidth = usize::from(self.dec.mbwidth);
        let modes = mode_unit_rows(mbwidth) * mbwidth * std::mem::size_of::<MacroBlock>();
        (ahead.max(1) + 2) * (PARSED_UNIT_BYTES + modes)
    }

    /// Decodes macroblock rows until a band of rows is final and returns its first row and
    /// pixels, or `None` after the last band.
    pub(crate) fn next_band(&mut self) -> Result<Option<(u32, &mut [u8])>, DecodingError> {
        let Some((first, end)) = self.next_rows()? else {
            return Ok(None);
        };
        let len = (end - first) * usize::from(self.dec.frame.width) * BPP;
        let mut rgb = std::mem::take(&mut self.rgb);
        self.convert_rows(first, end, &mut rgb[..len]);
        self.rgb = rgb;
        Ok(Some((first as u32, &mut self.rgb[..len])))
    }

    /// [`next_band`](Self::next_band) straight into its rows of `frame`, the whole frame's
    /// rows: the first row and the row past the band, or `None` after the last band.
    pub(crate) fn next_band_into(&mut self, frame: &mut [u8]) -> Result<Option<(usize, usize)>, DecodingError> {
        let Some((first, end)) = self.next_rows()? else {
            return Ok(None);
        };
        let stride = usize::from(self.dec.frame.width) * BPP;
        self.convert_rows(first, end, &mut frame[first * stride..end * stride]);
        Ok(Some((first, end)))
    }

    /// Decodes macroblock rows until a band of rows is final: its first row and the row past
    /// it, or `None` after the last band.
    fn next_rows(&mut self) -> Result<Option<(usize, usize)>, DecodingError> {
        let mbheight = usize::from(self.dec.mbheight);
        let height = usize::from(self.dec.frame.height);
        while self.mby < mbheight {
            let mby = self.mby;
            self.mby += 1;
            self.decode_row(mby)?;
            let end = if mby + 1 == mbheight {
                height
            } else {
                (mby * 16 + 16 - BAND_KEEP_Y).min(height)
            };
            if end > self.emitted {
                let first = self.emitted;
                self.emitted = end;
                return Ok(Some((first, end)));
            }
        }
        Ok(None)
    }

    /// Converts rows `first..end`, still held in the planes, to RGB(A) in `out`.
    fn convert_rows(&mut self, first: usize, end: usize, out: &mut [u8]) {
        let dec = &self.dec;
        let width = usize::from(dec.frame.width);
        let height = usize::from(dec.frame.height);
        let luma_w = usize::from(dec.mbwidth) * 16;
        let chroma_w = luma_w / 2;
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let (y_base, uv_base) = (dec.y_base, dec.uv_base);
        let (ybuf, ubuf, vbuf) = (&dec.frame.ybuf, &dec.frame.ubuf, &dec.frame.vbuf);
        let y_row = |y: usize| &ybuf[y.wrapping_sub(y_base) * luma_w..][..width];
        let u_row = |c: usize| &ubuf[c.wrapping_sub(uv_base) * chroma_w..][..chroma_width];
        let v_row = |c: usize| &vbuf[c.wrapping_sub(uv_base) * chroma_w..][..chroma_width];
        for (y, out) in (first..end).zip(out.chunks_exact_mut(width * BPP)) {
            let (near, far) = match self.upsampling {
                UpsamplingMethod::Simple => {
                    yuv::fill_rgba_row_simple::<BPP>(y_row(y), u_row(y / 2), v_row(y / 2), out);
                    continue;
                }
                UpsamplingMethod::Bilinear => {
                    if y == 0 {
                        (0, 0)
                    } else if y + 1 == height && height % 2 == 0 {
                        (chroma_height - 1, chroma_height - 1)
                    } else {
                        let k = y.div_ceil(2);
                        if y % 2 == 1 {
                            (k - 1, k)
                        } else {
                            (k, k - 1)
                        }
                    }
                }
            };
            yuv::upsample_chroma_row(u_row(near), u_row(far), &mut self.u_full);
            yuv::upsample_chroma_row(v_row(near), v_row(far), &mut self.v_full);
            yuv::fill_row_yuv::<BPP>(out, y_row(y), &self.u_full, &self.v_full);
        }
    }
}

impl<const BPP: usize> BandDecoder<'static, BPP> {
    /// Reads the macroblocks ahead on `runner`'s helpers, at most `ahead` units of
    /// [`PARSED_UNIT_BYTES`] past the one taken; before the first row.
    pub(crate) fn run_ahead(&mut self, runner: Runner, ahead: usize) {
        if self.mby > 0 || self.parsed.is_some() || !self.reads_ahead() {
            return;
        }
        let modes_on = Some((runner.clone(), ahead));
        let parser = Parser { dec: self.dec.split_parser(), modes_on, modes: None, mbx: 0, mby: 0, done: false };
        self.parsed = Some(Box::new(ParsedAhead {
            ahead: Ahead::new(parser, runner, ahead),
            unit: Parsed::default(),
            at: 0,
            pos: 0,
        }));
    }
}

/// The fewest macroblocks a frame reads ahead: about a millisecond of work.
const MIN_AHEAD_MBS: usize = 256;

/// Rows of the previous macroblock row a band keeps: the luma rows the next row's loop
/// filter changes plus those whose fancy upsampling needs the next chroma rows, and the
/// chroma rows the filter reads.
const BAND_KEEP_Y: usize = 7;
/// The most rows a band of [`BandDecoder::next_band`] has.
pub(crate) const MAX_BAND_ROWS: usize = BAND_KEEP_Y + 16;
const BAND_KEEP_UV: usize = 4;

/// Loop filter parameters of one macroblock.
#[derive(Clone, Copy)]
struct MbFilter {
    level: u8,
    interior: u8,
    hev: u8,
    subblocks: bool,
}

/// The planes a macroblock row is filtered in: rows from `y_base` / `uv_base` on.
struct PlaneRefs<'a> {
    y: &'a mut [u8],
    u: &'a mut [u8],
    v: &'a mut [u8],
    y_base: usize,
    uv_base: usize,
    luma_w: usize,
    chroma_w: usize,
    filter_type: bool,
}

fn filter_macroblock(p: &mut PlaneRefs, mbx: usize, mby: usize, f: MbFilter) {
    if f.level == 0 {
        return;
    }
    let limit = 2 * f.level + f.interior;
    let (ilevel, hev) = (f.interior, f.hev);
    let (lw, cw) = (p.luma_w, p.chroma_w);
    let y = (mby * 16).wrapping_sub(p.y_base) * lw + mbx * 16;
    if p.filter_type {
        if mbx > 0 {
            loop_filter::simple_h16(p.y, y, lw, limit + 4);
        }
        if f.subblocks {
            loop_filter::simple_h16i(p.y, y, lw, limit);
        }
        if mby > 0 {
            loop_filter::simple_v16(p.y, y, lw, limit + 4);
        }
        if f.subblocks {
            loop_filter::simple_v16i(p.y, y, lw, limit);
        }
    } else {
        let uv = (mby * 8).wrapping_sub(p.uv_base) * cw + mbx * 8;
        if mbx > 0 {
            loop_filter::h16(p.y, y, lw, limit + 4, ilevel, hev);
            loop_filter::h8(p.u, p.v, uv, cw, limit + 4, ilevel, hev);
        }
        if f.subblocks {
            loop_filter::h16i(p.y, y, lw, limit, ilevel, hev);
            loop_filter::h8i(p.u, p.v, uv, cw, limit, ilevel, hev);
        }
        if mby > 0 {
            loop_filter::v16(p.y, y, lw, limit + 4, ilevel, hev);
            loop_filter::v8(p.u, p.v, uv, cw, limit + 4, ilevel, hev);
        }
        if f.subblocks {
            loop_filter::v16i(p.y, y, lw, limit, ilevel, hev);
            loop_filter::v8i(p.u, p.v, uv, cw, limit, ilevel, hev);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_level_clamps_once_after_the_deltas() {
        let mut d = Vp8Decoder::new();
        d.frame.filter_level = 60;
        d.segments_enabled = true;
        d.segment[0].delta_values = true;
        d.segment[0].loopfilter_level = 20;
        d.loop_filter_adjustments_enabled = true;
        d.ref_delta[0] = -30;
        d.mode_delta[0] = -8;
        let mut mb = MacroBlock::default();
        assert_eq!(d.calculate_filter_parameters(&mb).0, 50);
        mb.luma_mode = LumaMode::B;
        assert_eq!(d.calculate_filter_parameters(&mb).0, 42);
        d.segment[0].loopfilter_level = -63;
        d.ref_delta[0] = 10;
        assert_eq!(d.calculate_filter_parameters(&mb).0, 0);
    }
}
