//! Lossy (VP8) encoder ported from libwebp 1.6.0, producing the same bitstream as
//! `WebPEncode` with the default preset for any `method`.
//!
//! libwebp: Copyright 2010 Google Inc. All Rights Reserved. Distributed under libwebp's
//! BSD 3-Clause license and patent grant, reproduced in LICENSE-LIBWEBP.

mod bit_writer;
mod cost;
pub(crate) mod csp;
mod dsp;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse2;
mod tables;

use bit_writer::BitWriter;
use cost::{bit_cost, level_cost, Proba, Residual, TokenBuffer};
use csp::Source;
use dsp::*;
use tables::{
    AC_TABLE, AC_TABLE2, BMODES_PROBA, DC_TABLE, ENC_BANDS, FIXED_COSTS_I4, LEVELS_FROM_DELTA,
};

const NUM_MB_SEGMENTS: usize = 4;
const NUM_BMODES: usize = 10;
const NUM_PRED_MODES: usize = 4;
const B_DC_PRED: u8 = 0;
const MAX_COST: i64 = 0x7f_ffff_ffff_ffff;
const RD_DISTO_MULT: i64 = 256;
const ERROR_DIFFUSION_QUALITY: f32 = 98.;
const VP8_MAX_PARTITION0_SIZE: u64 = 1 << 19;
const PARTITION0_SIZE_LIMIT: u64 = (VP8_MAX_PARTITION0_SIZE - 2048) << 11;

const FIXED_COSTS_UV: [u16; 4] = [302, 984, 439, 642];
const FIXED_COSTS_I16: [u16; 4] = [663, 919, 872, 919];

const WEIGHT_Y: [u16; 16] = [38, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2];
const WEIGHT_TRELLIS: [u16; 16] = [30, 27, 19, 11, 27, 24, 17, 10, 19, 17, 12, 8, 11, 10, 8, 6];

const BIAS_MATRICES: [[u32; 2]; 3] = [[96, 110], [96, 108], [110, 115]];
const FREQ_SHARPENING: [u16; 16] = [
    0, 30, 60, 90, 30, 60, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90,
];
const SHARPEN_BITS: u32 = 11;

const FLATNESS_LIMIT_I16: i32 = 0;
const FLATNESS_LIMIT_I4: i32 = 3;
const FLATNESS_LIMIT_UV: i32 = 2;
const FLATNESS_PENALTY: i64 = 140;

const TOP_LEFT_I4: [usize; 16] = [17, 21, 25, 29, 13, 17, 21, 25, 9, 13, 17, 21, 5, 9, 13, 17];

/// The `WebPConfig` fields the lossy encoder reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    pub(crate) quality: f32,
    pub(crate) method: i32,
    pub(crate) sns_strength: i32,
    pub(crate) filter_strength: i32,
    pub(crate) filter_sharpness: i32,
    pub(crate) filter_type: i32,
    pub(crate) segments: i32,
    pub(crate) partition_limit: i32,
    pub(crate) preprocessing: i32,
    /// Keep the converted picture between passes instead of converting it in each.
    pub(crate) keep_strips: bool,
}

impl Config {
    /// `WebPConfigInit` with `WEBP_PRESET_DEFAULT`.
    pub(crate) fn new(quality: f32, method: i32) -> Self {
        Self {
            quality,
            method,
            sns_strength: 50,
            filter_strength: 60,
            filter_sharpness: 0,
            filter_type: 1,
            segments: 4,
            partition_limit: 0,
            preprocessing: 0,
            keep_strips: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum RdLevel {
    None,
    Basic,
    Trellis,
    TrellisAll,
}

#[derive(Clone, Copy, Default)]
struct MbInfo {
    /// 0 = intra4x4, 1 = intra16x16
    mb_type: u8,
    uv_mode: u8,
    skip: bool,
    segment: u8,
    alpha: u8,
}

#[derive(Clone, Copy, Default)]
struct SegmentInfo {
    y1: Matrix,
    y2: Matrix,
    uv: Matrix,
    alpha: i32,
    beta: i32,
    quant: i32,
    fstrength: i32,
    max_edge: i32,
    min_disto: i64,
    lambda_i16: i32,
    lambda_i4: i32,
    lambda_uv: i32,
    lambda_mode: i32,
    lambda_trellis_i16: i32,
    lambda_trellis_i4: i32,
    lambda_trellis_uv: i32,
    tlambda: i32,
    i4_penalty: i64,
}

#[derive(Clone)]
struct ModeScore {
    d: i64,
    sd: i64,
    h: i64,
    r: i64,
    score: i64,
    y_dc_levels: [i16; 16],
    y_ac_levels: [[i16; 16]; 16],
    uv_levels: [[i16; 16]; 8],
    mode_i16: i32,
    modes_i4: [u8; 16],
    mode_uv: i32,
    nz: u32,
    derr: [[i8; 3]; 2],
}

impl Default for ModeScore {
    fn default() -> Self {
        Self {
            d: 0,
            sd: 0,
            h: 0,
            r: 0,
            score: MAX_COST,
            y_dc_levels: [0; 16],
            y_ac_levels: [[0; 16]; 16],
            uv_levels: [[0; 16]; 8],
            mode_i16: 0,
            modes_i4: [0; 16],
            mode_uv: 0,
            nz: 0,
            derr: [[0; 3]; 2],
        }
    }
}

impl ModeScore {
    fn init_score(&mut self) {
        self.d = 0;
        self.sd = 0;
        self.r = 0;
        self.h = 0;
        self.nz = 0;
        self.score = MAX_COST;
    }

    fn copy_score(&mut self, src: &ModeScore) {
        self.d = src.d;
        self.sd = src.sd;
        self.r = src.r;
        self.h = src.h;
        self.nz = src.nz;
        self.score = src.score;
    }

    fn set_rd_score(&mut self, lambda: i32) {
        self.score = (self.r + self.h) * i64::from(lambda) + RD_DISTO_MULT * (self.d + self.sd);
    }
}

/// The rate/distortion part of a `ModeScore`, for candidates whose levels live elsewhere.
#[derive(Clone, Copy)]
struct Score {
    d: i64,
    sd: i64,
    h: i64,
    r: i64,
    score: i64,
    nz: u32,
}

impl Score {
    const INIT: Score = Score {
        d: 0,
        sd: 0,
        h: 0,
        r: 0,
        score: MAX_COST,
        nz: 0,
    };

    fn set_rd_score(&mut self, lambda: i32) {
        self.score = (self.r + self.h) * i64::from(lambda) + RD_DISTO_MULT * (self.d + self.sd);
    }
}

impl ModeScore {
    fn add(&mut self, src: &Score) {
        self.d += src.d;
        self.sd += src.sd;
        self.r += src.r;
        self.h += src.h;
        self.nz |= src.nz;
        self.score += src.score;
    }
}

type DError = [[i8; 2]; 2];

struct Encoder<'a> {
    config: Config,
    src: Source<'a>,

    // headers
    filter_simple: bool,
    filter_level: i32,
    filter_sharpness: i32,
    num_segments: usize,
    update_map: bool,
    segment_hdr_size: i64,
    profile: u32,

    mb_w: usize,
    mb_h: usize,
    preds_w: usize,

    dqm: [SegmentInfo; NUM_MB_SEGMENTS],
    base_quant: i32,
    alpha: i32,
    uv_alpha: i32,
    dq_y1_dc: i32,
    dq_y2_dc: i32,
    dq_y2_ac: i32,
    dq_uv_dc: i32,
    dq_uv_ac: i32,

    proba: Proba,
    tokens: TokenBuffer,

    method: i32,
    rd_opt_level: RdLevel,
    max_i4_header_bits: i32,
    mb_header_limit: i64,

    mb_info: Vec<MbInfo>,
    /// Intra modes per 4x4 block, with a border row and column; see `preds_origin`.
    preds: Vec<u8>,
    nz: Vec<u32>,
    y_top: Vec<u8>,
    uv_top: Vec<u8>,
    top_derr: Option<Vec<DError>>,

    it: Iter,
}

/// libwebp's `VP8EncIterator`.
struct Iter {
    x: usize,
    y: usize,
    yuv_in: Box<[u8; YUV_SIZE]>,
    yuv_out: Box<[u8; YUV_SIZE]>,
    yuv_out2: Box<[u8; YUV_SIZE]>,
    yuv_p: Box<[u8; PRED_SIZE]>,
    i4_tmp: [u8; 4 * BPS],
    i4_boundary: [u8; 40],
    i4_top: usize,
    i4: usize,
    top_nz: [i32; 9],
    left_nz: [i32; 9],
    do_trellis: bool,
    left_derr: DError,
    /// y_left at 1 (top-left at 0), u_left at 33, v_left at 49.
    left: [u8; 64],
    /// Source samples above the macroblock during analysis.
    analysis_top: Option<[u8; 32]>,
}

impl Iter {
    fn new() -> Self {
        Self {
            x: 0,
            y: 0,
            yuv_in: Box::new([0; YUV_SIZE]),
            yuv_out: Box::new([0; YUV_SIZE]),
            yuv_out2: Box::new([0; YUV_SIZE]),
            yuv_p: Box::new([0; PRED_SIZE]),
            i4_tmp: [0; 4 * BPS],
            i4_boundary: [0; 40],
            i4_top: 0,
            i4: 0,
            top_nz: [0; 9],
            left_nz: [0; 9],
            do_trellis: false,
            left_derr: [[0; 2]; 2],
            left: [0; 64],
            analysis_top: None,
        }
    }

    fn init_left(&mut self, has_derr: bool) {
        let tl = if self.y > 0 { 129 } else { 127 };
        self.left[0] = tl;
        self.left[32] = tl;
        self.left[48] = tl;
        self.left[1..17].fill(129);
        self.left[33..41].fill(129);
        self.left[49..57].fill(129);
        self.left_nz[8] = 0;
        if has_derr {
            self.left_derr = [[0; 2]; 2];
        }
    }

    fn swap_out(&mut self) {
        std::mem::swap(&mut self.yuv_out, &mut self.yuv_out2);
    }
}

impl<'a> Encoder<'a> {
    fn new(config: Config, src: Source<'a>) -> Self {
        let mb_w = src.width.div_ceil(16);
        let mb_h = src.height.div_ceil(16);
        let preds_w = 4 * mb_w + 1;
        let preds_h = 4 * mb_h + 1;
        let use_filter = config.filter_strength > 0;
        let method = config.method;
        let limit = 100 - config.partition_limit;
        let rd_opt_level = if method >= 6 {
            RdLevel::TrellisAll
        } else if method >= 5 {
            RdLevel::Trellis
        } else if method >= 3 {
            RdLevel::Basic
        } else {
            RdLevel::None
        };
        let mut enc = Self {
            config,
            src,
            filter_simple: true,
            filter_level: 0,
            filter_sharpness: 0,
            num_segments: config.segments as usize,
            update_map: config.segments > 1,
            segment_hdr_size: 0,
            profile: if use_filter {
                if config.filter_type == 1 {
                    0
                } else {
                    1
                }
            } else {
                2
            },
            mb_w,
            mb_h,
            preds_w,
            dqm: [SegmentInfo::default(); NUM_MB_SEGMENTS],
            base_quant: 0,
            alpha: 0,
            uv_alpha: 0,
            dq_y1_dc: 0,
            dq_y2_dc: 0,
            dq_y2_ac: 0,
            dq_uv_dc: 0,
            dq_uv_ac: 0,
            proba: Proba::new(),
            tokens: TokenBuffer::default(),
            method,
            rd_opt_level,
            max_i4_header_bits: 256 * 16 * 16 * (limit * limit) / (100 * 100),
            mb_header_limit: 256 * 510 * 8 * 1024 / (mb_w * mb_h) as i64,
            mb_info: vec![MbInfo::default(); mb_w * mb_h],
            preds: vec![B_DC_PRED; preds_w * preds_h],
            nz: vec![0; mb_w + 1],
            y_top: vec![0; mb_w * 16],
            uv_top: vec![0; mb_w * 16],
            top_derr: (config.quality <= ERROR_DIFFUSION_QUALITY).then(|| vec![[[0; 2]; 2]; mb_w]),
            it: Iter::new(),
        };
        enc.iterator_reset();
        enc
    }

    // ---------------------------------------------------------------------------------------
    // Iterator

    fn preds_origin(&self) -> usize {
        1 + self.preds_w
    }

    /// Index in `preds` of the top-left 4x4 block of the current macroblock.
    fn preds_index(&self) -> usize {
        self.preds_origin() + self.it.y * 4 * self.preds_w + self.it.x * 4
    }

    fn mb_index(&self) -> usize {
        self.it.y * self.mb_w + self.it.x
    }

    fn mb(&self) -> &MbInfo {
        &self.mb_info[self.mb_index()]
    }

    fn mb_mut(&mut self) -> &mut MbInfo {
        let i = self.mb_index();
        &mut self.mb_info[i]
    }

    fn iterator_set_row(&mut self, y: usize) {
        self.it.x = 0;
        self.it.y = y;
        self.it.init_left(self.top_derr.is_some());
    }

    fn iterator_reset(&mut self) {
        self.iterator_set_row(0);
        self.y_top.fill(127);
        self.uv_top.fill(127);
        self.nz.fill(0);
        if let Some(top_derr) = self.top_derr.as_mut() {
            top_derr.fill([[0; 2]; 2]);
        }
        self.it.do_trellis = false;
    }

    /// Returns false once past the last macroblock.
    fn iterator_next(&mut self) -> bool {
        self.it.x += 1;
        if self.it.x == self.mb_w {
            self.iterator_set_row(self.it.y + 1);
        }
        self.it.y < self.mb_h
    }

    fn iterator_import(&mut self, with_boundary: bool) {
        let (x, y) = (self.it.x, self.it.y);
        self.src.load(y);
        let pic = &self.src;
        let w = (pic.width - x * 16).min(16);
        let h = (pic.height - y * 16).min(16);
        let uv_w = w.div_ceil(2);
        let uv_h = h.div_ceil(2);
        // Strip row 1 is the macroblock's first row.
        let ysrc = pic.y_stride + x * 16;
        let uvsrc = pic.uv_stride + x * 8;
        import_block::<16>(
            &pic.y[ysrc..],
            pic.y_stride,
            &mut self.it.yuv_in[Y_OFF..],
            w,
            h,
        );
        import_block::<8>(
            &pic.u[uvsrc..],
            pic.uv_stride,
            &mut self.it.yuv_in[U_OFF..],
            uv_w,
            uv_h,
        );
        import_block::<8>(
            &pic.v[uvsrc..],
            pic.uv_stride,
            &mut self.it.yuv_in[V_OFF..],
            uv_w,
            uv_h,
        );

        if !with_boundary {
            return;
        }
        let left = &mut self.it.left;
        if x == 0 {
            let tl = if y > 0 { 129 } else { 127 };
            left[0] = tl;
            left[32] = tl;
            left[48] = tl;
            left[1..17].fill(129);
            left[33..41].fill(129);
            left[49..57].fill(129);
            self.it.left_nz[8] = 0;
            if self.top_derr.is_some() {
                self.it.left_derr = [[0; 2]; 2];
            }
        } else {
            if y == 0 {
                left[0] = 127;
                left[32] = 127;
                left[48] = 127;
            } else {
                left[0] = pic.y[ysrc - 1 - pic.y_stride];
                left[32] = pic.u[uvsrc - 1 - pic.uv_stride];
                left[48] = pic.v[uvsrc - 1 - pic.uv_stride];
            }
            import_line(&pic.y[ysrc - 1..], pic.y_stride, &mut left[1..17], h);
            import_line(&pic.u[uvsrc - 1..], pic.uv_stride, &mut left[33..41], uv_h);
            import_line(&pic.v[uvsrc - 1..], pic.uv_stride, &mut left[49..57], uv_h);
        }
        let mut top = [127u8; 32];
        if y > 0 {
            import_line(&pic.y[ysrc - pic.y_stride..], 1, &mut top[..16], w);
            import_line(&pic.u[uvsrc - pic.uv_stride..], 1, &mut top[16..24], uv_w);
            import_line(&pic.v[uvsrc - pic.uv_stride..], 1, &mut top[24..32], uv_w);
        }
        self.it.analysis_top = Some(top);
    }

    fn top_y(&self) -> &[u8] {
        match &self.it.analysis_top {
            Some(top) => &top[..16],
            None => &self.y_top[self.it.x * 16..],
        }
    }

    fn top_uv(&self) -> &[u8] {
        match &self.it.analysis_top {
            Some(top) => &top[16..32],
            None => &self.uv_top[self.it.x * 16..self.it.x * 16 + 16],
        }
    }

    fn nz_to_bytes(&mut self) {
        let x = self.it.x;
        let tnz = self.nz[1 + x];
        let lnz = self.nz[x];
        let bit = |nz: u32, n: u32| ((nz >> n) & 1) as i32;
        let t = &mut self.it.top_nz;
        t[0] = bit(tnz, 12);
        t[1] = bit(tnz, 13);
        t[2] = bit(tnz, 14);
        t[3] = bit(tnz, 15);
        t[4] = bit(tnz, 18);
        t[5] = bit(tnz, 19);
        t[6] = bit(tnz, 22);
        t[7] = bit(tnz, 23);
        t[8] = bit(tnz, 24);
        let l = &mut self.it.left_nz;
        l[0] = bit(lnz, 3);
        l[1] = bit(lnz, 7);
        l[2] = bit(lnz, 11);
        l[3] = bit(lnz, 15);
        l[4] = bit(lnz, 17);
        l[5] = bit(lnz, 19);
        l[6] = bit(lnz, 21);
        l[7] = bit(lnz, 23);
    }

    fn bytes_to_nz(&mut self) {
        let t = &self.it.top_nz;
        let l = &self.it.left_nz;
        let mut nz = 0u32;
        nz |= ((t[0] << 12) | (t[1] << 13)) as u32;
        nz |= ((t[2] << 14) | (t[3] << 15)) as u32;
        nz |= ((t[4] << 18) | (t[5] << 19)) as u32;
        nz |= ((t[6] << 22) | (t[7] << 23)) as u32;
        nz |= (t[8] << 24) as u32;
        nz |= ((l[0] << 3) | (l[1] << 7)) as u32;
        nz |= (l[2] << 11) as u32;
        nz |= ((l[4] << 17) | (l[6] << 21)) as u32;
        self.nz[1 + self.it.x] = nz;
    }

    fn save_boundary(&mut self) {
        let (x, y) = (self.it.x, self.it.y);
        let out = &self.it.yuv_out;
        if x < self.mb_w - 1 {
            let left = &mut self.it.left;
            for i in 0..16 {
                left[1 + i] = out[Y_OFF + 15 + i * BPS];
            }
            for i in 0..8 {
                left[33 + i] = out[U_OFF + 7 + i * BPS];
                left[49 + i] = out[U_OFF + 15 + i * BPS];
            }
            left[0] = self.y_top[x * 16 + 15];
            left[32] = self.uv_top[x * 16 + 7];
            left[48] = self.uv_top[x * 16 + 8 + 7];
        }
        if y < self.mb_h - 1 {
            self.y_top[x * 16..x * 16 + 16]
                .copy_from_slice(&out[Y_OFF + 15 * BPS..Y_OFF + 15 * BPS + 16]);
            self.uv_top[x * 16..x * 16 + 16]
                .copy_from_slice(&out[U_OFF + 7 * BPS..U_OFF + 7 * BPS + 16]);
        }
    }

    fn set_intra16_mode(&mut self, mode: u8) {
        let p = self.preds_index();
        for y in 0..4 {
            let row = p + y * self.preds_w;
            self.preds[row..row + 4].fill(mode);
        }
        self.mb_mut().mb_type = 1;
    }

    fn set_intra4_mode(&mut self, modes: &[u8; 16]) {
        let p = self.preds_index();
        for y in 0..4 {
            let row = p + y * self.preds_w;
            self.preds[row..row + 4].copy_from_slice(&modes[y * 4..y * 4 + 4]);
        }
        self.mb_mut().mb_type = 0;
    }

    fn start_i4(&mut self) {
        let x = self.it.x;
        self.it.i4 = 0;
        self.it.i4_top = TOP_LEFT_I4[0];
        for i in 0..17 {
            self.it.i4_boundary[i] = self.it.left[16 - i];
        }
        let top = &self.y_top[x * 16..];
        self.it.i4_boundary[17..33].copy_from_slice(&top[..16]);
        if x < self.mb_w - 1 {
            self.it.i4_boundary[33..37].copy_from_slice(&top[16..20]);
        } else {
            let v = self.it.i4_boundary[17 + 15];
            self.it.i4_boundary[33..37].fill(v);
        }
        self.nz_to_bytes();
    }

    /// Updates the 4x4 boundary from the block just reconstructed in `yuv_out2` (or
    /// `yuv_out` when `in_out`). Returns false after the last block.
    fn rotate_i4(&mut self, in_out: bool) -> bool {
        let it = &mut self.it;
        let buf = if in_out { &it.yuv_out } else { &it.yuv_out2 };
        let blk = Y_OFF + SCAN[it.i4];
        let top = it.i4_top;
        for i in 0..4 {
            it.i4_boundary[top - 4 + i] = buf[blk + i + 3 * BPS];
        }
        if it.i4 & 3 != 3 {
            for i in 0..3 {
                it.i4_boundary[top + i] = buf[blk + 3 + (2 - i) * BPS];
            }
        } else {
            for i in 0..4 {
                it.i4_boundary[top + i] = it.i4_boundary[top + i + 4];
            }
        }
        it.i4 += 1;
        if it.i4 == 16 {
            return false;
        }
        it.i4_top = TOP_LEFT_I4[it.i4];
        true
    }

    // ---------------------------------------------------------------------------------------
    // Analysis

    fn make_luma16_preds(&mut self) {
        let left = (self.it.x > 0).then(|| self.it.left);
        let top = (self.it.y > 0).then(|| {
            let mut t = [0u8; 16];
            t.copy_from_slice(&self.top_y()[..16]);
            t
        });
        pred_luma16(
            &mut self.it.yuv_p[..],
            left.as_ref().map(|l| &l[..17]),
            top.as_ref().map(|t| &t[..]),
        );
    }

    fn make_chroma8_preds(&mut self) {
        let left = (self.it.x > 0).then(|| self.it.left);
        let top = (self.it.y > 0).then(|| {
            let mut t = [0u8; 16];
            t.copy_from_slice(self.top_uv());
            t
        });
        pred_chroma8(
            &mut self.it.yuv_p[..],
            left.as_ref().map(|l| &l[32..]),
            top.as_ref().map(|t| &t[..]),
        );
    }

    fn mb_analyze(&mut self, alphas: &mut [i32; 256], alpha: &mut i32, uv_alpha: &mut i32) {
        self.set_intra16_mode(0);
        self.mb_mut().skip = false;
        self.mb_mut().segment = 0;

        let best_alpha = if self.method <= 1 {
            self.fast_mb_analyze()
        } else {
            self.mb_analyze_best_intra16_mode()
        };
        let best_uv_alpha = self.mb_analyze_best_uv_mode();

        let best_alpha = (3 * best_alpha + best_uv_alpha + 2) >> 2;
        let best_alpha = (255 - best_alpha).clamp(0, 255);
        alphas[best_alpha as usize] += 1;
        self.mb_mut().alpha = best_alpha as u8;
        *alpha += best_alpha;
        *uv_alpha += best_uv_alpha;
    }

    fn mb_analyze_best_intra16_mode(&mut self) -> i32 {
        let mut best_alpha = -1;
        let mut best_mode = 0;
        self.make_luma16_preds();
        for mode in 0..2 {
            let histo = collect_histogram(
                &self.it.yuv_in[Y_OFF..],
                &self.it.yuv_p[I16_MODE_OFFSETS[mode]..],
                0,
                16,
            );
            let alpha = get_alpha(histo);
            if alpha > best_alpha {
                best_alpha = alpha;
                best_mode = mode;
            }
        }
        self.set_intra16_mode(best_mode as u8);
        best_alpha
    }

    fn fast_mb_analyze(&mut self) -> i32 {
        let q = self.config.quality as i32;
        let threshold = (8 + (17 - 8) * q / 100) as u32;
        let mut dc = [0u32; 16];
        for k in (0..16).step_by(4) {
            mean16x4(&self.it.yuv_in[Y_OFF + k * BPS..], &mut dc[k..k + 4]);
        }
        let (mut m, mut m2) = (0u32, 0u32);
        for &v in &dc {
            m = m.wrapping_add(v);
            m2 = m2.wrapping_add(v.wrapping_mul(v));
        }
        if threshold.wrapping_mul(m2) < m.wrapping_mul(m) {
            self.set_intra16_mode(0);
        } else {
            self.set_intra4_mode(&[0; 16]);
        }
        0
    }

    fn mb_analyze_best_uv_mode(&mut self) -> i32 {
        let mut best_alpha = -1;
        let mut smallest_alpha = 0;
        let mut best_mode = 0;
        self.make_chroma8_preds();
        for mode in 0..2 {
            let histo = collect_histogram(
                &self.it.yuv_in[U_OFF..],
                &self.it.yuv_p[UV_MODE_OFFSETS[mode]..],
                16,
                24,
            );
            let alpha = get_alpha(histo);
            if alpha > best_alpha {
                best_alpha = alpha;
            }
            if mode == 0 || alpha < smallest_alpha {
                smallest_alpha = alpha;
                best_mode = mode;
            }
        }
        self.mb_mut().uv_mode = best_mode as u8;
        best_alpha
    }

    fn analyze(&mut self) {
        let do_segments = self.num_segments > 1 || self.method <= 1;
        if do_segments {
            let mut alphas = [0i32; 256];
            let (mut alpha, mut uv_alpha) = (0, 0);
            self.iterator_reset();
            loop {
                self.iterator_import(true);
                self.mb_analyze(&mut alphas, &mut alpha, &mut uv_alpha);
                if !self.iterator_next() {
                    break;
                }
            }
            self.it.analysis_top = None;
            let total_mb = (self.mb_w * self.mb_h) as i32;
            self.alpha = alpha / total_mb;
            self.uv_alpha = uv_alpha / total_mb;
            self.assign_segments(&alphas);
        } else {
            for mb in &mut self.mb_info {
                *mb = MbInfo {
                    mb_type: 1,
                    ..MbInfo::default()
                };
            }
            self.dqm[0].alpha = 0;
            self.dqm[0].beta = 0;
            self.alpha = 0;
            self.uv_alpha = 0;
        }
    }

    fn assign_segments(&mut self, alphas: &[i32; 256]) {
        const MAX_ITERS_K_MEANS: usize = 6;
        let nb = self.num_segments.min(NUM_MB_SEGMENTS);
        let mut centers = [0i32; NUM_MB_SEGMENTS];
        let mut weighted_average = 0;
        let mut map = [0usize; 256];

        let mut n = 0;
        while n <= 255 && alphas[n] == 0 {
            n += 1;
        }
        let min_a = n as i32;
        let mut n = 255;
        while n > min_a as usize && alphas[n] == 0 {
            n -= 1;
        }
        let max_a = n as i32;
        let range_a = max_a - min_a;

        for k in 0..nb {
            let n = (2 * k + 1) as i32;
            centers[k] = min_a + (n * range_a) / (2 * nb as i32);
        }

        for _ in 0..MAX_ITERS_K_MEANS {
            let mut accum = [0i32; NUM_MB_SEGMENTS];
            let mut dist_accum = [0i32; NUM_MB_SEGMENTS];
            let mut n = 0;
            for a in min_a..=max_a {
                if alphas[a as usize] != 0 {
                    while n + 1 < nb && (a - centers[n + 1]).abs() < (a - centers[n]).abs() {
                        n += 1;
                    }
                    map[a as usize] = n;
                    dist_accum[n] += a * alphas[a as usize];
                    accum[n] += alphas[a as usize];
                }
            }
            let mut displaced = 0;
            weighted_average = 0;
            let mut total_weight = 0;
            for n in 0..nb {
                if accum[n] != 0 {
                    let new_center = (dist_accum[n] + accum[n] / 2) / accum[n];
                    displaced += (centers[n] - new_center).abs();
                    centers[n] = new_center;
                    weighted_average += new_center * accum[n];
                    total_weight += accum[n];
                }
            }
            weighted_average = (weighted_average + total_weight / 2) / total_weight;
            if displaced < 5 {
                break;
            }
        }

        for mb in &mut self.mb_info {
            let alpha = mb.alpha as usize;
            mb.segment = map[alpha] as u8;
            mb.alpha = centers[map[alpha]] as u8;
        }

        if nb > 1 && self.config.preprocessing & 1 != 0 {
            self.smooth_segment_map();
        }

        self.set_segment_alphas(&centers, weighted_average);
    }

    fn smooth_segment_map(&mut self) {
        let (w, h) = (self.mb_w, self.mb_h);
        let mut tmp = vec![0u8; w * h];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w - 1 {
                let i = x + w * y;
                let mut cnt = [0i32; NUM_MB_SEGMENTS];
                let mut majority_seg = self.mb_info[i].segment;
                for j in [
                    i - w - 1,
                    i - w,
                    i - w + 1,
                    i - 1,
                    i + 1,
                    i + w - 1,
                    i + w,
                    i + w + 1,
                ] {
                    cnt[self.mb_info[j].segment as usize] += 1;
                }
                for (n, &c) in cnt.iter().enumerate() {
                    if c >= 5 {
                        majority_seg = n as u8;
                        break;
                    }
                }
                tmp[i] = majority_seg;
            }
        }
        for y in 1..h.saturating_sub(1) {
            for x in 1..w - 1 {
                self.mb_info[x + w * y].segment = tmp[x + w * y];
            }
        }
    }

    fn set_segment_alphas(&mut self, centers: &[i32; NUM_MB_SEGMENTS], mid: i32) {
        let nb = self.num_segments;
        let mut min = centers[0];
        let mut max = centers[0];
        if nb > 1 {
            for &c in &centers[..nb] {
                min = min.min(c);
                max = max.max(c);
            }
        }
        if max == min {
            max = min + 1;
        }
        for n in 0..nb {
            let alpha = 255 * (centers[n] - mid) / (max - min);
            let beta = 255 * (centers[n] - min) / (max - min);
            self.dqm[n].alpha = alpha.clamp(-127, 127);
            self.dqm[n].beta = beta.clamp(0, 255);
        }
    }

    // ---------------------------------------------------------------------------------------
    // Segment parameters

    fn set_segment_params(&mut self, quality: f32) {
        const SNS_TO_DQ: f64 = 0.9;
        const MID_ALPHA: i32 = 64;
        const MIN_ALPHA: i32 = 30;
        const MAX_ALPHA: i32 = 100;
        const MAX_DQ_UV: i32 = 6;
        const MIN_DQ_UV: i32 = -4;

        let num_segments = self.num_segments;
        let sns = self.config.sns_strength;
        let amp = SNS_TO_DQ * f64::from(sns) / 100. / 128.;
        let q = f64::from(quality) / 100.;
        let c_base = quality_to_compression(q);
        for i in 0..num_segments {
            let expn = 1. - amp * f64::from(self.dqm[i].alpha);
            let c = c_base.powf(expn);
            let q = (127. * (1. - c)) as i32;
            self.dqm[i].quant = q.clamp(0, 127);
        }
        self.base_quant = self.dqm[0].quant;
        for i in num_segments..NUM_MB_SEGMENTS {
            self.dqm[i].quant = self.base_quant;
        }

        let dq_uv_ac =
            (self.uv_alpha - MID_ALPHA) * (MAX_DQ_UV - MIN_DQ_UV) / (MAX_ALPHA - MIN_ALPHA);
        let dq_uv_ac = (dq_uv_ac * sns / 100).clamp(MIN_DQ_UV, MAX_DQ_UV);
        let dq_uv_dc = (-4 * sns / 100).clamp(-15, 15);

        self.dq_y1_dc = 0;
        self.dq_y2_dc = 0;
        self.dq_y2_ac = 0;
        self.dq_uv_dc = dq_uv_dc;
        self.dq_uv_ac = dq_uv_ac;

        self.setup_filter_strength();
        if num_segments > 1 {
            self.simplify_segments();
        }
        self.setup_matrices();
    }

    fn setup_filter_strength(&mut self) {
        const FSTRENGTH_CUTOFF: i32 = 2;
        let level0 = 5 * self.config.filter_strength;
        for m in &mut self.dqm {
            let qstep = i32::from(AC_TABLE[m.quant.clamp(0, 127) as usize]) >> 2;
            let base_strength = filter_strength_from_delta(self.filter_sharpness, qstep);
            let f = base_strength * level0 / (256 + m.beta);
            m.fstrength = if f < FSTRENGTH_CUTOFF { 0 } else { f.min(63) };
        }
        self.filter_level = self.dqm[0].fstrength;
        self.filter_simple = self.config.filter_type == 0;
        self.filter_sharpness = self.config.filter_sharpness;
    }

    fn simplify_segments(&mut self) {
        let mut map = [0u8, 1, 2, 3];
        let num_segments = self.num_segments.min(NUM_MB_SEGMENTS);
        let mut num_final_segments = 1;
        for s1 in 1..num_segments {
            let mut found = false;
            let mut s2 = 0;
            while s2 < num_final_segments {
                if self.dqm[s1].quant == self.dqm[s2].quant
                    && self.dqm[s1].fstrength == self.dqm[s2].fstrength
                {
                    found = true;
                    break;
                }
                s2 += 1;
            }
            map[s1] = s2 as u8;
            if !found {
                if num_final_segments != s1 {
                    self.dqm[num_final_segments] = self.dqm[s1];
                }
                num_final_segments += 1;
            }
        }
        if num_final_segments < num_segments {
            for mb in &mut self.mb_info {
                mb.segment = map[mb.segment as usize];
            }
            self.num_segments = num_final_segments;
            for i in num_final_segments..num_segments {
                self.dqm[i] = self.dqm[num_final_segments - 1];
            }
        }
    }

    fn setup_matrices(&mut self) {
        let tlambda_scale = if self.method >= 4 {
            self.config.sns_strength
        } else {
            0
        };
        for i in 0..self.num_segments {
            let (dq_y1_dc, dq_y2_dc, dq_y2_ac, dq_uv_dc, dq_uv_ac) = (
                self.dq_y1_dc,
                self.dq_y2_dc,
                self.dq_y2_ac,
                self.dq_uv_dc,
                self.dq_uv_ac,
            );
            let m = &mut self.dqm[i];
            let q = m.quant;
            m.y1.q[0] = u16::from(DC_TABLE[(q + dq_y1_dc).clamp(0, 127) as usize]);
            m.y1.q[1] = AC_TABLE[q.clamp(0, 127) as usize];
            m.y2.q[0] = u16::from(DC_TABLE[(q + dq_y2_dc).clamp(0, 127) as usize]) * 2;
            m.y2.q[1] = AC_TABLE2[(q + dq_y2_ac).clamp(0, 127) as usize];
            m.uv.q[0] = u16::from(DC_TABLE[(q + dq_uv_dc).clamp(0, 117) as usize]);
            m.uv.q[1] = AC_TABLE[(q + dq_uv_ac).clamp(0, 127) as usize];

            let q_i4 = expand_matrix(&mut m.y1, 0);
            let q_i16 = expand_matrix(&mut m.y2, 1);
            let q_uv = expand_matrix(&mut m.uv, 2);

            m.lambda_i4 = ((3 * q_i4 * q_i4) >> 7).max(1);
            m.lambda_i16 = (3 * q_i16 * q_i16).max(1);
            m.lambda_uv = ((3 * q_uv * q_uv) >> 6).max(1);
            m.lambda_mode = ((q_i4 * q_i4) >> 7).max(1);
            m.lambda_trellis_i4 = ((7 * q_i4 * q_i4) >> 3).max(1);
            m.lambda_trellis_i16 = ((q_i16 * q_i16) >> 2).max(1);
            m.lambda_trellis_uv = ((q_uv * q_uv) << 1).max(1);
            m.tlambda = ((tlambda_scale * q_i4) >> 5).max(1);

            m.min_disto = 20 * i64::from(m.y1.q[0]);
            m.max_edge = 0;
            m.i4_penalty = 1000 * i64::from(q_i4) * i64::from(q_i4);
        }
    }

    fn set_segment_probas(&mut self) {
        let mut p = [0i32; NUM_MB_SEGMENTS];
        for mb in &self.mb_info {
            p[mb.segment as usize] += 1;
        }
        if self.num_segments > 1 {
            let probas = &mut self.proba.segments;
            probas[0] = get_proba(p[0] + p[1], p[2] + p[3]);
            probas[1] = get_proba(p[0], p[1]);
            probas[2] = get_proba(p[2], p[3]);
            self.update_map = probas[0] != 255 || probas[1] != 255 || probas[2] != 255;
            if !self.update_map {
                for mb in &mut self.mb_info {
                    mb.segment = 0;
                }
            }
            let probas = self.proba.segments;
            let size = p[0] * (bit_cost(false, probas[0]) + bit_cost(false, probas[1]))
                + p[1] * (bit_cost(false, probas[0]) + bit_cost(true, probas[1]))
                + p[2] * (bit_cost(true, probas[0]) + bit_cost(false, probas[2]))
                + p[3] * (bit_cost(true, probas[0]) + bit_cost(true, probas[2]));
            self.segment_hdr_size = i64::from(size);
        } else {
            self.update_map = false;
            self.segment_hdr_size = 0;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Reconstruction

    fn reconstruct_intra16(
        &mut self,
        dc_levels: &mut [i16; 16],
        ac_levels: &mut [[i16; 16]; 16],
        to_out2: bool,
        mode: usize,
    ) -> u32 {
        let seg = self.mb().segment as usize;
        let dqm = &self.dqm[seg];
        let r = &self.it.yuv_p[I16_MODE_OFFSETS[mode]..];
        let src = &self.it.yuv_in[Y_OFF..];
        let mut nz = 0u32;
        let mut tmp = [[0i16; 16]; 16];
        let mut dc_tmp = [0i16; 16];

        for n in (0..16).step_by(2) {
            ftransform2(
                &src[SCAN[n]..],
                &r[SCAN[n]..],
                tmp[n..n + 2].as_flattened_mut(),
            );
        }
        ftransform_wht(tmp.as_flattened(), &mut dc_tmp);
        nz |= u32::from(quantize_block(&mut dc_tmp, dc_levels, &dqm.y2)) << 24;

        if self.it.do_trellis {
            let y1 = dqm.y1;
            let lambda = dqm.lambda_trellis_i16;
            self.nz_to_bytes();
            for y in 0..4 {
                for x in 0..4 {
                    let n = y * 4 + x;
                    let ctx = (self.it.top_nz[x] + self.it.left_nz[y]) as usize;
                    let non_zero = trellis_quantize_block(
                        &self.proba,
                        &mut tmp[n],
                        &mut ac_levels[n],
                        ctx,
                        0,
                        &y1,
                        lambda,
                    );
                    self.it.top_nz[x] = i32::from(non_zero);
                    self.it.left_nz[y] = i32::from(non_zero);
                    ac_levels[n][0] = 0;
                    nz |= u32::from(non_zero) << n;
                }
            }
        } else {
            for n in (0..16).step_by(2) {
                tmp[n][0] = 0;
                tmp[n + 1][0] = 0;
                nz |= quantize2_blocks(
                    tmp[n..n + 2].as_flattened_mut(),
                    ac_levels[n..n + 2].as_flattened_mut(),
                    &dqm.y1,
                ) << n;
            }
        }

        transform_wht(&dc_tmp, tmp.as_flattened_mut());
        let it = &mut self.it;
        let r = &it.yuv_p[I16_MODE_OFFSETS[mode]..];
        let dst = if to_out2 {
            &mut it.yuv_out2[..]
        } else {
            &mut it.yuv_out[..]
        };
        for n in (0..16).step_by(2) {
            itransform(
                &r[SCAN[n]..],
                tmp[n..n + 2].as_flattened(),
                &mut dst[Y_OFF + SCAN[n]..],
                true,
            );
        }
        nz
    }

    /// Reconstructs 4x4 block `i4` with `mode` into `yuv_out` or `yuv_out2`.
    fn reconstruct_intra4(
        &mut self,
        levels: &mut [i16; 16],
        mode: usize,
        target: I4Target,
    ) -> bool {
        let (coeffs, nz) = self.quantize_intra4(levels, mode);
        let src_off = Y_OFF + SCAN[self.it.i4];
        let it = &mut self.it;
        let r = &it.yuv_p[I4_MODE_OFFSETS[mode]..];
        let dst = match target {
            I4Target::Out => &mut it.yuv_out[src_off..],
            I4Target::Out2 => &mut it.yuv_out2[src_off..],
        };
        itransform(r, &coeffs, dst, false);
        nz
    }

    /// The levels of 4x4 block `i4` predicted with `mode`, returning the dequantized
    /// coefficients and whether a level is non-zero.
    fn quantize_intra4(&self, levels: &mut [i16; 16], mode: usize) -> ([i16; 16], bool) {
        let dqm = &self.dqm[self.mb().segment as usize];
        let src_off = Y_OFF + SCAN[self.it.i4];
        let mut tmp = [0i16; 16];
        ftransform(
            &self.it.yuv_in[src_off..],
            &self.it.yuv_p[I4_MODE_OFFSETS[mode]..],
            &mut tmp,
        );
        let nz = if self.it.do_trellis {
            let x = self.it.i4 & 3;
            let y = self.it.i4 >> 2;
            let ctx = (self.it.top_nz[x] + self.it.left_nz[y]) as usize;
            trellis_quantize_block(
                &self.proba,
                &mut tmp,
                levels,
                ctx,
                3,
                &dqm.y1,
                dqm.lambda_trellis_i4,
            )
        } else {
            quantize_block(&mut tmp, levels, &dqm.y1)
        };
        (tmp, nz)
    }

    fn quantize_single(v: &mut i16, mtx: &Matrix) -> i32 {
        const DSCALE: i32 = 1;
        let mut value = i32::from(*v);
        let sign = value < 0;
        if sign {
            value = -value;
        }
        if value > mtx.zthresh[0] as i32 {
            let qv =
                quantdiv(value as u32, u32::from(mtx.iq[0]), mtx.bias[0]) * i32::from(mtx.q[0]);
            let err = value - qv;
            *v = if sign { -qv } else { qv } as i16;
            return (if sign { -err } else { err }) >> DSCALE;
        }
        *v = 0;
        (if sign { -value } else { value }) >> DSCALE
    }

    fn correct_dc_values(&self, mtx: &Matrix, tmp: &mut [[i16; 16]; 8], derr: &mut [[i8; 3]; 2]) {
        const C1: i32 = 7;
        const C2: i32 = 8;
        const DSHIFT: i32 = 4;
        const DSCALE: i32 = 1;
        let top_derr = self.top_derr.as_ref().unwrap();
        for ch in 0..2 {
            let top = top_derr[self.it.x][ch];
            let left = self.it.left_derr[ch];
            let c = &mut tmp[ch * 4..ch * 4 + 4];
            c[0][0] = c[0][0].wrapping_add(
                ((C1 * i32::from(top[0]) + C2 * i32::from(left[0])) >> (DSHIFT - DSCALE)) as i16,
            );
            let err0 = Self::quantize_single(&mut c[0][0], mtx);
            c[1][0] = c[1][0]
                .wrapping_add(((C1 * i32::from(top[1]) + C2 * err0) >> (DSHIFT - DSCALE)) as i16);
            let err1 = Self::quantize_single(&mut c[1][0], mtx);
            c[2][0] = c[2][0]
                .wrapping_add(((C1 * err0 + C2 * i32::from(left[1])) >> (DSHIFT - DSCALE)) as i16);
            let err2 = Self::quantize_single(&mut c[2][0], mtx);
            c[3][0] = c[3][0].wrapping_add(((C1 * err1 + C2 * err2) >> (DSHIFT - DSCALE)) as i16);
            let err3 = Self::quantize_single(&mut c[3][0], mtx);
            derr[ch][0] = err1 as i8;
            derr[ch][1] = err2 as i8;
            derr[ch][2] = err3 as i8;
        }
    }

    fn store_diffusion_errors(&mut self, rd: &ModeScore) {
        let x = self.it.x;
        let top_derr = self.top_derr.as_mut().unwrap();
        for ch in 0..2 {
            let left = &mut self.it.left_derr[ch];
            let top = &mut top_derr[x][ch];
            left[0] = rd.derr[ch][0];
            left[1] = ((3 * i32::from(rd.derr[ch][2])) >> 2) as i8;
            top[0] = rd.derr[ch][1];
            top[1] = rd.derr[ch][2].wrapping_sub(left[1]);
        }
    }

    fn reconstruct_uv(
        &mut self,
        uv_levels: &mut [[i16; 16]; 8],
        derr: &mut [[i8; 3]; 2],
        to_out2: bool,
        mode: usize,
    ) -> u32 {
        let seg = self.mb().segment as usize;
        let mut nz = 0u32;
        let mut tmp = [[0i16; 16]; 8];
        {
            let r = &self.it.yuv_p[UV_MODE_OFFSETS[mode]..];
            let src = &self.it.yuv_in[U_OFF..];
            for n in (0..8).step_by(2) {
                ftransform2(
                    &src[SCAN_UV[n]..],
                    &r[SCAN_UV[n]..],
                    tmp[n..n + 2].as_flattened_mut(),
                );
            }
        }
        let dqm = &self.dqm[seg];
        let uv = &dqm.uv;
        if self.top_derr.is_some() {
            self.correct_dc_values(uv, &mut tmp, derr);
        }
        const DO_TRELLIS_UV: bool = false;
        if DO_TRELLIS_UV && self.it.do_trellis {
            let mut n = 0;
            for ch in [0, 2] {
                for y in 0..2 {
                    for x in 0..2 {
                        let ctx =
                            (self.it.top_nz[4 + ch + x] + self.it.left_nz[4 + ch + y]) as usize;
                        let non_zero = trellis_quantize_block(
                            &self.proba,
                            &mut tmp[n],
                            &mut uv_levels[n],
                            ctx,
                            2,
                            uv,
                            dqm.lambda_trellis_uv,
                        );
                        self.it.top_nz[4 + ch + x] = i32::from(non_zero);
                        self.it.left_nz[4 + ch + y] = i32::from(non_zero);
                        nz |= u32::from(non_zero) << n;
                        n += 1;
                    }
                }
            }
        } else {
            for n in (0..8).step_by(2) {
                nz |= quantize2_blocks(
                    tmp[n..n + 2].as_flattened_mut(),
                    uv_levels[n..n + 2].as_flattened_mut(),
                    uv,
                ) << n;
            }
        }
        let it = &mut self.it;
        let r = &it.yuv_p[UV_MODE_OFFSETS[mode]..];
        let dst = if to_out2 {
            &mut it.yuv_out2[..]
        } else {
            &mut it.yuv_out[..]
        };
        for n in (0..8).step_by(2) {
            itransform(
                &r[SCAN_UV[n]..],
                tmp[n..n + 2].as_flattened(),
                &mut dst[U_OFF + SCAN_UV[n]..],
                true,
            );
        }
        nz << 16
    }

    // ---------------------------------------------------------------------------------------
    // Rate

    fn cost_luma16(&mut self, dc_levels: &[i16; 16], ac_levels: &[[i16; 16]; 16]) -> i64 {
        self.nz_to_bytes();
        let mut r = Residual::new(0, 1, dc_levels).cost(
            (self.it.top_nz[8] + self.it.left_nz[8]) as usize,
            &self.proba,
        );
        for y in 0..4 {
            for x in 0..4 {
                let ctx = (self.it.top_nz[x] + self.it.left_nz[y]) as usize;
                let res = Residual::new(1, 0, &ac_levels[x + y * 4]);
                r += res.cost(ctx, &self.proba);
                let nz = i32::from(res.last >= 0);
                self.it.top_nz[x] = nz;
                self.it.left_nz[y] = nz;
            }
        }
        i64::from(r)
    }

    /// The rate of `levels`, whose last non-zero one is `last`.
    fn cost_luma4(&self, levels: &[i16; 16], last: i32) -> i64 {
        let x = self.it.i4 & 3;
        let y = self.it.i4 >> 2;
        let ctx = (self.it.top_nz[x] + self.it.left_nz[y]) as usize;
        let res = Residual {
            first: 0,
            last,
            coeffs: levels,
            coeff_type: 3,
        };
        i64::from(res.cost(ctx, &self.proba))
    }

    fn cost_uv(&mut self, uv_levels: &[[i16; 16]; 8]) -> i64 {
        self.nz_to_bytes();
        let mut r = 0;
        for ch in [0, 2] {
            for y in 0..2 {
                for x in 0..2 {
                    let ctx = (self.it.top_nz[4 + ch + x] + self.it.left_nz[4 + ch + y]) as usize;
                    let res = Residual::new(0, 2, &uv_levels[ch * 2 + x + y * 2]);
                    r += res.cost(ctx, &self.proba);
                    let nz = i32::from(res.last >= 0);
                    self.it.top_nz[4 + ch + x] = nz;
                    self.it.left_nz[4 + ch + y] = nz;
                }
            }
        }
        i64::from(r)
    }

    // ---------------------------------------------------------------------------------------
    // Mode decision

    fn pick_best_intra16(&mut self, rd: &mut ModeScore) {
        let seg = self.mb().segment as usize;
        let lambda = self.dqm[seg].lambda_i16;
        let tlambda = self.dqm[seg].tlambda;
        let mut is_flat = is_flat_source16(&self.it.yuv_in[Y_OFF..]);
        // Candidate levels in two slots: the best so far and the one being tried.
        let mut levels = [([0i16; 16], [[0i16; 16]; 16]); 2];
        let mut best = 0;
        let mut best_score = Score::INIT;
        let mut best_mode = -1;
        for mode in 0..NUM_PRED_MODES {
            let slot = if best_mode < 0 { 0 } else { 1 - best };
            let (dc, ac) = &mut levels[slot];
            let mut cur = Score::INIT;
            cur.nz = self.reconstruct_intra16(dc, ac, true, mode);
            let src = &self.it.yuv_in[Y_OFF..];
            let out = &self.it.yuv_out2[Y_OFF..];
            cur.d = i64::from(sse16x16(src, out));
            cur.sd = if tlambda != 0 {
                i64::from(mult_8b(tlambda, tdisto16x16(src, out, &WEIGHT_Y)))
            } else {
                0
            };
            cur.h = i64::from(FIXED_COSTS_I16[mode]);
            let (dc, ac) = &levels[slot];
            cur.r = self.cost_luma16(dc, ac);
            if is_flat {
                is_flat = is_flat_fn(ac.as_flattened(), 16, FLATNESS_LIMIT_I16);
                if is_flat {
                    cur.d *= 2;
                    cur.sd *= 2;
                }
            }
            cur.set_rd_score(lambda);
            if mode == 0 || cur.score < best_score.score {
                best = slot;
                best_score = cur;
                best_mode = mode as i32;
                self.it.swap_out();
            }
        }
        (rd.y_dc_levels, rd.y_ac_levels) = levels[best];
        rd.d = best_score.d;
        rd.sd = best_score.sd;
        rd.h = best_score.h;
        rd.r = best_score.r;
        rd.nz = best_score.nz;
        rd.mode_i16 = best_mode;
        rd.set_rd_score(self.dqm[seg].lambda_mode);
        self.set_intra16_mode(rd.mode_i16 as u8);

        if (rd.nz & 0x100_ffff) == 0x100_0000 && rd.d > self.dqm[seg].min_disto {
            let dqm = &mut self.dqm[seg];
            let v0 = i32::from(rd.y_dc_levels[1]).abs();
            let v1 = i32::from(rd.y_dc_levels[2]).abs();
            let v2 = i32::from(rd.y_dc_levels[4]).abs();
            let max_v = v0.max(v1).max(v2);
            if max_v > dqm.max_edge {
                dqm.max_edge = max_v;
            }
        }
    }

    fn cost_mode_i4(&self, modes: &[u8; 16]) -> &'static [u16; 10] {
        let x = self.it.i4 & 3;
        let y = self.it.i4 >> 2;
        let p = self.preds_index();
        let left = if x == 0 {
            self.preds[p + y * self.preds_w - 1]
        } else {
            modes[self.it.i4 - 1]
        };
        let top = if y == 0 {
            self.preds[p - self.preds_w + x]
        } else {
            modes[self.it.i4 - 4]
        };
        &FIXED_COSTS_I4[top as usize][left as usize]
    }

    fn make_intra4_preds(&mut self) {
        let it = &mut self.it;
        pred_luma4(&mut it.yuv_p[..], &it.i4_boundary, it.i4_top);
    }

    fn pick_best_intra4(&mut self, rd: &mut ModeScore) -> bool {
        let seg = self.mb().segment as usize;
        let lambda = self.dqm[seg].lambda_i4;
        let tlambda = self.dqm[seg].tlambda;
        let lambda_mode = self.dqm[seg].lambda_mode;
        let mut total_header_bits = 0;
        let mut rd_best = ModeScore::default();

        if self.max_i4_header_bits == 0 {
            return false;
        }

        rd_best.init_score();
        rd_best.h = 211;
        rd_best.set_rd_score(lambda_mode);
        self.start_i4();
        loop {
            let i4 = self.it.i4;
            let src_off = Y_OFF + SCAN[i4];
            let mut rd_i4 = Score::INIT;
            let mut best_mode: i32 = -1;
            let mode_costs = self.cost_mode_i4(&rd.modes_i4);
            // Whether the scratch block, rather than yuv_out2, holds the best candidate.
            let mut best_in_tmp = false;

            self.make_intra4_preds();
            for mode in 0..NUM_BMODES {
                let mut tmp_levels = [0i16; 16];
                let w = (tlambda != 0).then_some(&WEIGHT_Y);
                let (non_zero, d, disto) = if self.it.do_trellis {
                    let (coeffs, _) = self.quantize_intra4(&mut tmp_levels, mode);
                    let it = &mut self.it;
                    let dst = if best_in_tmp {
                        &mut it.yuv_out2[src_off..]
                    } else {
                        &mut it.i4_tmp[..]
                    };
                    let r = &it.yuv_p[I4_MODE_OFFSETS[mode]..];
                    let (d, disto) = itransform_disto(r, &coeffs, dst, &it.yuv_in[src_off..], w);
                    (non_zero_mask(&tmp_levels), d, disto)
                } else {
                    let it = &mut self.it;
                    let dst = if best_in_tmp {
                        &mut it.yuv_out2[src_off..]
                    } else {
                        &mut it.i4_tmp[..]
                    };
                    reconstruct4x4_disto(
                        &it.yuv_in[src_off..],
                        &it.yuv_p[I4_MODE_OFFSETS[mode]..],
                        &self.dqm[seg].y1,
                        &mut tmp_levels,
                        dst,
                        w,
                    )
                };
                let mut tmp = Score::INIT;
                tmp.nz = u32::from(non_zero != 0) << i4;
                tmp.d = i64::from(d);
                tmp.sd = i64::from(mult_8b(tlambda, disto));
                tmp.h = i64::from(mode_costs[mode]);
                tmp.r = if mode > 0 && (non_zero & !1).count_ones() as i32 <= FLATNESS_LIMIT_I4 {
                    FLATNESS_PENALTY
                } else {
                    0
                };

                tmp.set_rd_score(lambda);
                if best_mode >= 0 && tmp.score >= rd_i4.score {
                    continue;
                }
                tmp.r += self.cost_luma4(&tmp_levels, 31 - non_zero.leading_zeros() as i32);
                tmp.set_rd_score(lambda);

                if best_mode < 0 || tmp.score < rd_i4.score {
                    rd_i4 = tmp;
                    best_mode = mode as i32;
                    best_in_tmp = !best_in_tmp;
                    rd_best.y_ac_levels[i4] = tmp_levels;
                }
            }
            rd_i4.set_rd_score(lambda_mode);
            rd_best.add(&rd_i4);
            if rd_best.score >= rd.score {
                return false;
            }
            total_header_bits += rd_i4.h as i32;
            if total_header_bits > self.max_i4_header_bits {
                return false;
            }
            if best_in_tmp {
                let it = &mut self.it;
                copy(&it.i4_tmp, &mut it.yuv_out2[src_off..], 4, 4);
            }
            rd.modes_i4[i4] = best_mode as u8;
            let nz = i32::from(rd_i4.nz != 0);
            self.it.top_nz[i4 & 3] = nz;
            self.it.left_nz[i4 >> 2] = nz;
            if !self.rotate_i4(false) {
                break;
            }
        }

        rd.copy_score(&rd_best);
        let modes = rd.modes_i4;
        self.set_intra4_mode(&modes);
        self.it.swap_out();
        rd.y_ac_levels = rd_best.y_ac_levels;
        true
    }

    fn pick_best_uv(&mut self, rd: &mut ModeScore) {
        let seg = self.mb().segment as usize;
        let lambda = self.dqm[seg].lambda_uv;
        let mut rd_best = Score::INIT;
        // Whether the scratch target is yuv_out2 (and the kept result in yuv_out).
        let mut tmp_is_out2 = true;
        let mut levels = [([[0i16; 16]; 8], [[0i8; 3]; 2]); 2];
        let mut best = 0;

        rd.mode_uv = -1;
        for mode in 0..NUM_PRED_MODES {
            let slot = if rd.mode_uv < 0 { 0 } else { 1 - best };
            let (uv_levels, derr) = &mut levels[slot];
            let mut rd_uv = Score::INIT;
            rd_uv.nz = self.reconstruct_uv(uv_levels, derr, tmp_is_out2, mode);
            let src = &self.it.yuv_in[U_OFF..];
            let dst = if tmp_is_out2 {
                &self.it.yuv_out2[U_OFF..]
            } else {
                &self.it.yuv_out[U_OFF..]
            };
            rd_uv.d = i64::from(sse16x8(src, dst));
            rd_uv.sd = 0;
            rd_uv.h = i64::from(FIXED_COSTS_UV[mode]);
            let uv_levels = &levels[slot].0;
            rd_uv.r = self.cost_uv(uv_levels);
            if mode > 0 && is_flat_fn(uv_levels.as_flattened(), 8, FLATNESS_LIMIT_UV) {
                rd_uv.r += FLATNESS_PENALTY * 8;
            }
            rd_uv.set_rd_score(lambda);
            if mode == 0 || rd_uv.score < rd_best.score {
                rd_best = rd_uv;
                rd.mode_uv = mode as i32;
                best = slot;
                tmp_is_out2 = !tmp_is_out2;
            }
        }
        rd.uv_levels = levels[best].0;
        if self.top_derr.is_some() {
            rd.derr = levels[best].1;
        }
        self.mb_mut().uv_mode = rd.mode_uv as u8;
        rd.add(&rd_best);
        if !tmp_is_out2 {
            let it = &mut self.it;
            let (out, out2) = (&mut it.yuv_out, &it.yuv_out2);
            copy(&out2[U_OFF..], &mut out[U_OFF..], 16, 8);
        }
        if self.top_derr.is_some() {
            self.store_diffusion_errors(rd);
        }
    }

    fn simple_quantize(&mut self, rd: &mut ModeScore) {
        let is_i16 = self.mb().mb_type == 1;
        let mut nz = 0u32;
        if is_i16 {
            let mode = self.preds[self.preds_index()] as usize;
            nz = self.reconstruct_intra16(&mut rd.y_dc_levels, &mut rd.y_ac_levels, false, mode);
        } else {
            self.start_i4();
            loop {
                let i4 = self.it.i4;
                let mode =
                    self.preds[self.preds_index() + (i4 & 3) + (i4 >> 2) * self.preds_w] as usize;
                self.make_intra4_preds();
                let mut levels = rd.y_ac_levels[i4];
                nz |= u32::from(self.reconstruct_intra4(&mut levels, mode, I4Target::Out)) << i4;
                rd.y_ac_levels[i4] = levels;
                if !self.rotate_i4(true) {
                    break;
                }
            }
        }
        let uv_mode = self.mb().uv_mode as usize;
        nz |= self.reconstruct_uv(&mut rd.uv_levels, &mut rd.derr, false, uv_mode);
        rd.nz = nz;
    }

    fn refine_using_distortion(
        &mut self,
        mut try_both_modes: bool,
        refine_uv_mode: bool,
        rd: &mut ModeScore,
    ) {
        let mut best_score = MAX_COST;
        let mut nz = 0u32;
        let mut is_i16 = try_both_modes || self.mb().mb_type == 1;
        let seg = self.mb().segment as usize;
        let lambda_d_i16 = 106i64;
        let lambda_d_i4 = 11i64;
        let lambda_d_uv = 120i64;
        let mut score_i4 = self.dqm[seg].i4_penalty;
        let mut i4_bit_sum = 0i64;
        let bit_limit = if try_both_modes {
            self.mb_header_limit
        } else {
            MAX_COST
        };

        if is_i16 {
            let mut best_mode = -1i32;
            let src = &self.it.yuv_in[Y_OFF..];
            for mode in 0..NUM_PRED_MODES {
                let r = &self.it.yuv_p[I16_MODE_OFFSETS[mode]..];
                let score = i64::from(sse16x16(src, r)) * RD_DISTO_MULT
                    + i64::from(FIXED_COSTS_I16[mode]) * lambda_d_i16;
                if mode > 0 && i64::from(FIXED_COSTS_I16[mode]) > bit_limit {
                    continue;
                }
                if score < best_score {
                    best_mode = mode as i32;
                    best_score = score;
                }
            }
            if (self.it.x == 0 || self.it.y == 0) && is_flat_source16(src) {
                best_mode = if self.it.x == 0 { 0 } else { 2 };
                try_both_modes = false;
            }
            self.set_intra16_mode(best_mode as u8);
        }

        if try_both_modes || !is_i16 {
            is_i16 = false;
            self.start_i4();
            loop {
                let i4 = self.it.i4;
                let mut best_i4_mode = 0usize;
                let mut best_i4_score = MAX_COST;
                let src_off = Y_OFF + SCAN[i4];
                let mode_costs = self.cost_mode_i4(&rd.modes_i4);
                self.make_intra4_preds();
                for mode in 0..NUM_BMODES {
                    let r = &self.it.yuv_p[I4_MODE_OFFSETS[mode]..];
                    let score = i64::from(sse4x4(&self.it.yuv_in[src_off..], r)) * RD_DISTO_MULT
                        + i64::from(mode_costs[mode]) * lambda_d_i4;
                    if score < best_i4_score {
                        best_i4_mode = mode;
                        best_i4_score = score;
                    }
                }
                i4_bit_sum += i64::from(mode_costs[best_i4_mode]);
                rd.modes_i4[i4] = best_i4_mode as u8;
                score_i4 += best_i4_score;
                if score_i4 >= best_score || i4_bit_sum > bit_limit {
                    is_i16 = true;
                    break;
                }
                let mut levels = rd.y_ac_levels[i4];
                nz |= u32::from(self.reconstruct_intra4(&mut levels, best_i4_mode, I4Target::Out2))
                    << i4;
                rd.y_ac_levels[i4] = levels;
                if !self.rotate_i4(false) {
                    break;
                }
            }
        }

        if !is_i16 {
            let modes = rd.modes_i4;
            self.set_intra4_mode(&modes);
            self.it.swap_out();
            best_score = score_i4;
        } else {
            let mode = self.preds[self.preds_index()] as usize;
            nz = self.reconstruct_intra16(&mut rd.y_dc_levels, &mut rd.y_ac_levels, false, mode);
        }

        if refine_uv_mode {
            let mut best_mode = 0usize;
            let mut best_uv_score = MAX_COST;
            let src = &self.it.yuv_in[U_OFF..];
            for mode in 0..NUM_PRED_MODES {
                let r = &self.it.yuv_p[UV_MODE_OFFSETS[mode]..];
                let score = i64::from(sse16x8(src, r)) * RD_DISTO_MULT
                    + i64::from(FIXED_COSTS_UV[mode]) * lambda_d_uv;
                if score < best_uv_score {
                    best_mode = mode;
                    best_uv_score = score;
                }
            }
            self.mb_mut().uv_mode = best_mode as u8;
        }
        let uv_mode = self.mb().uv_mode as usize;
        nz |= self.reconstruct_uv(&mut rd.uv_levels, &mut rd.derr, false, uv_mode);

        rd.nz = nz;
        rd.score = best_score;
    }

    fn decimate(&mut self, rd: &mut ModeScore, rd_opt: RdLevel) -> bool {
        rd.init_score();
        self.make_luma16_preds();
        self.make_chroma8_preds();

        if rd_opt > RdLevel::None {
            self.it.do_trellis = rd_opt >= RdLevel::TrellisAll;
            self.pick_best_intra16(rd);
            if self.method >= 2 {
                self.pick_best_intra4(rd);
            }
            self.pick_best_uv(rd);
            if rd_opt == RdLevel::Trellis {
                self.it.do_trellis = true;
                self.simple_quantize(rd);
            }
        } else {
            self.refine_using_distortion(self.method >= 2, self.method >= 1, rd);
        }
        let is_skipped = rd.nz == 0;
        self.mb_mut().skip = is_skipped;
        is_skipped
    }

    // ---------------------------------------------------------------------------------------
    // Main loop

    fn set_loop_params(&mut self, q: f32) {
        let q = q.clamp(0., 100.);
        self.set_segment_params(q);
        self.set_segment_probas();
        self.proba.calculate_level_costs();
        self.proba.nb_skip = 0;
    }

    fn record_tokens(&mut self, rd: &ModeScore) {
        self.nz_to_bytes();
        let is_i16 = self.mb().mb_type == 1;
        let (tokens, stats) = (&mut self.tokens, &mut self.proba.stats);
        walk_residuals(&mut self.it, is_i16, rd, |ctx, res| {
            tokens.record_coeff_tokens(ctx, res, &mut stats[res.coeff_type])
        });
        self.bytes_to_nz();
    }

    fn record_residuals(&mut self, rd: &ModeScore) {
        self.nz_to_bytes();
        let is_i16 = self.mb().mb_type == 1;
        let stats = &mut self.proba.stats;
        walk_residuals(&mut self.it, is_i16, rd, |ctx, res| {
            res.record(ctx, &mut stats[res.coeff_type])
        });
        self.bytes_to_nz();
    }

    fn code_residuals(&mut self, bw: &mut BitWriter, rd: &ModeScore) {
        self.nz_to_bytes();
        let is_i16 = self.mb().mb_type == 1;
        let coeffs = &self.proba.coeffs;
        walk_residuals(&mut self.it, is_i16, rd, |ctx, res| {
            res.put(bw, ctx, &coeffs[res.coeff_type])
        });
        self.bytes_to_nz();
    }

    /// Codes the recorded tokens of every macroblock with the final probabilities.
    fn emit_tokens(&mut self, bw: &mut BitWriter) {
        let tokens = std::mem::take(&mut self.tokens);
        let mut reader = tokens.reader();
        let probas: Vec<u8> = self
            .proba
            .coeffs
            .as_flattened()
            .as_flattened()
            .as_flattened()
            .to_vec();
        self.iterator_reset();
        loop {
            self.nz_to_bytes();
            let is_i16 = self.mb().mb_type == 1;
            walk_blocks(&mut self.it, is_i16, |ctx, first, ctype, _| {
                reader.emit_block(bw, &probas, ctype, first, ctx)
            });
            self.bytes_to_nz();
            if !self.iterator_next() {
                break;
            }
        }
        self.tokens = tokens;
        self.tokens.clear();
    }

    fn reset_after_skip(&mut self) {
        let x = self.it.x;
        if self.mb().mb_type == 1 {
            self.nz[1 + x] = 0;
            self.it.left_nz[8] = 0;
        } else {
            self.nz[1 + x] &= 1 << 24;
        }
    }

    /// `VP8EncLoop`: a statistics pass to choose the probabilities, then the coding pass.
    fn enc_loop(&mut self) -> BitWriter {
        let average_bytes_per_mb = [50, 24, 16, 9, 7, 5, 3, 2];
        let bytes_per_part =
            self.mb_w * self.mb_h * average_bytes_per_mb[(self.base_quant >> 4) as usize];
        self.stat_loop();

        let mut bw = BitWriter::new(bytes_per_part);
        let rd_opt = self.rd_opt_level;
        self.iterator_reset();
        loop {
            let dont_use_skip = !self.proba.use_skip_proba;
            self.iterator_import(false);
            let mut info = ModeScore::default();
            if !self.decimate(&mut info, rd_opt) || dont_use_skip {
                self.code_residuals(&mut bw, &info);
            } else {
                self.reset_after_skip();
            }
            self.save_boundary();
            if !self.iterator_next() {
                break;
            }
        }
        bw.finish();
        self.adjust_filter_strength();
        bw
    }

    fn stat_loop(&mut self) {
        let method = self.method;
        let fast_probe = method == 0 || method == 3;
        let rd_opt = if method >= 3 {
            RdLevel::Basic
        } else {
            RdLevel::None
        };
        let mut nb_mbs = (self.mb_w * self.mb_h) as i32;
        if fast_probe {
            nb_mbs = if method == 3 {
                if nb_mbs > 200 {
                    nb_mbs >> 1
                } else {
                    100
                }
            } else if nb_mbs > 200 {
                nb_mbs >> 2
            } else {
                50
            };
        }
        self.proba.reset_token_stats();
        loop {
            let mut size_p0 = 0u64;
            let mut left = nb_mbs;
            self.iterator_reset();
            self.set_loop_params(self.config.quality);
            loop {
                self.iterator_import(false);
                let mut info = ModeScore::default();
                if self.decimate(&mut info, rd_opt) {
                    self.proba.nb_skip += 1;
                }
                self.record_residuals(&info);
                size_p0 += info.h as u64;
                self.save_boundary();
                left -= 1;
                if !self.iterator_next() || left <= 0 {
                    break;
                }
            }
            size_p0 += self.segment_hdr_size as u64;
            // Without rate estimates (methods 0-2) and a segment map this is 0, which
            // libwebp takes for a failure: it keeps the default probabilities and no skip flags.
            if size_p0 == 0 {
                return;
            }
            if self.max_i4_header_bits > 0 && size_p0 > PARTITION0_SIZE_LIMIT {
                self.max_i4_header_bits >>= 1;
                continue;
            }
            break;
        }
        self.proba
            .finalize_skip_proba((self.mb_w * self.mb_h) as i32);
        self.proba.finalize_token_probas();
        self.proba.calculate_level_costs();
    }

    /// `VP8EncTokenLoop` for one pass without a size or PSNR target.
    fn token_loop(&mut self) -> BitWriter {
        const MIN_COUNT: i32 = 96;
        let mut max_count = ((self.mb_w * self.mb_h) >> 3) as i32;
        if max_count < MIN_COUNT {
            max_count = MIN_COUNT;
        }
        let rd_opt = self.rd_opt_level;
        let average_bytes_per_mb = [50, 24, 16, 9, 7, 5, 3, 2];
        let quality = self.config.quality;

        loop {
            let mut size_p0 = 0u64;
            let mut cnt = max_count;
            self.iterator_reset();
            self.set_loop_params(quality);
            self.proba.reset_token_stats();
            self.tokens.clear();
            loop {
                self.iterator_import(false);
                cnt -= 1;
                if cnt < 0 {
                    self.proba.finalize_token_probas();
                    self.proba.calculate_level_costs();
                    cnt = max_count;
                }
                let mut info = ModeScore::default();
                self.decimate(&mut info, rd_opt);
                self.record_tokens(&info);
                size_p0 += info.h as u64;
                self.save_boundary();
                if !self.iterator_next() {
                    break;
                }
            }
            size_p0 += self.segment_hdr_size as u64;
            if self.max_i4_header_bits > 0 && size_p0 > PARTITION0_SIZE_LIMIT {
                self.max_i4_header_bits >>= 1;
                continue;
            }
            break;
        }
        self.proba.finalize_token_probas();
        let bytes_per_part =
            self.mb_w * self.mb_h * average_bytes_per_mb[(self.base_quant >> 4) as usize];
        let mut bw = BitWriter::new(bytes_per_part);
        self.emit_tokens(&mut bw);
        bw.finish();
        self.adjust_filter_strength();
        bw
    }

    fn adjust_filter_strength(&mut self) {
        if self.config.filter_strength > 0 {
            let mut max_level = 0;
            for dqm in &mut self.dqm {
                let delta = (dqm.max_edge * i32::from(dqm.y2.q[1])) >> 3;
                let level = filter_strength_from_delta(self.filter_sharpness, delta);
                if level > dqm.fstrength {
                    dqm.fstrength = level;
                }
                max_level = max_level.max(dqm.fstrength);
            }
            self.filter_level = max_level;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Bitstream

    fn code_intra_modes(&mut self, bw: &mut BitWriter) {
        self.iterator_reset();
        loop {
            let mb = *self.mb();
            let preds = self.preds_index();
            if self.update_map {
                let s = mb.segment;
                let p = &self.proba.segments;
                if bw.put_bit(s >= 2, p[0]) {
                    bw.put_bit(s & 1 != 0, p[2]);
                } else {
                    bw.put_bit(s & 1 != 0, p[1]);
                }
            }
            if self.proba.use_skip_proba {
                bw.put_bit(mb.skip, self.proba.skip_proba);
            }
            if bw.put_bit(mb.mb_type != 0, 145) {
                let mode = self.preds[preds];
                if bw.put_bit(mode == 1 || mode == 3, 156) {
                    bw.put_bit(mode == 1, 128);
                } else {
                    bw.put_bit(mode == 2, 163);
                }
            } else {
                for y in 0..4 {
                    let row = preds + y * self.preds_w;
                    let mut left = self.preds[row - 1];
                    for x in 0..4 {
                        let top = self.preds[row - self.preds_w + x];
                        let probas = &BMODES_PROBA[top as usize][left as usize];
                        left = put_i4_mode(bw, self.preds[row + x], probas);
                    }
                }
            }
            let uv_mode = mb.uv_mode;
            if bw.put_bit(uv_mode != 0, 142) && bw.put_bit(uv_mode != 2, 114) {
                bw.put_bit(uv_mode != 3, 183);
            }
            if !self.iterator_next() {
                break;
            }
        }
    }

    fn generate_partition0(&mut self) -> BitWriter {
        let mut bw = BitWriter::new(self.mb_w * self.mb_h * 7 / 8);
        bw.put_bit_uniform(false);
        bw.put_bit_uniform(false);

        // segment header
        if bw.put_bit_uniform(self.num_segments > 1) {
            bw.put_bit_uniform(self.update_map);
            if bw.put_bit_uniform(true) {
                bw.put_bit_uniform(true);
                for s in 0..NUM_MB_SEGMENTS {
                    bw.put_signed_bits(self.dqm[s].quant, 7);
                }
                for s in 0..NUM_MB_SEGMENTS {
                    bw.put_signed_bits(self.dqm[s].fstrength, 6);
                }
            }
            if self.update_map {
                for s in 0..3 {
                    if bw.put_bit_uniform(self.proba.segments[s] != 255) {
                        bw.put_bits(u32::from(self.proba.segments[s]), 8);
                    }
                }
            }
        }

        // filter header
        bw.put_bit_uniform(self.filter_simple);
        bw.put_bits(self.filter_level as u32, 6);
        bw.put_bits(self.filter_sharpness as u32, 3);
        bw.put_bit_uniform(false);

        // one token partition
        bw.put_bits(0, 2);

        bw.put_bits(self.base_quant as u32, 7);
        bw.put_signed_bits(self.dq_y1_dc, 4);
        bw.put_signed_bits(self.dq_y2_dc, 4);
        bw.put_signed_bits(self.dq_y2_ac, 4);
        bw.put_signed_bits(self.dq_uv_dc, 4);
        bw.put_signed_bits(self.dq_uv_ac, 4);
        bw.put_bit_uniform(false);
        self.proba.write(&mut bw);
        self.code_intra_modes(&mut bw);
        bw.finish();
        bw
    }
}

/// Visits the blocks of a macroblock in coding order with their non-zero context, recording
/// whether each had non-zero levels as `f` returns. `f` gets the context, the first
/// coefficient, the coefficient type and the block: 0 for luma DC, 1..=16 luma, 17..=24 chroma.
fn walk_blocks(it: &mut Iter, is_i16: bool, mut f: impl FnMut(usize, usize, usize, usize) -> bool) {
    let (first, ctype) = if is_i16 {
        let ctx = (it.top_nz[8] + it.left_nz[8]) as usize;
        let nz = i32::from(f(ctx, 0, 1, 0));
        it.top_nz[8] = nz;
        it.left_nz[8] = nz;
        (1, 0)
    } else {
        (0, 3)
    };
    for y in 0..4 {
        for x in 0..4 {
            let ctx = (it.top_nz[x] + it.left_nz[y]) as usize;
            let nz = i32::from(f(ctx, first, ctype, 1 + x + y * 4));
            it.top_nz[x] = nz;
            it.left_nz[y] = nz;
        }
    }
    for ch in [0, 2] {
        for y in 0..2 {
            for x in 0..2 {
                let ctx = (it.top_nz[4 + ch + x] + it.left_nz[4 + ch + y]) as usize;
                let nz = i32::from(f(ctx, 0, 2, 17 + ch * 2 + x + y * 2));
                it.top_nz[4 + ch + x] = nz;
                it.left_nz[4 + ch + y] = nz;
            }
        }
    }
}

fn walk_residuals(
    it: &mut Iter,
    is_i16: bool,
    rd: &ModeScore,
    mut f: impl FnMut(usize, &Residual) -> bool,
) {
    walk_blocks(it, is_i16, |ctx, first, ctype, block| {
        let levels = match block {
            0 => &rd.y_dc_levels,
            1..=16 => &rd.y_ac_levels[block - 1],
            _ => &rd.uv_levels[block - 17],
        };
        f(ctx, &Residual::new(first, ctype, levels))
    });
}

#[derive(Clone, Copy, PartialEq)]
enum I4Target {
    Out,
    Out2,
}

fn get_alpha((max_value, last_non_zero): (i32, i32)) -> i32 {
    if max_value > 1 {
        2 * 255 * last_non_zero / max_value
    } else {
        0
    }
}

fn get_proba(a: i32, b: i32) -> u8 {
    let total = a + b;
    if total == 0 {
        255
    } else {
        ((255 * a + total / 2) / total) as u8
    }
}

fn filter_strength_from_delta(sharpness: i32, delta: i32) -> i32 {
    let pos = delta.min(63) as usize;
    i32::from(LEVELS_FROM_DELTA[sharpness as usize][pos])
}

fn quality_to_compression(c: f64) -> f64 {
    let linear_c = if c < 0.75 { c * (2. / 3.) } else { 2. * c - 1. };
    linear_c.powf(1. / 3.)
}

fn expand_matrix(m: &mut Matrix, ty: usize) -> i32 {
    for i in 0..2 {
        let bias_v = BIAS_MATRICES[ty][usize::from(i > 0)];
        m.iq[i] = ((1 << QFIX) / u32::from(m.q[i])) as u16;
        m.bias[i] = bias(bias_v);
        m.zthresh[i] = ((1 << QFIX) - 1 - m.bias[i]) / u32::from(m.iq[i]);
    }
    for i in 2..16 {
        m.q[i] = m.q[1];
        m.iq[i] = m.iq[1];
        m.bias[i] = m.bias[1];
        m.zthresh[i] = m.zthresh[1];
    }
    let mut sum = 0i32;
    for i in 0..16 {
        m.sharpen[i] = if ty == 0 {
            (FREQ_SHARPENING[i] * m.q[i]) >> SHARPEN_BITS
        } else {
            0
        };
        sum += i32::from(m.q[i]);
    }
    (sum + 8) >> 4
}

#[inline]
fn mult_8b(a: i32, b: i32) -> i32 {
    (a * b + 128) >> 8
}

#[inline]
fn is_flat_fn(levels: &[i16], num_blocks: usize, thresh: i32) -> bool {
    is_flat(levels, num_blocks, thresh)
}

fn import_block<const SIZE: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    w: usize,
    h: usize,
) {
    if w == SIZE && h == SIZE {
        for i in 0..SIZE {
            let row: &[u8; SIZE] = src[i * src_stride..][..SIZE].try_into().unwrap();
            dst[i * BPS..][..SIZE].copy_from_slice(row);
        }
        return;
    }
    for i in 0..h {
        let row = &mut dst[i * BPS..i * BPS + SIZE];
        row[..w].copy_from_slice(&src[i * src_stride..i * src_stride + w]);
        let last = row[w - 1];
        row[w..].fill(last);
    }
    for i in h..SIZE {
        let (prev, cur) = dst.split_at_mut(i * BPS);
        cur[..SIZE].copy_from_slice(&prev[(i - 1) * BPS..(i - 1) * BPS + SIZE]);
    }
}

fn import_line(src: &[u8], src_stride: usize, dst: &mut [u8], len: usize) {
    for i in 0..len {
        dst[i] = src[i * src_stride];
    }
    let last = dst[len - 1];
    dst[len..].fill(last);
}

fn put_i4_mode(bw: &mut BitWriter, mode: u8, prob: &[u8; 9]) -> u8 {
    const B_TM_PRED: u8 = 1;
    const B_VE_PRED: u8 = 2;
    const B_HE_PRED: u8 = 3;
    const B_RD_PRED: u8 = 4;
    const B_LD_PRED: u8 = 6;
    const B_VL_PRED: u8 = 7;
    const B_HD_PRED: u8 = 8;
    if bw.put_bit(mode != B_DC_PRED, prob[0])
        && bw.put_bit(mode != B_TM_PRED, prob[1])
        && bw.put_bit(mode != B_VE_PRED, prob[2])
    {
        if !bw.put_bit(mode >= B_LD_PRED, prob[3]) {
            if bw.put_bit(mode != B_HE_PRED, prob[4]) {
                bw.put_bit(mode != B_RD_PRED, prob[5]);
            }
        } else if bw.put_bit(mode != B_LD_PRED, prob[6]) && bw.put_bit(mode != B_VL_PRED, prob[7]) {
            bw.put_bit(mode != B_HD_PRED, prob[8]);
        }
    }
    mode
}

/// Trellis quantization of one block (libwebp's `TrellisQuantizeBlock`), used by methods 5
/// and 6.
fn trellis_quantize_block(
    proba: &Proba,
    input: &mut [i16; 16],
    out: &mut [i16; 16],
    ctx0: usize,
    coeff_type: usize,
    mtx: &Matrix,
    lambda: i32,
) -> bool {
    const MIN_DELTA: i32 = 0;
    const MAX_DELTA: i32 = 1;
    const NUM_NODES: usize = (MIN_DELTA + 1 + MAX_DELTA) as usize;

    #[derive(Clone, Copy, Default)]
    struct Node {
        prev: i8,
        sign: bool,
        level: i16,
    }
    #[derive(Clone, Copy)]
    struct ScoreState {
        score: i64,
        /// (n, ctx) of the cost table
        costs: (usize, usize),
    }

    let rd_score =
        |rate: i64, distortion: i64| rate * i64::from(lambda) + RD_DISTO_MULT * distortion;
    let probas = &proba.coeffs[coeff_type];
    let first = usize::from(coeff_type == 0);
    let mut nodes = [[Node::default(); NUM_NODES]; 16];
    let mut ss = [[ScoreState {
        score: 0,
        costs: (0, 0),
    }; NUM_NODES]; 2];
    let (mut cur, mut prev) = (0usize, 1usize);
    let mut best_path = [-1i32; 3];

    let thresh = i32::from(mtx.q[1]) * i32::from(mtx.q[1]) / 4;
    let last_proba = probas[ENC_BANDS[first] as usize][ctx0][0];
    let mut last = first as i32 - 1;
    for n in (first..16).rev() {
        let j = ZIGZAG[n];
        let err = i32::from(input[j]) * i32::from(input[j]);
        if err > thresh {
            last = n as i32;
            break;
        }
    }
    if last < 15 {
        last += 1;
    }

    let mut best_score = rd_score(i64::from(bit_cost(false, last_proba)), 0);
    for m in 0..NUM_NODES {
        let rate = if ctx0 == 0 {
            i64::from(bit_cost(true, last_proba))
        } else {
            0
        };
        ss[cur][m] = ScoreState {
            score: rd_score(rate, 0),
            costs: (first, ctx0),
        };
    }

    for n in first..(last + 1).max(0) as usize {
        let j = ZIGZAG[n];
        let q = u32::from(mtx.q[j]);
        let iq = u32::from(mtx.iq[j]);
        let b = bias(0);
        let sign = input[j] < 0;
        let coeff0 = i32::from(input[j]).unsigned_abs() + u32::from(mtx.sharpen[j]);
        let level0 = quantdiv(coeff0, iq, b).min(MAX_LEVEL);
        let thresh_level = quantdiv(coeff0, iq, bias(0x80)).min(MAX_LEVEL);

        std::mem::swap(&mut cur, &mut prev);

        for m in 0..NUM_NODES {
            let level = level0 + m as i32 - MIN_DELTA;
            let ctx = level.clamp(0, 2) as usize;
            let band = ENC_BANDS[n + 1] as usize;
            ss[cur][m].costs = (n + 1, ctx);
            if level < 0 || level > thresh_level {
                ss[cur][m].score = MAX_COST;
                continue;
            }
            let new_error = (coeff0 as i32).wrapping_sub(level.wrapping_mul(q as i32));
            let delta_error = i32::from(WEIGHT_TRELLIS[j]).wrapping_mul(
                new_error
                    .wrapping_mul(new_error)
                    .wrapping_sub(coeff0.wrapping_mul(coeff0) as i32),
            );
            let delta_error = i64::from(delta_error);
            let base_score = rd_score(0, delta_error);

            let cost_of = |s: &ScoreState| {
                let table = proba.costs(coeff_type, s.costs.0, s.costs.1);
                i64::from(level_cost(table, level as usize))
            };
            let mut best_cur_score = ss[prev][0].score + rd_score(cost_of(&ss[prev][0]), 0);
            let mut best_prev = 0i8;
            for p in 1..NUM_NODES {
                let score = ss[prev][p].score + rd_score(cost_of(&ss[prev][p]), 0);
                if score < best_cur_score {
                    best_cur_score = score;
                    best_prev = p as i8;
                }
            }
            best_cur_score += base_score;
            nodes[n][m] = Node {
                prev: best_prev,
                sign,
                level: level as i16,
            };
            ss[cur][m].score = best_cur_score;

            if level != 0 && best_cur_score < best_score {
                let last_pos_cost = if n < 15 {
                    i64::from(bit_cost(false, probas[band][ctx][0]))
                } else {
                    0
                };
                let score = best_cur_score + rd_score(last_pos_cost, 0);
                if score < best_score {
                    best_score = score;
                    best_path = [n as i32, m as i32, i32::from(best_prev)];
                }
            }
        }
    }

    if coeff_type == 0 {
        input[1..].fill(0);
        out[1..].fill(0);
    } else {
        input.fill(0);
        out.fill(0);
    }
    if best_path[0] == -1 {
        return false;
    }
    let mut nz = 0;
    let mut best_node = best_path[1] as usize;
    let mut n = best_path[0];
    nodes[n as usize][best_node].prev = best_path[2] as i8;
    while n >= first as i32 {
        let node = nodes[n as usize][best_node];
        let j = ZIGZAG[n as usize];
        out[n as usize] = if node.sign { -node.level } else { node.level };
        nz |= node.level;
        input[j] = out[n as usize] * mtx.q[j] as i16;
        best_node = node.prev as usize;
        n -= 1;
    }
    nz != 0
}

/// The pieces of a `VP8 ` payload, whether some pixel is not opaque and the alpha plane if
/// it was kept.
pub(crate) type EncodedFrame = (Vec<Vec<u8>>, (bool, Option<Vec<u8>>));

/// Encodes `src` as a VP8 bitstream (the payload of a `VP8 ` chunk), in pieces to write one
/// after the other. A first partition that does not fit its 19-bit size field is coded again
/// without intra 4x4 modes, at method 3 at least, as libwebp's callers restart it with
/// `partition_limit` 100; one that still does not fit is `Partition0Overflow`.
pub(crate) fn encode(
    mut src: Source,
    mut config: Config,
) -> Result<EncodedFrame, crate::EncodingError> {
    let (width, height) = (src.width, src.height);
    if config.keep_strips {
        src.keep_strips();
    }
    let (mut enc, part0, tokens) = loop {
        let mut enc = Encoder::new(config, src);
        enc.analyze();
        let tokens = if enc.rd_opt_level >= RdLevel::Basic {
            enc.token_loop()
        } else {
            enc.enc_loop()
        };
        let part0 = enc.generate_partition0();
        if (part0.buf.len() as u64) < VP8_MAX_PARTITION0_SIZE {
            break (enc, part0, tokens);
        }
        if config.partition_limit >= 100 {
            return Err(crate::EncodingError::Partition0Overflow);
        }
        config.method = config.method.max(3);
        config.partition_limit = 100;
        src = enc.src;
    };

    let size0 = part0.buf.len();
    let mut head = Vec::with_capacity(10 + size0);
    let bits = (enc.profile << 1) | (1 << 4) | ((size0 as u32) << 5);
    head.extend_from_slice(&bits.to_le_bytes()[..3]);
    head.extend_from_slice(&[0x9d, 0x01, 0x2a]);
    head.extend_from_slice(&(width as u16).to_le_bytes());
    head.extend_from_slice(&(height as u16).to_le_bytes());
    head.extend_from_slice(&part0.buf);
    let mut parts = vec![head, tokens.buf];
    // libwebp counts the padding byte as part of the frame.
    if parts.iter().map(Vec::len).sum::<usize>() % 2 == 1 {
        parts.push(vec![0]);
    }
    Ok((parts, enc.src.take_alpha()))
}

/// Encodes rows of `color` pixels as the payload of a `VP8 ` chunk, in pieces, like libwebp at
/// `quality` and `method` with the default preset. `read_row(y, row)` fills `row`; rows are
/// read top to bottom once per pass. Also returns whether some pixel is not opaque, and then
/// the alpha plane if `keep_alpha`.
pub(crate) fn encode_frame_lossy(
    width: u32,
    height: u32,
    color: crate::ColorType,
    quality: u8,
    method: u8,
    keep_alpha: bool,
    read_row: impl FnMut(u32, &mut [u8]),
) -> Result<EncodedFrame, crate::EncodingError> {
    let config = Config::new(f32::from(quality.min(100)), i32::from(method.min(6)));
    encode_frame_lossy_with(config, width, height, color, keep_alpha, read_row)
}

/// [`encode_frame_lossy`] with a `config` of its own.
pub(crate) fn encode_frame_lossy_with(
    config: Config,
    width: u32,
    height: u32,
    color: crate::ColorType,
    keep_alpha: bool,
    mut read_row: impl FnMut(u32, &mut [u8]),
) -> Result<EncodedFrame, crate::EncodingError> {
    use crate::ColorType;
    if width == 0 || width > 16383 || height == 0 || height > 16383 {
        return Err(crate::EncodingError::InvalidDimensions);
    }
    let (w, h) = (width as usize, height as usize);
    let mut src = match color {
        ColorType::Rgb8 => Source::new(w, h, 3, move |y, row| read_row(y as u32, row)),
        ColorType::Rgba8 => Source::new(w, h, 4, move |y, row| read_row(y as u32, row)),
        ColorType::L8 => {
            let mut gray = vec![0u8; w];
            Source::new(w, h, 3, move |y, row| {
                read_row(y as u32, &mut gray);
                for (px, &l) in row.chunks_exact_mut(3).zip(&gray) {
                    px.fill(l);
                }
            })
        }
        ColorType::La8 => {
            let mut gray = vec![0u8; w * 2];
            Source::new(w, h, 4, move |y, row| {
                read_row(y as u32, &mut gray);
                for (px, la) in row.chunks_exact_mut(4).zip(gray.chunks_exact(2)) {
                    px.copy_from_slice(&[la[0], la[0], la[0], la[1]]);
                }
            })
        }
    };
    src.keep_alpha = keep_alpha;
    encode(src, config)
}
