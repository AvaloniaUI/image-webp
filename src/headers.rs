//! Where libwebp's decoder finds the bitstream (`ParseHeadersInternal` in `dec/webp_dec.c`):
//! the RIFF and VP8X headers, the optional chunks up to the first VP8/VP8L chunk with the last
//! ALPH chunk among them, and the bitstream's own header. Accepts what `WebPGetFeatures` and
//! `WebPDecode` accept, including the bare `ALPH` + `VP8` and `VP8`/`VP8L` data a demuxer hands
//! out for a frame.

use std::io::{Cursor, Read, Seek, SeekFrom};
use std::ops::Range;

const TAG_SIZE: usize = 4;
const CHUNK_HEADER_SIZE: usize = 8;
const RIFF_HEADER_SIZE: usize = 12;
const VP8X_CHUNK_SIZE: usize = 10;
const MAX_CHUNK_PAYLOAD: usize = !0u32 as usize - CHUNK_HEADER_SIZE - 1;
const MAX_IMAGE_AREA: u64 = 1 << 32;
const VP8_FRAME_HEADER_SIZE: usize = 10;
const VP8L_FRAME_HEADER_SIZE: usize = 5;
const ANIMATION_FLAG: u32 = 0x02;
const ALPHA_FLAG: u32 = 0x10;

/// Why the headers were refused, as libwebp's status codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderError {
    /// `VP8_STATUS_BITSTREAM_ERROR`.
    Bitstream,
    /// `VP8_STATUS_NOT_ENOUGH_DATA`.
    NotEnoughData,
}

/// What the headers say.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Headers {
    /// The bitstream's size, or the canvas' for an animation.
    pub width: u32,
    /// The bitstream's size, or the canvas' for an animation.
    pub height: u32,
    /// VP8X alpha flag, VP8L alpha bit, or an ALPH chunk.
    pub has_alpha: bool,
    /// VP8X animation flag.
    pub has_animation: bool,
    /// The VP8L chunk or raw VP8L data.
    pub is_lossless: bool,
    /// The ALPH chunk's payload.
    pub alpha: Option<Range<usize>>,
    /// The bitstream: from its first byte to the size its chunk header gives.
    pub image: Range<usize>,
}

fn le32(b: &[u8]) -> usize {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
}

fn le24(b: &[u8]) -> u64 {
    u64::from(b[0]) | u64::from(b[1]) << 8 | u64::from(b[2]) << 16
}

/// `WebPGetFeatures`: an animation is answered from its VP8X header.
pub fn features(data: &[u8]) -> Result<Headers, HeaderError> {
    read_features(&mut Cursor::new(data), data.len() as u64)
}

/// The headers as `WebPDecode` parses them before decoding, all data present.
pub fn decoding(data: &[u8]) -> Result<Headers, HeaderError> {
    read_decoding(&mut Cursor::new(data), 0, data.len() as u64)
}

/// [`features`] of the first `len` bytes `r` reads.
pub fn read_features<R: Read + Seek>(r: &mut R, len: u64) -> Result<Headers, HeaderError> {
    parse(&mut Data { r, start: 0, len: len as usize }, false)
}

/// [`decoding`] of the `len` bytes `r` reads from `start` on; the ranges found are counted
/// from `start`.
pub fn read_decoding<R: Read + Seek>(r: &mut R, start: u64, len: u64) -> Result<Headers, HeaderError> {
    parse(&mut Data { r, start, len: len as usize }, true)
}

/// The data the headers are read from: small pieces at offsets.
struct Data<'r, R> {
    r: &'r mut R,
    start: u64,
    len: usize,
}

impl<R: Read + Seek> Data<'_, R> {
    /// The `N` bytes at `at`, which the caller checked are there.
    fn get<const N: usize>(&mut self, at: usize) -> Result<[u8; N], HeaderError> {
        let mut b = [0; N];
        self.r
            .seek(SeekFrom::Start(self.start + at as u64))
            .and_then(|_| self.r.read_exact(&mut b))
            .map_err(|_| HeaderError::NotEnoughData)?;
        Ok(b)
    }

    fn le32(&mut self, at: usize) -> Result<usize, HeaderError> {
        Ok(u32::from_le_bytes(self.get(at)?) as usize)
    }

    fn tag(&mut self, at: usize) -> Result<[u8; 4], HeaderError> {
        self.get(at)
    }
}

fn parse<R: Read + Seek>(data: &mut Data<R>, decoding: bool) -> Result<Headers, HeaderError> {
    use HeaderError::*;
    let mut h = Headers::default();
    let len = data.len;
    if len < RIFF_HEADER_SIZE {
        return Err(NotEnoughData);
    }
    let mut at = 0;
    let mut riff_size = 0;
    let riff: [u8; RIFF_HEADER_SIZE] = data.get(0)?;
    if &riff[..4] == b"RIFF" {
        if &riff[8..12] != b"WEBP" {
            return Err(Bitstream);
        }
        let size = le32(&riff[4..]);
        if size < TAG_SIZE + CHUNK_HEADER_SIZE || size > MAX_CHUNK_PAYLOAD {
            return Err(Bitstream);
        }
        if decoding && size > len - CHUNK_HEADER_SIZE {
            return Err(NotEnoughData);
        }
        riff_size = size;
        at = RIFF_HEADER_SIZE;
    }
    let found_riff = riff_size > 0;
    let rest = |at: usize| len - at;

    if rest(at) < CHUNK_HEADER_SIZE {
        return Err(NotEnoughData);
    }
    let mut found_vp8x = false;
    let (mut canvas_width, mut canvas_height) = (0, 0);
    if &data.tag(at)? == b"VP8X" {
        if data.le32(at + 4)? != VP8X_CHUNK_SIZE {
            return Err(Bitstream);
        }
        if rest(at) < CHUNK_HEADER_SIZE + VP8X_CHUNK_SIZE {
            return Err(NotEnoughData);
        }
        let vp8x: [u8; VP8X_CHUNK_SIZE] = data.get(at + CHUNK_HEADER_SIZE)?;
        let flags = le32(&vp8x) as u32;
        canvas_width = 1 + le24(&vp8x[4..]);
        canvas_height = 1 + le24(&vp8x[7..]);
        if canvas_width * canvas_height >= MAX_IMAGE_AREA {
            return Err(Bitstream);
        }
        h.has_alpha = flags & ALPHA_FLAG != 0;
        h.has_animation = flags & ANIMATION_FLAG != 0;
        at += CHUNK_HEADER_SIZE + VP8X_CHUNK_SIZE;
        found_vp8x = true;
    }
    if !found_riff && found_vp8x {
        return Err(Bitstream);
    }
    h.width = canvas_width as u32;
    h.height = canvas_height as u32;
    // Not enough data is no error for `WebPGetFeatures` once a VP8X header was read.
    let short = |mut h: Headers| {
        h.has_alpha |= h.alpha.is_some();
        if found_vp8x && !decoding { Ok(h) } else { Err(NotEnoughData) }
    };
    if found_vp8x && h.has_animation && !decoding {
        return Ok(h);
    }
    if rest(at) < TAG_SIZE {
        return short(h);
    }

    if (found_riff && found_vp8x) || (!found_riff && !found_vp8x && &data.tag(at)? == b"ALPH") {
        let mut total = TAG_SIZE + CHUNK_HEADER_SIZE + VP8X_CHUNK_SIZE;
        loop {
            if rest(at) < CHUNK_HEADER_SIZE {
                return short(h);
            }
            let size = data.le32(at + 4)?;
            if size > MAX_CHUNK_PAYLOAD {
                return Err(Bitstream);
            }
            let disk = (CHUNK_HEADER_SIZE + size + 1) & !1;
            total += disk;
            if riff_size > 0 && total > riff_size {
                return Err(Bitstream);
            }
            let tag = data.tag(at)?;
            if &tag == b"VP8 " || &tag == b"VP8L" {
                break;
            }
            if rest(at) < disk {
                return short(h);
            }
            if &tag == b"ALPH" {
                h.alpha = Some(at + CHUNK_HEADER_SIZE..at + CHUNK_HEADER_SIZE + size);
            }
            at += disk;
        }
    }

    if rest(at) < CHUNK_HEADER_SIZE {
        return short(h);
    }
    let tag = data.tag(at)?;
    let compressed_size;
    if &tag == b"VP8 " || &tag == b"VP8L" {
        let size = data.le32(at + 4)?;
        let minimal = TAG_SIZE + CHUNK_HEADER_SIZE;
        if riff_size >= minimal && size > riff_size - minimal {
            return Err(Bitstream);
        }
        if decoding && size > rest(at) - CHUNK_HEADER_SIZE {
            return Err(NotEnoughData);
        }
        h.is_lossless = &tag == b"VP8L";
        compressed_size = size;
        at += CHUNK_HEADER_SIZE;
    } else {
        h.is_lossless = rest(at) >= VP8L_FRAME_HEADER_SIZE && {
            let b: [u8; VP8L_FRAME_HEADER_SIZE] = data.get(at)?;
            b[0] == 0x2f && b[4] >> 5 == 0
        };
        compressed_size = rest(at);
    }
    if compressed_size > MAX_CHUNK_PAYLOAD {
        return Err(Bitstream);
    }
    let (width, height) = if !h.is_lossless {
        if rest(at) < VP8_FRAME_HEADER_SIZE {
            return short(h);
        }
        vp8_info(&data.get::<VP8_FRAME_HEADER_SIZE>(at)?, compressed_size).ok_or(Bitstream)?
    } else {
        if rest(at) < VP8L_FRAME_HEADER_SIZE {
            return short(h);
        }
        let (w, hh, alpha) = vp8l_info(&data.get::<VP8L_FRAME_HEADER_SIZE>(at)?).ok_or(Bitstream)?;
        h.has_alpha = alpha;
        (w, hh)
    };
    if found_vp8x && (u64::from(width), u64::from(height)) != (canvas_width, canvas_height) {
        return Err(Bitstream);
    }
    h.width = width;
    h.height = height;
    h.has_alpha |= h.alpha.is_some();
    h.image = at..at + compressed_size;
    Ok(h)
}

/// `VP8GetInfo`.
fn vp8_info(b: &[u8], chunk_size: usize) -> Option<(u32, u32)> {
    if b[3..6] != [0x9d, 0x01, 0x2a] {
        return None;
    }
    let bits = le24(b) as usize;
    let w = (u32::from(b[7]) << 8 | u32::from(b[6])) & 0x3fff;
    let h = (u32::from(b[9]) << 8 | u32::from(b[8])) & 0x3fff;
    let ok = bits & 1 == 0 && (bits >> 1) & 7 <= 3 && (bits >> 4) & 1 == 1 && bits >> 5 < chunk_size && w != 0 && h != 0;
    ok.then_some((w, h))
}

/// `VP8LGetInfo`.
fn vp8l_info(b: &[u8]) -> Option<(u32, u32, bool)> {
    if b[0] != 0x2f || b[4] >> 5 != 0 {
        return None;
    }
    let bits = u64::from_le_bytes([b[1], b[2], b[3], b[4], 0, 0, 0, 0]);
    Some(((bits & 0x3fff) as u32 + 1, ((bits >> 14) & 0x3fff) as u32 + 1, (bits >> 28) & 1 != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_lossless_header() {
        let mut f = b"RIFF\x1a\0\0\0WEBPVP8L\x0d\0\0\0\x2f".to_vec();
        // 3x2, alpha bit set, version 0.
        let bits: u64 = 2 | 1 << 14 | 1 << 28;
        f.extend_from_slice(&bits.to_le_bytes()[..4]);
        f.extend_from_slice(&[0; 9]);
        let h = decoding(&f).unwrap();
        assert_eq!((h.width, h.height, h.has_alpha, h.is_lossless), (3, 2, true, true));
        assert_eq!(h.image, 20..33);
        assert_eq!(features(&f[..10]), Err(HeaderError::NotEnoughData));
    }
}
