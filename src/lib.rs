//! Decoding and Encoding of WebP Images

#![deny(unsafe_code)]
#![deny(missing_docs)]
// Increase recursion limit for the `quick_error!` macro.
#![recursion_limit = "256"]
// Enable nightly benchmark functionality if "_benchmarks" feature is enabled.
#![cfg_attr(all(test, feature = "_benchmarks"), feature(test))]
#[cfg(all(test, feature = "_benchmarks"))]
extern crate test;

pub use decoder::{
    BandReader, ChunkData, ChunkInfo, DecodingError, FrameInfo, LoopCount, UpsamplingMethod,
    WebPDecodeOptions, WebPDecoder, ALPHA_FLAG, ANIMATION_FLAG, EXIF_FLAG, ICCP_FLAG, XMP_FLAG,
};
pub use frame::FrameDecoder;
pub use encoder::{encode_alpha_chunk, ColorType, EncodingError, WebPEncoder};
pub use lossless::EncoderParams;

/// The data a decoder reads at offsets.
pub(crate) trait ReadSeek: std::io::Read + std::io::Seek {}
impl<R: std::io::Read + std::io::Seek> ReadSeek for R {}

mod alpha_blending;
mod decoder;
mod encoder;
mod extended;
mod frame;
pub mod headers;
mod rescaler;
mod lossless;
mod lossy;

/// An implementation of the VP8 Video Codec
///
/// This module contains a partial implementation of the
/// VP8 video format as defined in RFC-6386.
///
/// It decodes Keyframes only.
/// VP8 is the underpinning of the WebP image format
///
/// # Related Links
/// * [rfc-6386](http://tools.ietf.org/html/rfc6386) - The VP8 Data Format and Decoding Guide
/// * [VP8.pdf](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37073.pdf) - An overview of of the VP8 format
pub mod vp8 {
    pub use crate::lossy::{Frame, Vp8Decoder};
}

/// The `ALPH` chunk payload of an alpha plane at `method`, for benchmarks.
#[doc(hidden)]
pub fn bench_encode_alpha(alpha: &[u8], width: u32, height: u32, method: u8) -> usize {
    lossless::vp8l::encode_alpha(alpha, width, height, method).len()
}

/// The unfiltered VP8L stream of an alpha plane at `method`, for benchmarks.
#[doc(hidden)]
pub fn bench_encode_alpha_plane(alpha: &[u8], width: u32, height: u32, method: u8) -> usize {
    lossless::vp8l::bench_alpha_plane(alpha, width, height, method)
}
