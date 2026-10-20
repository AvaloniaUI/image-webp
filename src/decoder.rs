use quick_error::quick_error;

use std::io::{self, BufRead, Read, Seek, SeekFrom};
use std::num::NonZeroU16;

use crate::extended;
use crate::frame::FrameDecoder;

quick_error! {
    /// Errors that can occur when attempting to decode a WebP image
    #[derive(Debug)]
    #[non_exhaustive]
    pub enum DecodingError {
        /// An IO error occurred while reading the file
        IoError(err: io::Error) {
            from()
            display("IO Error: {}", err)
            source(err)
        }

        /// RIFF's "RIFF" signature not found or invalid
        RiffSignatureInvalid(err: [u8; 4]) {
            display("Invalid RIFF signature: {err:x?}")
        }

        /// WebP's "WEBP" signature not found or invalid
        WebpSignatureInvalid(err: [u8; 4]) {
            display("Invalid WebP signature: {err:x?}")
        }

        /// An expected chunk was missing
        ChunkMissing {
            display("An expected chunk was missing")
        }

        /// Chunk Header was incorrect or invalid in its usage
        ChunkHeaderInvalid(err: [u8; 4]) {
            display("Invalid Chunk header: {err:x?}")
        }

        #[allow(deprecated)]
        #[deprecated]
        /// Some bits were invalid
        ReservedBitSet {
            display("Reserved bits set")
        }

        /// The ALPH chunk preprocessing info flag was invalid
        InvalidAlphaPreprocessing {
            display("Alpha chunk preprocessing flag invalid")
        }

        /// Invalid compression method
        InvalidCompressionMethod {
            display("Invalid compression method")
        }

        /// Alpha chunk doesn't match the frame's size
        AlphaChunkSizeMismatch {
            display("Alpha chunk size mismatch")
        }

        /// Image is too large, either for the platform's pointer size or generally
        ImageTooLarge {
            display("Image too large")
        }

        /// Frame would go out of the canvas
        FrameOutsideImage {
            display("Frame outside image")
        }

        /// Signature of 0x2f not found
        LosslessSignatureInvalid(err: u8) {
            display("Invalid lossless signature: {err:x?}")
        }

        /// Version Number was not zero
        VersionNumberInvalid(err: u8) {
            display("Invalid lossless version number: {err}")
        }

        /// Invalid color cache bits
        InvalidColorCacheBits(err: u8) {
            display("Invalid color cache bits: {err}")
        }

        /// An invalid Huffman code was encountered
        HuffmanError {
            display("Invalid Huffman code")
        }

        /// The bitstream was somehow corrupt
        BitStreamError {
            display("Corrupt bitstream")
        }

        /// The transforms specified were invalid
        TransformError {
            display("Invalid transform")
        }

        /// VP8's `[0x9D, 0x01, 0x2A]` magic not found or invalid
        Vp8MagicInvalid(err: [u8; 3]) {
            display("Invalid VP8 magic: {err:x?}")
        }

        /// VP8 Decoder initialisation wasn't provided with enough data
        NotEnoughInitData {
            display("Not enough VP8 init data")
        }

        /// At time of writing, only the YUV colour-space encoded as `0` is specified
        ColorSpaceInvalid(err: u8) {
            display("Invalid VP8 color space: {err}")
        }

        /// LUMA prediction mode was not recognised
        LumaPredictionModeInvalid(err: i8) {
            display("Invalid VP8 luma prediction mode: {err}")
        }

        /// Intra-prediction mode was not recognised
        IntraPredictionModeInvalid(err: i8) {
            display("Invalid VP8 intra prediction mode: {err}")
        }

        /// Chroma prediction mode was not recognised
        ChromaPredictionModeInvalid(err: i8) {
            display("Invalid VP8 chroma prediction mode: {err}")
        }

        /// Inconsistent image sizes
        InconsistentImageSizes {
            display("Inconsistent image sizes")
        }

        /// The file may be valid, but this crate doesn't support decoding it.
        UnsupportedFeature(err: String) {
            display("Unsupported feature: {err}")
        }

        /// Invalid function call or parameter
        InvalidParameter(err: String) {
            display("Invalid parameter: {err}")
        }

        /// Memory limit exceeded
        MemoryLimitExceeded {
            display("Memory limit exceeded")
        }

        /// Invalid chunk size
        InvalidChunkSize {
            display("Invalid chunk size")
        }

        /// No more frames in image
        NoMoreFrames {
            display("No more frames")
        }

        /// The chunks break the rules libwebp's demuxer holds them to
        InvalidContainer {
            display("Invalid WebP container")
        }
    }
}

struct AnimationState {
    next_frame: u32,
    dispose_next_frame: bool,
    previous_frame_width: u32,
    previous_frame_height: u32,
    previous_frame_x_offset: u32,
    previous_frame_y_offset: u32,
    canvas: Option<Vec<u8>>,
}
impl Default for AnimationState {
    fn default() -> Self {
        Self {
            next_frame: 0,
            dispose_next_frame: true,
            previous_frame_width: 0,
            previous_frame_height: 0,
            previous_frame_x_offset: 0,
            previous_frame_y_offset: 0,
            canvas: None,
        }
    }
}

/// Number of times that an animation loops.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LoopCount {
    /// The animation loops forever.
    Forever,
    /// Each frame of the animation is displayed the specified number of times.
    Times(NonZeroU16),
}

/// WebP decoder configuration options
#[derive(Clone)]
#[non_exhaustive]
pub struct WebPDecodeOptions {
    /// The upsampling method used in conversion from lossy yuv to rgb
    ///
    /// Defaults to `Bilinear`.
    pub lossy_upsampling: UpsamplingMethod,
}

impl Default for WebPDecodeOptions {
    fn default() -> Self {
        Self {
            lossy_upsampling: UpsamplingMethod::Bilinear,
        }
    }
}

/// Methods for upsampling the chroma values in lossy decoding
///
/// The chroma red and blue planes are encoded in VP8 as half the size of the luma plane
/// Therefore we need to upsample these values up to fit each pixel in the image.
#[derive(Clone, Copy, Default)]
pub enum UpsamplingMethod {
    /// Fancy upsampling
    ///
    /// Does bilinear interpolation using the 4 values nearest to the pixel, weighting based on the distance
    /// from the pixel.
    #[default]
    Bilinear,
    /// Simple upsampling, just uses the closest u/v value to the pixel when upsampling
    ///
    /// Matches the -nofancy option in dwebp.
    /// Should be faster but may lead to slightly jagged edges.
    Simple,
}

/// WebP image format decoder.
///
/// Creating it walks the file's chunks as libwebp's demuxer does (`src/demux/demux.c`), with
/// the same accepted and rejected files, reading only chunk headers and the first bytes of each
/// bitstream; pixels are decoded a frame at a time when asked for.
pub struct WebPDecoder<R> {
    r: R,
    memory_limit: usize,
    layout: Layout,
    walk: Walk,
    background_color: Option<[u8; 4]>,
    animation: AnimationState,
    webp_decode_options: WebPDecodeOptions,
}

impl<R: BufRead + Seek> WebPDecoder<R> {
    /// Create a new `WebPDecoder` from the reader `r`. The decoder performs many small reads, so the
    /// reader should be buffered.
    pub fn new(r: R) -> Result<Self, DecodingError> {
        Self::new_with_options(r, WebPDecodeOptions::default())
    }

    /// Create a new `WebPDecoder` from the reader `r` with the options `WebPDecodeOptions`. The decoder
    /// performs many small reads, so the reader should be buffered.
    pub fn new_with_options(
        r: R,
        webp_decode_options: WebPDecodeOptions,
    ) -> Result<Self, DecodingError> {
        let mut d = Self::start_with_options(r, webp_decode_options)?;
        d.finish()?;
        Ok(d)
    }

    /// [`new`](Self::new) as far as the first frame: the RIFF and VP8X headers and the chunks
    /// through the first frame's image chunk are read, the rest of the file as
    /// [`walk_frame`](Self::walk_frame) and [`finish`](Self::finish) ask for it, so a frame
    /// decodes before the file has arrived. [`frames`](Self::frames) and
    /// [`chunks`](Self::chunks) hold what has been walked, and the ANIM chunk of a file that is
    /// no animation may still follow; the checks `new` makes of the whole file are `finish`'s.
    pub fn start(r: R) -> Result<Self, DecodingError> {
        Self::start_with_options(r, WebPDecodeOptions::default())
    }

    /// [`start`](Self::start) with the options `WebPDecodeOptions`.
    pub fn start_with_options(
        mut r: R,
        webp_decode_options: WebPDecodeOptions,
    ) -> Result<Self, DecodingError> {
        let (layout, walk) = begin(&mut r)?;
        let mut d = Self {
            r,
            memory_limit: usize::MAX,
            layout,
            walk,
            background_color: None,
            animation: Default::default(),
            webp_decode_options,
        };
        d.walk_frame(0)?;
        Ok(d)
    }

    /// Walks the chunks on until frame `n` (from 0) is stored or the file ends: whether it is.
    pub fn walk_frame(&mut self, n: usize) -> Result<bool, DecodingError> {
        while self.layout.frames.len() <= n && !self.walk.done {
            step(&mut self.layout, &mut self.walk, &mut self.r)?;
        }
        Ok(self.layout.frames.len() > n)
    }

    /// Walks the rest of the chunks and checks the file as [`new`](Self::new) does.
    pub fn finish(&mut self) -> Result<(), DecodingError> {
        while !self.walk.done {
            step(&mut self.layout, &mut self.walk, &mut self.r)?;
        }
        if !self.walk.checked {
            complete(&self.layout, &self.walk, &mut self.r)?;
            self.walk.checked = true;
        }
        Ok(())
    }

    /// Whether [`finish`](Self::finish) has checked the whole file.
    pub fn finished(&self) -> bool {
        self.walk.checked
    }

    /// Sets the maximum amount of memory that the decoder is allowed to allocate at once.
    ///
    /// TODO: Some allocations currently ignore this limit.
    pub fn set_memory_limit(&mut self, limit: usize) {
        self.memory_limit = limit;
    }

    /// The VP8X flags (`WEBP_FF_FORMAT_FLAGS`): [`ALPHA_FLAG`], [`ANIMATION_FLAG`], ...; a
    /// simple file has [`ALPHA_FLAG`] where its bitstream has alpha.
    pub fn format_flags(&self) -> u32 {
        self.layout.feature_flags
    }

    /// The frames in file order: one for a still image.
    pub fn frames(&self) -> &[FrameInfo] {
        &self.layout.frames
    }

    /// The non-image chunks the demuxer keeps, in file order: ICCP, EXIF and XMP where the
    /// VP8X flags name them, and unknown chunks.
    pub fn chunks(&self) -> &[ChunkInfo] {
        &self.layout.chunks
    }

    /// The `n`th (from 1) chunk with `fourcc`, as `WebPDemuxGetChunk`.
    pub fn chunk(&self, fourcc: &[u8; 4], n: usize) -> Option<&ChunkInfo> {
        self.layout.chunks.iter().filter(|c| &c.fourcc == fourcc).nth(n.checked_sub(1)?)
    }

    /// The ANIM chunk's background colour and loop count as libwebp's demuxer reports them
    /// (`WEBP_FF_BACKGROUND_COLOR`, `WEBP_FF_LOOP_COUNT`): `0xffffffff` and 1 without one.
    pub fn animation_params(&self) -> (u32, i32) {
        (self.layout.bgcolor, self.layout.loop_count)
    }

    /// Get the background color specified in the image file if the image is extended and animated webp.
    pub fn background_color_hint(&self) -> Option<[u8; 4]> {
        self.is_animated().then(|| self.layout.bgcolor.to_le_bytes())
    }

    /// Sets the background color if the image is an extended and animated webp.
    pub fn set_background_color(&mut self, color: [u8; 4]) -> Result<(), DecodingError> {
        if self.is_animated() {
            self.background_color = Some(color);
            Ok(())
        } else {
            Err(DecodingError::InvalidParameter(
                "Background color can only be set on animated webp".to_owned(),
            ))
        }
    }

    /// Returns the (width, height) of the image in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.layout.canvas_width as u32, self.layout.canvas_height as u32)
    }

    /// Returns whether the image has an alpha channel. If so, the pixel format is Rgba8 and
    /// otherwise Rgb8.
    pub fn has_alpha(&self) -> bool {
        self.layout.feature_flags & ALPHA_FLAG != 0
    }

    /// Returns true if the image is animated.
    pub fn is_animated(&self) -> bool {
        self.layout.feature_flags & ANIMATION_FLAG != 0
    }

    /// Returns whether the image is lossy. For animated images, this is true if any frame is lossy.
    pub fn is_lossy(&mut self) -> bool {
        self.layout.frames.iter().any(|f| !f.lossless)
    }

    /// Returns the number of frames of a single loop of the animation, or zero if the image is not
    /// animated.
    pub fn num_frames(&self) -> u32 {
        if self.is_animated() {
            self.layout.frames.len() as u32
        } else {
            0
        }
    }

    /// Returns the number of times the animation should loop.
    pub fn loop_count(&self) -> LoopCount {
        match NonZeroU16::new(self.layout.loop_count as u16) {
            Some(n) => LoopCount::Times(n),
            None => LoopCount::Forever,
        }
    }

    /// Returns the total duration of one loop through the animation in milliseconds, or zero if the
    /// image is not animated.
    ///
    /// This is the sum of the durations of all individual frames of the image.
    pub fn loop_duration(&self) -> u64 {
        if !self.is_animated() {
            return 0;
        }
        self.layout.frames.iter().map(|f| f.duration as u64).sum()
    }

    fn read_range(&mut self, start: u64, len: u64) -> Result<Vec<u8>, DecodingError> {
        if len > self.memory_limit as u64 {
            return Err(DecodingError::MemoryLimitExceeded);
        }
        self.r.seek(io::SeekFrom::Start(start))?;
        let mut data = vec![0; len as usize];
        self.r.read_exact(&mut data)?;
        Ok(data)
    }

    fn read_chunk(&mut self, fourcc: &[u8; 4]) -> Result<Option<Vec<u8>>, DecodingError> {
        match self.chunk(fourcc, 1).copied() {
            Some(c) => self.read_range(c.offset, c.size.into()).map(Some),
            None => Ok(None),
        }
    }

    /// Returns the raw bytes of the ICC profile, or None if there is no ICC profile.
    pub fn icc_profile(&mut self) -> Result<Option<Vec<u8>>, DecodingError> {
        self.read_chunk(b"ICCP")
    }

    /// Returns the raw bytes of the EXIF metadata, or None if there is no EXIF metadata.
    pub fn exif_metadata(&mut self) -> Result<Option<Vec<u8>>, DecodingError> {
        self.read_chunk(b"EXIF")
    }

    /// Returns the raw bytes of the XMP metadata, or None if there is no XMP metadata.
    pub fn xmp_metadata(&mut self) -> Result<Option<Vec<u8>>, DecodingError> {
        self.read_chunk(b"XMP ")
    }

    /// Returns the number of bytes required to store the image or a single frame, or None if that
    /// would take more than `usize::MAX` bytes.
    pub fn output_buffer_size(&self) -> Option<usize> {
        let (width, height) = self.dimensions();
        let bytes_per_pixel = if self.has_alpha() { 4 } else { 3 };
        (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(bytes_per_pixel)
    }

}

impl<R: BufRead + Seek + Send> WebPDecoder<R> {
    /// A decoder for frame `n` (from 0) to RGBA (`alpha`) or RGB, scaled to `scaled` where
    /// given, as libwebp's `WebPDecode` decodes the bytes the demuxer hands out for it: its
    /// ALPH chunk, if any, through its image chunk. It reads them through this decoder's reader.
    pub fn frame(&mut self, n: usize, scaled: Option<(u32, u32)>, alpha: bool) -> Result<FrameDecoder<'_>, DecodingError> {
        if !self.walk_frame(n)? {
            return Err(DecodingError::NoMoreFrames);
        }
        let f = &self.layout.frames[n];
        let start = if f.alpha.size > 0 { f.alpha.offset } else { f.image.offset };
        let len = f.image.offset + f.image.size - start;
        let headers = crate::headers::read_decoding(&mut self.r, start, len).map_err(|_| DecodingError::InvalidContainer)?;
        FrameDecoder::read(&mut self.r, start, len, &headers, scaled, alpha, self.webp_decode_options.lossy_upsampling)
    }

    fn frame_decoder(&mut self, n: usize, alpha: bool) -> Result<FrameDecoder<'_>, DecodingError> {
        self.frame(n, None, alpha)
    }

    /// Returns the raw bytes of the image. For animated images, this is the first frame.
    ///
    /// Fails with `ImageTooLarge` if `buf` has length different than `output_buffer_size()`
    pub fn read_image(&mut self, buf: &mut [u8]) -> Result<(), DecodingError> {
        if Some(buf.len()) != self.output_buffer_size() {
            return Err(DecodingError::ImageTooLarge);
        }
        if self.is_animated() {
            let saved = std::mem::take(&mut self.animation);
            let result = self.read_frame(buf);
            self.animation = saved;
            result?;
            return Ok(());
        }
        let alpha = self.has_alpha();
        self.frame_decoder(0, alpha)?.read_into(buf)
    }

    /// Decodes the image a band of rows at a time, top to bottom, calling `f(first_row, rows)`
    /// with whole rows in the layout of [`read_image`](Self::read_image). For animated images
    /// this is the first frame. See [`band_reader`](Self::band_reader).
    pub fn read_image_rows(&mut self, mut f: impl FnMut(u32, &[u8])) -> Result<(), DecodingError> {
        let mut bands = self.band_reader()?;
        while let Some((y, rows)) = bands.next_band()? {
            f(y, rows);
        }
        Ok(())
    }

    /// Returns a reader that decodes the image a band of rows at a time, top to bottom, each
    /// time [`BandReader::next_band`] is called. For animated images this is the first frame.
    ///
    /// A lossy image is decoded one macroblock row (16 rows) at a time, so only a band of rows
    /// and the alpha plane are held instead of the whole image; a lossless image holds the
    /// pixels a backward reference can reach; animations are decoded into the output layout.
    pub fn band_reader(&mut self) -> Result<BandReader<'_>, DecodingError> {
        let (width, height) = self.dimensions();
        let bpp = if self.has_alpha() { 4 } else { 3 };
        let source = if self.is_animated() {
            let size = self
                .output_buffer_size()
                .ok_or(DecodingError::ImageTooLarge)?;
            let mut pixels = vec![0; size];
            self.read_image(&mut pixels)?;
            BandSource::Buffered { pixels, next_row: 0 }
        } else {
            BandSource::Frame(self.frame_decoder(0, bpp == 4)?)
        };
        Ok(BandReader {
            width,
            height,
            bpp,
            source,
        })
    }

    /// Reads the next frame of the animation.
    ///
    /// The frame contents are written into `buf` and the method returns the duration of the frame
    /// in milliseconds. If there are no more frames, the method returns
    /// `DecodingError::NoMoreFrames` and `buf` is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the image is not animated.
    pub fn read_frame(&mut self, buf: &mut [u8]) -> Result<u32, DecodingError> {
        assert!(self.is_animated());
        assert_eq!(Some(buf.len()), self.output_buffer_size());

        let n = self.animation.next_frame as usize;
        let Some(f) = self.layout.frames.get(n).cloned() else {
            return Err(DecodingError::NoMoreFrames);
        };
        let (width, height) = self.dimensions();
        let frame_has_alpha = f.alpha.size > 0 || f.lossless;
        let mut frame = Vec::new();
        let mut decoder = self.frame_decoder(n, frame_has_alpha)?;
        while let Some((_, rows)) = decoder.next_band()? {
            frame.extend_from_slice(rows);
        }
        drop(decoder);
        let (frame_x, frame_y) = (f.x_offset as u32, f.y_offset as u32);
        let (frame_width, frame_height) = (f.width as u32, f.height as u32);

        // A frame disposes to transparent whatever the next frame carries, as libwebp's
        // animation decoder disposes it.
        let clear_color = match self.animation.dispose_next_frame {
            true => Some(self.background_color.unwrap_or([0, 0, 0, 0])),
            false => None,
        };

        // fill starting canvas with clear color
        if self.animation.canvas.is_none() {
            let mut canvas = vec![0; (width * height * 4) as usize];
            if let Some(color) = self.background_color.as_ref() {
                canvas.chunks_exact_mut(4).for_each(|c| c.copy_from_slice(color))
            }
            self.animation.canvas = Some(canvas);
        }
        extended::composite_frame(
            self.animation.canvas.as_mut().unwrap(),
            width,
            height,
            clear_color,
            &frame,
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            frame_has_alpha,
            !f.no_blend,
            self.animation.previous_frame_width,
            self.animation.previous_frame_height,
            self.animation.previous_frame_x_offset,
            self.animation.previous_frame_y_offset,
        );

        self.animation.previous_frame_width = frame_width;
        self.animation.previous_frame_height = frame_height;
        self.animation.previous_frame_x_offset = frame_x;
        self.animation.previous_frame_y_offset = frame_y;
        self.animation.dispose_next_frame = f.dispose_to_background;
        self.animation.next_frame += 1;

        let canvas = self.animation.canvas.as_ref().unwrap();
        if self.has_alpha() {
            buf.copy_from_slice(canvas);
        } else {
            for (b, c) in buf.chunks_exact_mut(3).zip(canvas.chunks_exact(4)) {
                b.copy_from_slice(&c[..3]);
            }
        }
        Ok(f.duration as u32)
    }

}

impl<R: BufRead + Seek> WebPDecoder<R> {
    /// The reader.
    pub fn get_ref(&self) -> &R {
        &self.r
    }

    /// The reader, to decode the data some other way.
    pub fn into_inner(self) -> R {
        self.r
    }

    /// Resets the animation to the first frame.
    ///
    /// # Panics
    ///
    /// Panics if the image is not animated.
    pub fn reset_animation(&mut self) {
        assert!(self.is_animated());
        self.animation.next_frame = 0;
        self.animation.dispose_next_frame = true;
    }

    /// Sets the upsampling method that is used in lossy decoding
    pub fn set_lossy_upsampling(&mut self, upsampling_method: UpsamplingMethod) {
        self.webp_decode_options.lossy_upsampling = upsampling_method;
    }
}

const BUFFERED_BAND_ROWS: usize = 16;

/// Decodes an image a band of rows at a time, see [`WebPDecoder::band_reader`].
pub struct BandReader<'a> {
    width: u32,
    height: u32,
    bpp: usize,
    source: BandSource<'a>,
}

enum BandSource<'a> {
    Frame(FrameDecoder<'a>),
    Buffered { pixels: Vec<u8>, next_row: usize },
}

impl BandReader<'_> {
    /// Decodes the next band and returns its first row and its whole rows in the layout of
    /// [`WebPDecoder::read_image`], or `None` after the last band.
    pub fn next_band(&mut self) -> Result<Option<(u32, &[u8])>, DecodingError> {
        match &mut self.source {
            BandSource::Frame(d) => d.next_band(),
            BandSource::Buffered { pixels, next_row } => {
                let stride = self.width as usize * self.bpp;
                let y = *next_row;
                if y >= self.height as usize {
                    return Ok(None);
                }
                let rows = BUFFERED_BAND_ROWS.min(self.height as usize - y);
                *next_row += rows;
                Ok(Some((y as u32, &pixels[y * stride..(y + rows) * stride])))
            }
        }
    }

    /// Returns the (width, height) of the image in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the bytes per pixel of the bands: 4 (RGBA) if the image has alpha, 3 (RGB)
    /// otherwise.
    pub fn bytes_per_pixel(&self) -> usize {
        self.bpp
    }

    /// Returns the most rows a band has.
    pub fn max_band_rows(&self) -> u32 {
        match &self.source {
            BandSource::Frame(d) => d.max_band_rows(),
            BandSource::Buffered { .. } => (BUFFERED_BAND_ROWS as u32).min(self.height),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const RGB_BPP: usize = 3;

    #[test]
    fn add_with_overflow_size() {
        let bytes = vec![
            0x52, 0x49, 0x46, 0x46, 0xaf, 0x37, 0x80, 0x47, 0x57, 0x45, 0x42, 0x50, 0x6c, 0x64,
            0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xfb, 0x7e, 0x73, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65,
            0x40, 0xfb, 0xff, 0xff, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65,
            0x00, 0x00, 0x00, 0x00, 0x62, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49,
            0x49, 0x54, 0x55, 0x50, 0x4c, 0x54, 0x59, 0x50, 0x45, 0x33, 0x37, 0x44, 0x4d, 0x46,
        ];

        let data = std::io::Cursor::new(bytes);

        let _ = WebPDecoder::new(data);
    }

    #[test]
    fn decode_2x2_single_color_image() {
        // Image data created from imagemagick and output of xxd:
        // $ convert -size 2x2 xc:#f00 red.webp
        // $ xxd -g 1 red.webp | head

        const NUM_PIXELS: usize = 2 * 2 * RGB_BPP;
        // 2x2 red pixel image
        let bytes = [
            0x52, 0x49, 0x46, 0x46, 0x3c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50,
            0x38, 0x20, 0x30, 0x00, 0x00, 0x00, 0xd0, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x02, 0x00,
            0x02, 0x00, 0x02, 0x00, 0x34, 0x25, 0xa0, 0x02, 0x74, 0xba, 0x01, 0xf8, 0x00, 0x03,
            0xb0, 0x00, 0xfe, 0xf0, 0xc4, 0x0b, 0xff, 0x20, 0xb9, 0x61, 0x75, 0xc8, 0xd7, 0xff,
            0x20, 0x3f, 0xe4, 0x07, 0xfc, 0x80, 0xff, 0xf8, 0xf2, 0x00, 0x00, 0x00,
        ];

        let mut data = [0; NUM_PIXELS];
        let mut decoder = WebPDecoder::new(std::io::Cursor::new(bytes)).unwrap();
        decoder.read_image(&mut data).unwrap();

        // All pixels are the same value
        let first_pixel = &data[..RGB_BPP];
        assert!(data.chunks_exact(3).all(|ch| ch.iter().eq(first_pixel)));
    }

    #[test]
    fn decode_3x3_single_color_image() {
        // Test that any odd pixel "tail" is decoded properly

        const NUM_PIXELS: usize = 3 * 3 * RGB_BPP;
        // 3x3 red pixel image
        let bytes = [
            0x52, 0x49, 0x46, 0x46, 0x3c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50,
            0x38, 0x20, 0x30, 0x00, 0x00, 0x00, 0xd0, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x03, 0x00,
            0x03, 0x00, 0x02, 0x00, 0x34, 0x25, 0xa0, 0x02, 0x74, 0xba, 0x01, 0xf8, 0x00, 0x03,
            0xb0, 0x00, 0xfe, 0xf0, 0xc4, 0x0b, 0xff, 0x20, 0xb9, 0x61, 0x75, 0xc8, 0xd7, 0xff,
            0x20, 0x3f, 0xe4, 0x07, 0xfc, 0x80, 0xff, 0xf8, 0xf2, 0x00, 0x00, 0x00,
        ];

        let mut data = [0; NUM_PIXELS];
        let mut decoder = WebPDecoder::new(std::io::Cursor::new(bytes)).unwrap();
        decoder.read_image(&mut data).unwrap();

        // All pixels are the same value
        let first_pixel = &data[..RGB_BPP];
        assert!(data.chunks_exact(3).all(|ch| ch.iter().eq(first_pixel)));
    }
}

// The chunk walk: libwebp's demuxer.

/// VP8X flags.
pub const ANIMATION_FLAG: u32 = 0x02;
/// VP8X flags.
pub const XMP_FLAG: u32 = 0x04;
/// VP8X flags.
pub const EXIF_FLAG: u32 = 0x08;
/// VP8X flags.
pub const ALPHA_FLAG: u32 = 0x10;
/// VP8X flags.
pub const ICCP_FLAG: u32 = 0x20;
const ALL_VALID_FLAGS: u32 = ALPHA_FLAG | ANIMATION_FLAG | EXIF_FLAG | ICCP_FLAG | XMP_FLAG;

const CHUNK_HEADER_SIZE: u64 = 8;
const RIFF_HEADER_SIZE: u64 = 12;
const ANMF_CHUNK_SIZE: u32 = 16;
const ANIM_CHUNK_SIZE: u32 = 6;
const VP8X_CHUNK_SIZE: u32 = 10;
const MAX_CHUNK_PAYLOAD: u32 = !0u32 - 8 - 1;
const MAX_IMAGE_AREA: u64 = 1 << 32;

type WalkResult<T> = std::result::Result<T, DecodingError>;

/// Offset and size of a chunk, header included.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkData {
    /// Offset of the chunk header in the file.
    pub offset: u64,
    /// Header plus (padded, for image chunks) payload.
    pub size: u64,
}

/// A frame as the demuxer stores it.
#[derive(Clone, Debug, Default)]
pub struct FrameInfo {
    /// ANMF offsets.
    pub x_offset: i32,
    /// ANMF offsets.
    pub y_offset: i32,
    /// The bitstream's width.
    pub width: i32,
    /// The bitstream's height.
    pub height: i32,
    /// ALPH chunk or VP8L alpha bit.
    pub has_alpha: bool,
    /// ANMF duration.
    pub duration: i32,
    /// ANMF `WEBP_MUX_DISPOSE_BACKGROUND`.
    pub dispose_to_background: bool,
    /// ANMF `WEBP_MUX_NO_BLEND`.
    pub no_blend: bool,
    /// 1-based frame number.
    pub frame_num: i32,
    /// Whether an image bitstream was found.
    pub complete: bool,
    /// A VP8L bitstream.
    pub lossless: bool,
    /// The VP8/VP8L chunk.
    pub image: ChunkData,
    /// The ALPH chunk.
    pub alpha: ChunkData,
}

/// A stored non-image chunk.
#[derive(Clone, Copy, Debug)]
pub struct ChunkInfo {
    /// Its fourcc.
    pub fourcc: [u8; 4],
    /// Offset of the payload.
    pub offset: u64,
    /// Unpadded payload size.
    pub size: u32,
}

/// What libwebp's demuxer finds in a valid file.
#[derive(Clone, Debug, Default)]
struct Layout {
    /// `WEBP_FF_FORMAT_FLAGS`.
    feature_flags: u32,
    /// `WEBP_FF_CANVAS_WIDTH`.
    canvas_width: i32,
    /// `WEBP_FF_CANVAS_HEIGHT`.
    canvas_height: i32,
    /// `WEBP_FF_LOOP_COUNT`.
    loop_count: i32,
    /// `WEBP_FF_BACKGROUND_COLOR`.
    bgcolor: u32,
    /// The frames, numbered 1.. in order.
    frames: Vec<FrameInfo>,
    /// Non-image chunks, in file order.
    chunks: Vec<ChunkInfo>,
    is_ext_format: bool,
}

struct Mem<'r, R> {
    r: &'r mut R,
    start: u64,
    riff_end: u64,
}

impl<R: Read + Seek> Mem<'_, R> {
    fn left(&self) -> u64 {
        self.riff_end - self.start
    }

    fn read_at<const N: usize>(&mut self, at: u64) -> WalkResult<[u8; N]> {
        let mut b = [0u8; N];
        let eof = |e: io::Error| match e.kind() {
            io::ErrorKind::UnexpectedEof => DecodingError::InvalidContainer,
            _ => DecodingError::IoError(e),
        };
        self.r.seek(SeekFrom::Start(at)).map_err(eof)?;
        self.r.read_exact(&mut b).map_err(eof)?;
        Ok(b)
    }

    fn take<const N: usize>(&mut self) -> WalkResult<[u8; N]> {
        let b = self.read_at(self.start)?;
        self.start += N as u64;
        Ok(b)
    }

    fn chunk_header(&mut self) -> WalkResult<([u8; 4], u32)> {
        let h: [u8; 8] = self.take()?;
        Ok(([h[0], h[1], h[2], h[3]], u32::from_le_bytes([h[4], h[5], h[6], h[7]])))
    }
}

fn le24(b: &[u8]) -> u32 {
    u32::from(b[0]) | u32::from(b[1]) << 8 | u32::from(b[2]) << 16
}

fn fail<T>() -> WalkResult<T> {
    Err(DecodingError::InvalidContainer)
}

/// Where the walk of the chunks stands.
#[derive(Clone, Copy, Debug, Default)]
struct Walk {
    start: u64,
    riff_end: u64,
    is_ext_format: bool,
    anim_chunks: u32,
    done: bool,
    checked: bool,
}

/// `WebPDemux` as far as the RIFF and VP8X headers: the layout they give and where the chunks
/// start. libwebp falls back to a raw VP8/VP8L frame where the RIFF header fails: not ported,
/// as data starting "RIFF" never is one.
fn begin<R: Read + Seek>(r: &mut R) -> WalkResult<(Layout, Walk)> {
    let mut mem = Mem { r, start: 0, riff_end: 0 };
    let head: [u8; 20] = mem.read_at(0)?;
    if &head[0..4] != b"RIFF" || &head[8..12] != b"WEBP" {
        return fail();
    }
    let riff_size = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
    if riff_size < CHUNK_HEADER_SIZE as u32 || riff_size > MAX_CHUNK_PAYLOAD {
        return fail();
    }
    mem.riff_end = u64::from(riff_size) + CHUNK_HEADER_SIZE;
    mem.start = RIFF_HEADER_SIZE;

    let mut d = Layout { loop_count: 1, bgcolor: 0xFFFF_FFFF, canvas_width: -1, canvas_height: -1, ..Default::default() };
    match &head[12..16] {
        b"VP8 " | b"VP8L" => {}
        b"VP8X" => parse_vp8x(&mut d, &mut mem)?,
        _ => return fail(),
    }
    let walk = Walk { start: mem.start, riff_end: mem.riff_end, is_ext_format: d.is_ext_format, ..Default::default() };
    Ok((d, walk))
}

/// One chunk of the walk: the single image of a simple file, or a chunk of an extended one.
fn step<R: Read + Seek>(d: &mut Layout, w: &mut Walk, r: &mut R) -> WalkResult<()> {
    let mut mem = Mem { r, start: w.start, riff_end: w.riff_end };
    let done = if w.is_ext_format {
        parse_vp8x_chunk(d, &mut mem, &mut w.anim_chunks)
    } else {
        parse_single_image(d, &mut mem).map(|()| true)
    };
    w.start = mem.start;
    w.done = done?;
    Ok(())
}

/// The checks of the whole file, once walked: partial data is refused, and a file holds a frame.
fn complete<R: Read + Seek>(d: &Layout, w: &Walk, r: &mut R) -> WalkResult<()> {
    let mut mem = Mem { r, start: w.start, riff_end: w.riff_end };
    let _: [u8; 1] = mem.read_at(mem.riff_end - 1)?;
    if d.frames.is_empty() || d.canvas_width <= 0 || d.canvas_height <= 0 || d.loop_count < 0 {
        return fail();
    }
    Ok(())
}

/// A frame as the demuxer stores it, checked as `IsValidSimpleFormat` and
/// `IsValidExtendedFormat` check it once the file is walked.
fn add_frame(d: &mut Layout, f: FrameInfo) -> WalkResult<()> {
    if d.frames.last().is_some_and(|f| !f.complete) {
        return fail();
    }
    if !d.is_ext_format {
        if !(d.canvas_width > 0 && d.canvas_height > 0 && f.width > 0 && f.height > 0) {
            return fail();
        }
        d.frames.push(f);
        return Ok(());
    }
    let is_animation = d.feature_flags & ANIMATION_FLAG != 0;
    if !is_animation && f.frame_num > 1 {
        return fail();
    }
    // A partial frame never passes in complete data.
    if !f.complete {
        return fail();
    }
    if f.alpha.size == 0 && f.image.size == 0 {
        return fail();
    }
    if f.alpha.size > 0 && f.alpha.offset > f.image.offset {
        return fail();
    }
    if f.width <= 0 || f.height <= 0 {
        return fail();
    }
    if !check_frame_bounds(&f, !is_animation, d.canvas_width, d.canvas_height) {
        return fail();
    }
    d.frames.push(f);
    Ok(())
}

struct Features {
    width: i32,
    height: i32,
    has_alpha: bool,
}

/// `WebPGetFeatures` over a VP8/VP8L chunk with `payload` bytes of its payload present.
fn get_features<R: Read + Seek>(mem: &mut Mem<R>, at: u64, lossless: bool, payload_size: u32, payload: u64) -> WalkResult<Features> {
    if payload < 4 {
        return fail();
    }
    if lossless {
        if payload < 5 {
            return fail();
        }
        let b: [u8; 5] = mem.read_at(at + CHUNK_HEADER_SIZE)?;
        if b[0] != 0x2f || b[4] >> 5 != 0 {
            return fail();
        }
        let bits = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], 0, 0, 0]);
        Ok(Features {
            width: ((bits >> 8) & 0x3fff) as i32 + 1,
            height: ((bits >> 22) & 0x3fff) as i32 + 1,
            has_alpha: (bits >> 36) & 1 != 0,
        })
    } else {
        if payload < 10 {
            return fail();
        }
        let b: [u8; 10] = mem.read_at(at + CHUNK_HEADER_SIZE)?;
        if b[3..6] != [0x9d, 0x01, 0x2a] {
            return fail();
        }
        let bits = le24(&b);
        let w = (u32::from(b[7]) << 8 | u32::from(b[6])) & 0x3fff;
        let h = (u32::from(b[9]) << 8 | u32::from(b[8])) & 0x3fff;
        if bits & 1 != 0 || (bits >> 1) & 7 > 3 || (bits >> 4) & 1 == 0 || bits >> 5 >= payload_size || w == 0 || h == 0 {
            return fail();
        }
        Ok(Features { width: w as i32, height: h as i32, has_alpha: false })
    }
}

fn store_frame<R: Read + Seek>(frame_num: i32, min_size: u32, mem: &mut Mem<R>, frame: &mut FrameInfo) -> WalkResult<()> {
    let mut alpha_chunks = 0;
    let mut image_chunks = 0;
    if mem.left() < CHUNK_HEADER_SIZE || mem.left() < u64::from(min_size) {
        return fail();
    }
    loop {
        let chunk_start = mem.start;
        let (fourcc, payload_size) = mem.chunk_header()?;
        if payload_size > MAX_CHUNK_PAYLOAD {
            return fail();
        }
        let padded = u64::from(payload_size + (payload_size & 1));
        if padded > mem.left() {
            return fail();
        }
        let chunk_size = CHUNK_HEADER_SIZE + padded;
        let mut done = false;
        match &fourcc {
            b"ALPH" if alpha_chunks == 0 => {
                alpha_chunks += 1;
                frame.alpha = ChunkData { offset: chunk_start, size: chunk_size };
                frame.has_alpha = true;
                frame.frame_num = frame_num;
                mem.start += padded;
            }
            b"VP8L" if alpha_chunks > 0 => return fail(),
            b"VP8 " | b"VP8L" if image_chunks == 0 => {
                let features = get_features(mem, chunk_start, &fourcc == b"VP8L", payload_size, padded)?;
                image_chunks += 1;
                frame.image = ChunkData { offset: chunk_start, size: chunk_size };
                frame.width = features.width;
                frame.height = features.height;
                frame.has_alpha |= features.has_alpha;
                frame.frame_num = frame_num;
                frame.complete = true;
                frame.lossless = &fourcc == b"VP8L";
                mem.start += padded;
            }
            _ => {
                mem.start -= CHUNK_HEADER_SIZE;
                done = true;
            }
        }
        if mem.start == mem.riff_end {
            done = true;
        } else if mem.left() < CHUNK_HEADER_SIZE {
            return fail();
        }
        if done {
            return Ok(());
        }
    }
}

fn parse_animation_frame<R: Read + Seek>(d: &mut Layout, mem: &mut Mem<R>, frame_chunk_size: u32) -> WalkResult<()> {
    let is_animation = d.feature_flags & ANIMATION_FLAG != 0;
    if u64::from(ANMF_CHUNK_SIZE) > mem.left() || frame_chunk_size < ANMF_CHUNK_SIZE {
        return fail();
    }
    let anmf_payload_size = frame_chunk_size - ANMF_CHUNK_SIZE;
    let b: [u8; 16] = mem.take()?;
    let mut frame = FrameInfo {
        x_offset: 2 * le24(&b[0..]) as i32,
        y_offset: 2 * le24(&b[3..]) as i32,
        width: 1 + le24(&b[6..]) as i32,
        height: 1 + le24(&b[9..]) as i32,
        duration: le24(&b[12..]) as i32,
        dispose_to_background: b[15] & 1 != 0,
        no_blend: b[15] & 2 != 0,
        ..Default::default()
    };
    if frame.width as u64 * frame.height as u64 >= MAX_IMAGE_AREA {
        return fail();
    }
    let start_offset = mem.start;
    store_frame(d.frames.len() as i32 + 1, anmf_payload_size, mem, &mut frame)?;
    if mem.start - start_offset > u64::from(anmf_payload_size) {
        return fail();
    }
    if is_animation && frame.frame_num > 0 {
        add_frame(d, frame)?;
    }
    Ok(())
}

fn parse_single_image<R: Read + Seek>(d: &mut Layout, mem: &mut Mem<R>) -> WalkResult<()> {
    if !d.frames.is_empty() || mem.left() < CHUNK_HEADER_SIZE {
        return fail();
    }
    let mut frame = FrameInfo::default();
    store_frame(1, 0, mem, &mut frame)?;
    if d.feature_flags & ALPHA_FLAG == 0 && frame.alpha.size > 0 {
        frame.alpha = ChunkData::default();
        frame.has_alpha = false;
    }
    if !d.is_ext_format && frame.width > 0 && frame.height > 0 {
        d.canvas_width = frame.width;
        d.canvas_height = frame.height;
        if frame.has_alpha {
            d.feature_flags |= ALPHA_FLAG;
        }
    }
    add_frame(d, frame)
}

/// One chunk of an extended file: whether the walk is at the RIFF end.
fn parse_vp8x_chunk<R: Read + Seek>(d: &mut Layout, mem: &mut Mem<R>, anim_chunks: &mut u32) -> WalkResult<bool> {
    let is_animation = d.feature_flags & ANIMATION_FLAG != 0;
    {
        let chunk_start = mem.start;
        let (fourcc, chunk_size) = mem.chunk_header()?;
        if chunk_size > MAX_CHUNK_PAYLOAD {
            return fail();
        }
        let padded = chunk_size + (chunk_size & 1);
        if u64::from(padded) > mem.left() {
            return fail();
        }
        let store = match &fourcc {
            b"VP8X" => return fail(),
            b"ALPH" | b"VP8 " | b"VP8L" => {
                if *anim_chunks > 0 || is_animation {
                    return fail();
                }
                mem.start -= CHUNK_HEADER_SIZE;
                parse_single_image(d, mem)?;
                None
            }
            b"ANIM" => {
                if padded < ANIM_CHUNK_SIZE {
                    return fail();
                }
                if *anim_chunks == 0 {
                    *anim_chunks += 1;
                    let b: [u8; 6] = mem.take()?;
                    d.bgcolor = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    d.loop_count = i32::from(u16::from_le_bytes([b[4], b[5]]));
                    mem.start += u64::from(padded - ANIM_CHUNK_SIZE);
                    None
                } else {
                    Some(false)
                }
            }
            b"ANMF" => {
                if *anim_chunks == 0 {
                    return fail();
                }
                parse_animation_frame(d, mem, padded)?;
                None
            }
            b"ICCP" => Some(d.feature_flags & ICCP_FLAG != 0),
            b"EXIF" => Some(d.feature_flags & EXIF_FLAG != 0),
            b"XMP " => Some(d.feature_flags & XMP_FLAG != 0),
            _ => Some(true),
        };
        if let Some(store) = store {
            if store {
                d.chunks.push(ChunkInfo { fourcc, offset: chunk_start + CHUNK_HEADER_SIZE, size: chunk_size });
            }
            mem.start += u64::from(padded);
        }
        if mem.start == mem.riff_end {
            return Ok(true);
        } else if mem.left() < CHUNK_HEADER_SIZE {
            return fail();
        }
    }
    Ok(false)
}

fn parse_vp8x<R: Read + Seek>(d: &mut Layout, mem: &mut Mem<R>) -> WalkResult<()> {
    if mem.left() < CHUNK_HEADER_SIZE {
        return fail();
    }
    d.is_ext_format = true;
    let (_, mut vp8x_size) = mem.chunk_header()?;
    if !(VP8X_CHUNK_SIZE..=MAX_CHUNK_PAYLOAD).contains(&vp8x_size) {
        return fail();
    }
    vp8x_size += vp8x_size & 1;
    if u64::from(vp8x_size) > mem.left() {
        return fail();
    }
    let b: [u8; 10] = mem.read_at(mem.start)?;
    d.feature_flags = u32::from(b[0]);
    d.canvas_width = 1 + le24(&b[4..]) as i32;
    d.canvas_height = 1 + le24(&b[7..]) as i32;
    if d.canvas_width as u64 * d.canvas_height as u64 >= MAX_IMAGE_AREA {
        return fail();
    }
    mem.start += u64::from(vp8x_size);
    if mem.left() < CHUNK_HEADER_SIZE {
        return fail();
    }
    if d.feature_flags & !ALL_VALID_FLAGS != 0 {
        return fail();
    }
    Ok(())
}

fn check_frame_bounds(f: &FrameInfo, exact: bool, canvas_width: i32, canvas_height: i32) -> bool {
    if exact {
        f.x_offset == 0 && f.y_offset == 0 && f.width == canvas_width && f.height == canvas_height
    } else {
        f.x_offset >= 0 && f.y_offset >= 0 && f.width + f.x_offset <= canvas_width && f.height + f.y_offset <= canvas_height
    }
}

