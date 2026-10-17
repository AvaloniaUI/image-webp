//! Encoding of WebP images.
use std::io::{self, Write};

use quick_error::quick_error;

use crate::{
    lossless::{vp8l, EncoderParams},
    lossy::vp8enc::encode_frame_lossy,
};

/// Color type of the image.
///
/// Note that the WebP format doesn't have a concept of color type. All images are encoded as RGBA
/// and some decoders may treat them as such. This enum is used to indicate the color type of the
/// input data provided to the encoder, which can help improve compression ratio.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColorType {
    /// Opaque image with a single luminance byte per pixel.
    L8,
    /// Image with a luminance and alpha byte per pixel.
    La8,
    /// Opaque image with a red, green, and blue byte per pixel.
    Rgb8,
    /// Image with a red, green, blue, and alpha byte per pixel.
    Rgba8,
}

impl ColorType {
    fn bytes_per_pixel(self) -> usize {
        match self {
            ColorType::L8 => 1,
            ColorType::La8 => 2,
            ColorType::Rgb8 => 3,
            ColorType::Rgba8 => 4,
        }
    }
}

quick_error! {
    /// Error that can occur during encoding.
    #[derive(Debug)]
    #[non_exhaustive]
    pub enum EncodingError {
        /// An IO error occurred.
        IoError(err: io::Error) {
            from()
            display("IO error: {}", err)
            source(err)
        }

        /// The image dimensions are not allowed by the WebP format.
        InvalidDimensions {
            display("Invalid dimensions")
        }

        /// The lossy frame's first partition does not fit its 19-bit size field, even without
        /// intra 4x4 modes.
        Partition0Overflow {
            display("First partition over 512 KiB")
        }
    }
}

pub(crate) const fn chunk_size(inner_bytes: usize) -> u32 {
    if inner_bytes % 2 == 1 {
        (inner_bytes + 1) as u32 + 8
    } else {
        inner_bytes as u32 + 8
    }
}

pub(crate) fn write_chunk<W: Write>(w: W, name: &[u8], data: &[u8]) -> io::Result<()> {
    write_chunk_parts(w, name, &[data])
}

pub(crate) fn write_chunk_parts<W: Write>(mut w: W, name: &[u8], parts: &[&[u8]]) -> io::Result<()> {
    debug_assert!(name.len() == 4);
    let len: usize = parts.iter().map(|p| p.len()).sum();

    w.write_all(name)?;
    w.write_all(&(len as u32).to_le_bytes())?;
    for part in parts {
        w.write_all(part)?;
    }
    if len % 2 == 1 {
        w.write_all(&[0])?;
    }
    Ok(())
}

/// WebP Encoder.
pub struct WebPEncoder<W> {
    writer: W,
    icc_profile: Vec<u8>,
    exif_metadata: Vec<u8>,
    xmp_metadata: Vec<u8>,
    params: EncoderParams,
}

impl<W: Write> WebPEncoder<W> {
    /// Create a new encoder that writes its output to `w`.
    ///
    /// Only supports "VP8L" lossless encoding.
    pub fn new(w: W) -> Self {
        Self {
            writer: w,
            icc_profile: Vec::new(),
            exif_metadata: Vec::new(),
            xmp_metadata: Vec::new(),
            params: EncoderParams::default(),
        }
    }

    /// Set the ICC profile to use for the image.
    pub fn set_icc_profile(&mut self, icc_profile: Vec<u8>) {
        self.icc_profile = icc_profile;
    }

    /// Set the EXIF metadata to use for the image.
    pub fn set_exif_metadata(&mut self, exif_metadata: Vec<u8>) {
        self.exif_metadata = exif_metadata;
    }

    /// Set the XMP metadata to use for the image.
    pub fn set_xmp_metadata(&mut self, xmp_metadata: Vec<u8>) {
        self.xmp_metadata = xmp_metadata;
    }

    /// Set the `EncoderParams` to use.
    pub fn set_params(&mut self, params: EncoderParams) {
        self.params = params;
    }

    /// Encode image data with the indicated color type.
    ///
    /// # Panics
    ///
    /// Panics if the image data is not of the indicated dimensions.
    pub fn encode(
        self,
        data: &[u8],
        width: u32,
        height: u32,
        color: ColorType,
    ) -> Result<(), EncodingError> {
        if self.params.use_lossy {
            let stride = width as usize * color.bytes_per_pixel();
            return self.encode_rows(width, height, color, |y, row| {
                row.copy_from_slice(&data[y as usize * stride..][..stride]);
            });
        }
        if width == 0 || width > 16384 || height == 0 || height > 16384 {
            return Err(EncodingError::InvalidDimensions);
        }
        let config = vp8l::Config {
            quality: f32::from(self.params.lossless_quality.min(100)),
            method: i32::from(self.params.lossless_method.min(6)),
            exact: self.params.lossless_exact,
            use_predictor: self.params.use_predictor_transform,
            near_lossless: i32::from(self.params.near_lossless.min(100)),
        };
        let frame = vp8l::encode_lossless(data, width, height, color, config);
        let has_alpha = frame[4] & 0x10 != 0;
        self.write_container(b"VP8L", vec![frame], None, has_alpha, width, height)
    }

    /// Encodes an image whose rows of `color` pixels `read_row(y, row)` writes into `row`, so
    /// that the whole image never has to be in memory. Lossy encoding reads the rows top to
    /// bottom, once for each of its two or three passes, and holds about 16 rows at a time;
    /// lossless encoding reads each row once and keeps the whole image.
    pub fn encode_rows(
        self,
        width: u32,
        height: u32,
        color: ColorType,
        mut read_row: impl FnMut(u32, &mut [u8]),
    ) -> Result<(), EncodingError> {
        if !self.params.use_lossy {
            let stride = width as usize * color.bytes_per_pixel();
            let mut data = vec![0; stride * height as usize];
            for (y, row) in data.chunks_exact_mut(stride).enumerate() {
                read_row(y as u32, row);
            }
            return self.encode(&data, width, height, color);
        }
        let params = self.params.clone();
        self.encode_lossy(width, height, color, read_row, true, |alpha| {
            alpha.map(|alpha| alpha_chunk_payload(&alpha, width, height, &params))
        })
    }

    /// Like [`encode_rows`](Self::encode_rows), but a lossy image with alpha gets its `ALPH`
    /// chunk from `alpha_chunk()` instead of encoding the alpha plane: the caller encodes the
    /// plane with [`encode_alpha_chunk`] and the same params, for example on a thread of its own
    /// while this encodes the `VP8 ` data, and `alpha_chunk` returns it (or joins that thread).
    /// It is called once, after the `VP8 ` data is encoded, unless encoding is lossless.
    ///
    /// The rows still have to carry their alpha (`Rgba8` or `La8`): the encoder flattens fully
    /// transparent areas before encoding them, as libwebp does, and writes no `ALPH` chunk when
    /// every pixel is opaque.
    pub fn encode_rows_with_alpha_chunk(
        self,
        width: u32,
        height: u32,
        color: ColorType,
        read_row: impl FnMut(u32, &mut [u8]),
        alpha_chunk: impl FnOnce() -> Option<Vec<u8>>,
    ) -> Result<(), EncodingError> {
        if self.params.use_lossy {
            self.encode_lossy(width, height, color, read_row, false, |_| alpha_chunk())
        } else {
            self.encode_rows(width, height, color, read_row)
        }
    }

    fn encode_lossy(
        self,
        width: u32,
        height: u32,
        color: ColorType,
        read_row: impl FnMut(u32, &mut [u8]),
        keep_alpha: bool,
        alpha_chunk: impl FnOnce(Option<Vec<u8>>) -> Option<Vec<u8>>,
    ) -> Result<(), EncodingError> {
        let (parts, (has_alpha, alpha_plane)) = encode_frame_lossy(
            width,
            height,
            color,
            self.params.lossy_quality,
            self.params.lossy_method,
            keep_alpha,
            read_row,
        )?;
        let alpha_chunk = alpha_chunk(alpha_plane).filter(|_| has_alpha);
        self.write_container(b"VP8 ", parts, alpha_chunk, false, width, height)
    }

    /// Writes the RIFF container around a frame chunk of `frame_parts`, with a lossy frame's
    /// `ALPH` chunk.
    pub(crate) fn write_container(
        mut self,
        frame_chunk: &[u8; 4],
        frame_parts: Vec<Vec<u8>>,
        alpha_chunk: Option<Vec<u8>>,
        lossless_with_alpha: bool,
        width: u32,
        height: u32,
    ) -> Result<(), EncodingError> {
        let frame: Vec<&[u8]> = frame_parts.iter().map(Vec::as_slice).collect();
        let frame_len: usize = frame.iter().map(|p| p.len()).sum();
        let lossy_with_alpha = alpha_chunk.is_some();

        // If the image has no metadata and isn't lossy with alpha,
        // it can be encoded with the "simple" WebP container format.
        let use_simple_container = self.icc_profile.is_empty()
            && self.exif_metadata.is_empty()
            && self.xmp_metadata.is_empty()
            && !lossy_with_alpha;

        if use_simple_container {
            self.writer.write_all(b"RIFF")?;
            self.writer
                .write_all(&(chunk_size(frame_len) + 4).to_le_bytes())?;
            self.writer.write_all(b"WEBP")?;
            write_chunk_parts(&mut self.writer, frame_chunk, &frame)?;
        } else {
            let mut total_bytes = 22 + chunk_size(frame_len);
            if !self.icc_profile.is_empty() {
                total_bytes += chunk_size(self.icc_profile.len());
            }
            if !self.exif_metadata.is_empty() {
                total_bytes += chunk_size(self.exif_metadata.len());
            }
            if !self.xmp_metadata.is_empty() {
                total_bytes += chunk_size(self.xmp_metadata.len());
            }

            if let Some(alpha_chunk) = &alpha_chunk {
                total_bytes += chunk_size(alpha_chunk.len());
            }

            let mut flags = 0;
            if !self.xmp_metadata.is_empty() {
                flags |= 1 << 2;
            }
            if !self.exif_metadata.is_empty() {
                flags |= 1 << 3;
            }
            if lossy_with_alpha || lossless_with_alpha {
                flags |= 1 << 4;
            }
            if !self.icc_profile.is_empty() {
                flags |= 1 << 5;
            }

            self.writer.write_all(b"RIFF")?;
            self.writer.write_all(&total_bytes.to_le_bytes())?;
            self.writer.write_all(b"WEBP")?;

            let mut vp8x = Vec::new();
            vp8x.write_all(&[flags])?; // flags
            vp8x.write_all(&[0; 3])?; // reserved
            vp8x.write_all(&(width - 1).to_le_bytes()[..3])?; // canvas width
            vp8x.write_all(&(height - 1).to_le_bytes()[..3])?; // canvas height
            write_chunk(&mut self.writer, b"VP8X", &vp8x)?;

            if !self.icc_profile.is_empty() {
                write_chunk(&mut self.writer, b"ICCP", &self.icc_profile)?;
            }

            if let Some(alpha_chunk) = &alpha_chunk {
                write_chunk(&mut self.writer, b"ALPH", alpha_chunk)?;
            }

            write_chunk_parts(&mut self.writer, frame_chunk, &frame)?;

            if !self.exif_metadata.is_empty() {
                write_chunk(&mut self.writer, b"EXIF", &self.exif_metadata)?;
            }

            if !self.xmp_metadata.is_empty() {
                write_chunk(&mut self.writer, b"XMP ", &self.xmp_metadata)?;
            }
        }

        Ok(())
    }
}

/// Encodes a `width` x `height` alpha plane, one byte per pixel, as the payload of the `ALPH`
/// chunk of a lossy image the way [`WebPEncoder`] does with `params` (`lossy_method` and
/// `alpha_compression`), for [`WebPEncoder::encode_rows_with_alpha_chunk`]. Returns `None`
/// when every pixel is opaque, as such an image has no `ALPH` chunk.
///
/// # Panics
///
/// Panics if `alpha` does not have `width * height` bytes.
pub fn encode_alpha_chunk(
    alpha: &[u8],
    width: u32,
    height: u32,
    params: &EncoderParams,
) -> Option<Vec<u8>> {
    assert_eq!(alpha.len(), width as usize * height as usize);
    if alpha.iter().all(|&a| a == 0xff) {
        return None;
    }
    Some(alpha_chunk_payload(alpha, width, height, params))
}

pub(crate) fn alpha_chunk_payload(alpha: &[u8], width: u32, height: u32, params: &EncoderParams) -> Vec<u8> {
    if params.alpha_compression {
        vp8l::encode_alpha(alpha, width, height, params.lossy_method)
    } else {
        vp8l::encode_alpha_raw(alpha)
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, RngCore, SeedableRng};

    use super::*;

    #[test]
    fn write_webp() {
        let mut img = vec![0; 256 * 256 * 4];
        rand::thread_rng().fill_bytes(&mut img);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(EncoderParams {
            lossless_exact: true,
            ..Default::default()
        });
        encoder
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
            .unwrap();

        let mut decoder = crate::WebPDecoder::new(std::io::Cursor::new(output)).unwrap();
        let mut img2 = vec![0; 256 * 256 * 4];
        decoder.read_image(&mut img2).unwrap();
        assert_eq!(img, img2);
    }

    #[test]
    fn write_webp_exif() {
        let mut img = vec![0; 256 * 256 * 3];
        rand::thread_rng().fill_bytes(&mut img);

        let mut exif = vec![0; 10];
        rand::thread_rng().fill_bytes(&mut exif);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_exif_metadata(exif.clone());
        encoder
            .encode(&img, 256, 256, crate::ColorType::Rgb8)
            .unwrap();

        let mut decoder = crate::WebPDecoder::new(std::io::Cursor::new(output)).unwrap();

        let mut img2 = vec![0; 256 * 256 * 3];
        decoder.read_image(&mut img2).unwrap();
        assert_eq!(img, img2);

        let exif2 = decoder.exif_metadata().unwrap();
        assert_eq!(Some(exif), exif2);
    }

    #[test]
    fn roundtrip_libwebp() {
        roundtrip_libwebp_params(EncoderParams {
            lossless_exact: true,
            ..Default::default()
        });
        roundtrip_libwebp_params(EncoderParams {
            use_predictor_transform: false,
            lossless_exact: true,
            ..Default::default()
        });
    }

    /// 4000x4000 grey noise at quality 100 and method 0 wants 16 intra 4x4 modes a macroblock,
    /// which overrun partition 0's 512 KiB: libwebp fails, its callers restart it at method 3
    /// with `partition_limit` 100, and the encoder writes that restart's bytes.
    #[test]
    fn restarts_without_intra4x4_when_partition0_overflows() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4000);
        let (width, height) = (4000u32, 4000u32);
        let mut img = vec![0u8; (width * height * 3) as usize];
        for px in img.chunks_exact_mut(3) {
            px.fill(rng.gen());
        }
        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(EncoderParams {
            use_lossy: true,
            lossy_quality: 100,
            lossy_method: 0,
            ..Default::default()
        });
        encoder
            .encode(&img, width, height, crate::ColorType::Rgb8)
            .unwrap();
        let at = output.windows(4).position(|w| w == b"VP8 ").unwrap();
        let size0 = u32::from_le_bytes([output[at + 8], output[at + 9], output[at + 10], 0]) >> 5;
        assert!(size0 < 1 << 19, "{size0}");
        let mut streamed = Vec::new();
        let mut encoder = WebPEncoder::new(&mut streamed);
        encoder.set_params(EncoderParams {
            use_lossy: true,
            lossy_quality: 100,
            lossy_method: 0,
            ..Default::default()
        });
        let stride = width as usize * 3;
        encoder
            .encode_rows(width, height, crate::ColorType::Rgb8, |y, row| {
                row.copy_from_slice(&img[y as usize * stride..][..stride])
            })
            .unwrap();
        assert!(streamed == output, "streamed restart differs");
        let reference = libwebp_encode_with(&img, width, height, false, 100.0, 3, false, |c| {
            c.partition_limit = 100
        });
        assert!(output == reference);
    }

    #[allow(unsafe_code)]
    fn libwebp_decode(data: &[u8]) -> Vec<u8> {
        use libwebp_sys as sys;
        let mut features = unsafe { std::mem::zeroed::<sys::WebPBitstreamFeatures>() };
        let status = unsafe { sys::WebPGetFeatures(data.as_ptr(), data.len(), &mut features) };
        assert_eq!(status, sys::VP8StatusCode::VP8_STATUS_OK);
        let (mut w, mut h) = (0, 0);
        let bpp = if features.has_alpha != 0 { 4 } else { 3 };
        let ptr = unsafe {
            if bpp == 4 {
                sys::WebPDecodeRGBA(data.as_ptr(), data.len(), &mut w, &mut h)
            } else {
                sys::WebPDecodeRGB(data.as_ptr(), data.len(), &mut w, &mut h)
            }
        };
        assert!(!ptr.is_null());
        let out = unsafe { std::slice::from_raw_parts(ptr, (w * h * bpp) as usize) }.to_vec();
        unsafe { sys::WebPFree(ptr.cast()) };
        out
    }

    fn libwebp_encode(
        data: &[u8],
        width: u32,
        height: u32,
        rgba: bool,
        quality: f32,
        method: i32,
        lossless: bool,
    ) -> Vec<u8> {
        libwebp_encode_with(data, width, height, rgba, quality, method, lossless, |_| {})
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    fn libwebp_encode_with(
        data: &[u8],
        width: u32,
        height: u32,
        rgba: bool,
        quality: f32,
        method: i32,
        lossless: bool,
        tweak: impl FnOnce(&mut libwebp_sys::WebPConfig),
    ) -> Vec<u8> {
        use libwebp_sys as sys;
        unsafe {
            let mut config = std::mem::zeroed::<sys::WebPConfig>();
            let preset = sys::WebPPreset::WEBP_PRESET_DEFAULT;
            assert_ne!(
                sys::WebPConfigInitInternal(
                    &mut config,
                    preset,
                    quality,
                    sys::WEBP_ENCODER_ABI_VERSION as i32
                ),
                0
            );
            config.method = method;
            config.lossless = i32::from(lossless);
            tweak(&mut config);
            let mut pic = std::mem::zeroed::<sys::WebPPicture>();
            assert_ne!(
                sys::WebPPictureInitInternal(&mut pic, sys::WEBP_ENCODER_ABI_VERSION as i32),
                0
            );
            pic.width = width as i32;
            pic.height = height as i32;
            pic.use_argb = i32::from(lossless);
            if rgba {
                sys::WebPPictureImportRGBA(&mut pic, data.as_ptr(), (width * 4) as i32);
            } else {
                sys::WebPPictureImportRGB(&mut pic, data.as_ptr(), (width * 3) as i32);
            }
            let mut writer = std::mem::zeroed::<sys::WebPMemoryWriter>();
            sys::WebPMemoryWriterInit(&mut writer);
            pic.writer = Some(sys::WebPMemoryWrite);
            pic.custom_ptr = (&mut writer as *mut sys::WebPMemoryWriter).cast();
            assert_ne!(sys::WebPEncode(&config, &mut pic), 0);
            let out = std::slice::from_raw_parts(writer.mem, writer.size).to_vec();
            sys::WebPMemoryWriterClear(&mut writer);
            sys::WebPPictureFree(&mut pic);
            out
        }
    }

    /// A smooth gradient with noise, flat and sharp areas, in odd dimensions.
    fn test_pattern(width: u32, height: u32, bpp: usize) -> Vec<u8> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut img = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let noise: u8 = rng.gen_range(0..24);
                let px = if x < width / 4 {
                    [90, 160, 200]
                } else if (x / 8 + y / 8) % 2 == 0 && y > height / 2 {
                    [250, 250, 250]
                } else {
                    [
                        (x * 255 / width) as u8 ^ noise,
                        (y * 255 / height) as u8,
                        noise * 4,
                    ]
                };
                img.extend_from_slice(&px);
                if bpp == 4 {
                    img.push(if x + 9 > width {
                        0
                    } else if y < 5 {
                        128
                    } else {
                        255
                    });
                }
            }
        }
        img
    }

    #[test]
    fn lossy_matches_libwebp() {
        let (width, height) = (83, 61);
        for rgba in [false, true] {
            let img = test_pattern(width, height, if rgba { 4 } else { 3 });
            let color = if rgba {
                crate::ColorType::Rgba8
            } else {
                crate::ColorType::Rgb8
            };
            for method in 0..=6u8 {
                for quality in [10u8, 75, 99] {
                    let mut params = EncoderParams::default();
                    params.use_lossy = true;
                    params.lossy_quality = quality;
                    params.lossy_method = method;
                    let mut ours = Vec::new();
                    let mut encoder = WebPEncoder::new(&mut ours);
                    encoder.set_params(params);
                    encoder.encode(&img, width, height, color).unwrap();
                    let lib = libwebp_encode(
                        &img,
                        width,
                        height,
                        rgba,
                        f32::from(quality),
                        i32::from(method),
                        false,
                    );
                    assert_eq!(ours, lib, "rgba {rgba} method {method} quality {quality}");
                }
            }
        }
    }

    #[test]
    fn uncompressed_alpha_matches_libwebp() {
        let (width, height) = (83, 61);
        let img = test_pattern(width, height, 4);
        for method in [0u8, 4, 6] {
            let mut params = EncoderParams::default();
            params.use_lossy = true;
            params.lossy_quality = 75;
            params.lossy_method = method;
            params.alpha_compression = false;
            let mut ours = Vec::new();
            let mut encoder = WebPEncoder::new(&mut ours);
            encoder.set_params(params);
            encoder
                .encode(&img, width, height, crate::ColorType::Rgba8)
                .unwrap();
            let lib = libwebp_encode_with(
                &img,
                width,
                height,
                true,
                75.0,
                i32::from(method),
                false,
                |c| {
                    c.alpha_compression = 0;
                },
            );
            assert_eq!(ours, lib, "method {method}");
        }
    }

    #[test]
    fn precomputed_alpha_chunk_matches_encode() {
        for (width, height, opaque) in [(88, 61, false), (83, 61, false), (40, 33, true)] {
            let mut img = test_pattern(width, height, 4);
            if opaque {
                img.chunks_exact_mut(4).for_each(|p| p[3] = 255);
            }
            let alpha: Vec<u8> = img.chunks_exact(4).map(|p| p[3]).collect();
            for method in 0..=6u8 {
                for alpha_compression in [true, false] {
                    let params = EncoderParams {
                        use_lossy: true,
                        lossy_method: method,
                        alpha_compression,
                        ..Default::default()
                    };
                    let mut whole = Vec::new();
                    let mut encoder = WebPEncoder::new(&mut whole);
                    encoder.set_params(params.clone());
                    encoder
                        .encode(&img, width, height, crate::ColorType::Rgba8)
                        .unwrap();
                    let mut split = Vec::new();
                    let mut encoder = WebPEncoder::new(&mut split);
                    encoder.set_params(params.clone());
                    let stride = width as usize * 4;
                    std::thread::scope(|scope| {
                        let alpha_job = scope
                            .spawn(|| super::encode_alpha_chunk(&alpha, width, height, &params));
                        let read_row = |y: u32, row: &mut [u8]| {
                            row.copy_from_slice(&img[y as usize * stride..][..stride])
                        };
                        encoder
                            .encode_rows_with_alpha_chunk(
                                width,
                                height,
                                crate::ColorType::Rgba8,
                                read_row,
                                || {
                                    let chunk = alpha_job.join().unwrap();
                                    assert_eq!(chunk.is_none(), opaque);
                                    chunk
                                },
                            )
                            .unwrap();
                    });
                    assert!(
                        split == whole,
                        "{width}x{height} method {method} alpha_compression {alpha_compression}"
                    );
                    if method == 4 && alpha_compression {
                        assert!(split == libwebp_encode(&img, width, height, true, 95.0, 4, false));
                    }
                }
            }
        }
    }

    /// Four colours in blocks with some noise rows, to reach the palette paths.
    fn palette_pattern(width: u32, height: u32) -> Vec<u8> {
        let colors = [
            [0u8, 0, 0, 0],
            [255, 255, 255, 255],
            [200, 30, 90, 255],
            [10, 100, 250, 128],
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut img = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let i = if y % 7 == 3 {
                    rng.gen_range(0..4)
                } else {
                    ((x / 5 + y / 3) % 4) as usize
                };
                img.extend_from_slice(&colors[i]);
            }
        }
        img
    }

    #[test]
    fn lossless_matches_libwebp() {
        let (width, height) = (83, 61);
        let images = [
            (test_pattern(width, height, 3), false),
            (test_pattern(width, height, 4), true),
            (palette_pattern(width, height), true),
        ];
        for (img, rgba) in &images {
            let color = if *rgba {
                crate::ColorType::Rgba8
            } else {
                crate::ColorType::Rgb8
            };
            for method in 0..=6u8 {
                for quality in [0u8, 50, 75, 100] {
                    let mut params = EncoderParams::default();
                    params.lossless_quality = quality;
                    params.lossless_method = method;
                    let mut ours = Vec::new();
                    let mut encoder = WebPEncoder::new(&mut ours);
                    encoder.set_params(params);
                    encoder.encode(img, width, height, color).unwrap();
                    let lib = libwebp_encode(
                        img,
                        width,
                        height,
                        *rgba,
                        f32::from(quality),
                        i32::from(method),
                        true,
                    );
                    assert_eq!(ours, lib, "rgba {rgba} method {method} quality {quality}");
                }
            }
        }
    }

    #[test]
    fn lossy_decode_matches_libwebp() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        for (width, height) in [(83, 61), (301, 203), (17, 15), (1, 1), (40, 130)] {
            let pattern = test_pattern(width, height, 3);
            let noise: Vec<u8> = (0..width * height * 3).map(|_| rng.gen()).collect();
            for (img, quality) in [
                (&pattern, 10.0),
                (&pattern, 60.0),
                (&pattern, 95.0),
                (&noise, 100.0),
                (&noise, 40.0),
            ] {
                for (filter_type, sharpness, strength, segments, partitions) in [
                    (1, 0, 60, 4, 0),
                    (0, 0, 60, 4, 3),
                    (1, 7, 100, 1, 2),
                    (0, 3, 30, 2, 1),
                    (1, 2, 0, 3, 0),
                ] {
                    let data =
                        libwebp_encode_with(img, width, height, false, quality, 4, false, |c| {
                            c.filter_type = filter_type;
                            c.filter_sharpness = sharpness;
                            c.filter_strength = strength;
                            c.segments = segments;
                            c.partitions = partitions;
                        });
                    let lib = libwebp_decode(&data);
                    let what = format!("{width}x{height} q{quality} filter {filter_type}/{sharpness}/{strength} seg {segments} part {partitions}");
                    assert!(decode_whole(&data) == lib, "{what}");
                    let mut decoder = crate::WebPDecoder::new(std::io::Cursor::new(&data)).unwrap();
                    assert!(
                        decode_bands(decoder.band_reader().unwrap()) == lib,
                        "bands {what}"
                    );
                }
            }
        }
        for file in [
            "gallery1/1",
            "gallery1/2",
            "gallery1/3",
            "gallery1/4",
            "gallery1/5",
            "gallery2/1_webp_a",
            "gallery2/2_webp_a",
            "gallery2/3_webp_a",
            "gallery2/4_webp_a",
            "gallery2/5_webp_a",
        ] {
            let data = std::fs::read(format!("tests/images/{file}.webp")).unwrap();
            assert!(decode_whole(&data) == libwebp_decode(&data), "{file}");
        }
    }

    #[test]
    fn read_image_rows_matches_read_image() {
        for (width, height) in [(1, 1), (17, 9), (83, 61), (33, 47), (40, 130)] {
            for (lossy, rgba) in [(true, false), (true, true), (false, true), (false, false)] {
                let img = test_pattern(width, height, if rgba { 4 } else { 3 });
                let mut params = EncoderParams::default();
                params.use_lossy = lossy;
                params.lossy_quality = 75;
                let mut data = Vec::new();
                let mut encoder = WebPEncoder::new(&mut data);
                encoder.set_params(params);
                let color = if rgba {
                    crate::ColorType::Rgba8
                } else {
                    crate::ColorType::Rgb8
                };
                encoder.encode(&img, width, height, color).unwrap();
                for (simple, upsampling) in [
                    crate::UpsamplingMethod::Bilinear,
                    crate::UpsamplingMethod::Simple,
                ]
                .into_iter()
                .enumerate()
                {
                    let options = || crate::WebPDecodeOptions {
                        lossy_upsampling: upsampling,
                    };
                    let mut decoder = crate::WebPDecoder::new_with_options(
                        std::io::Cursor::new(&data),
                        options(),
                    )
                    .unwrap();
                    let mut whole = vec![0; decoder.output_buffer_size().unwrap()];
                    decoder.read_image(&mut whole).unwrap();
                    let mut decoder = crate::WebPDecoder::new_with_options(
                        std::io::Cursor::new(&data),
                        options(),
                    )
                    .unwrap();
                    let mut rows = Vec::new();
                    decoder
                        .read_image_rows(|y, band| {
                            assert_eq!(y as usize * whole.len() / height as usize, rows.len());
                            rows.extend_from_slice(band);
                        })
                        .unwrap();
                    assert!(
                        rows == whole,
                        "{width}x{height} lossy {lossy} rgba {rgba} simple {simple}"
                    );
                }
            }
        }
    }

    fn encode_test_image(width: u32, height: u32, lossy: bool, rgba: bool) -> Vec<u8> {
        let img = test_pattern(width, height, if rgba { 4 } else { 3 });
        let params = EncoderParams {
            use_lossy: lossy,
            lossy_quality: 75,
            ..Default::default()
        };
        let mut data = Vec::new();
        let mut encoder = WebPEncoder::new(&mut data);
        encoder.set_params(params);
        let color = if rgba {
            crate::ColorType::Rgba8
        } else {
            crate::ColorType::Rgb8
        };
        encoder.encode(&img, width, height, color).unwrap();
        data
    }

    fn decode_whole(data: &[u8]) -> Vec<u8> {
        let mut decoder = crate::WebPDecoder::new(std::io::Cursor::new(data)).unwrap();
        let mut whole = vec![0; decoder.output_buffer_size().unwrap()];
        decoder.read_image(&mut whole).unwrap();
        whole
    }

    fn decode_bands(mut bands: crate::BandReader) -> Vec<u8> {
        let (width, height) = bands.dimensions();
        let stride = width as usize * bands.bytes_per_pixel();
        let max_rows = bands.max_band_rows() as usize;
        let mut rows = Vec::new();
        while let Some((y, band)) = bands.next_band().unwrap() {
            assert_eq!(y as usize * stride, rows.len());
            assert_eq!(band.len() % stride, 0);
            assert!(!band.is_empty() && band.len() / stride <= max_rows);
            rows.extend_from_slice(band);
        }
        assert_eq!(rows.len(), stride * height as usize);
        assert!(bands.next_band().unwrap().is_none());
        rows
    }

    #[test]
    fn band_reader_matches_read_image() {
        let sizes = [
            (1, 1),
            (5, 3),
            (16, 16),
            (17, 15),
            (31, 33),
            (9, 48),
            (13, 49),
            (7, 63),
            (64, 64),
            (3, 65),
            (45, 97),
        ];
        for (width, height) in sizes {
            for (lossy, rgba) in [(true, false), (true, true), (false, true), (false, false)] {
                let data = encode_test_image(width, height, lossy, rgba);
                let mut decoder = crate::WebPDecoder::new(std::io::Cursor::new(&data)).unwrap();
                let rows = decode_bands(decoder.band_reader().unwrap());
                assert!(
                    rows == decode_whole(&data),
                    "{width}x{height} lossy {lossy} rgba {rgba}"
                );
            }
        }
        for file in ["random_lossy", "random_lossless"] {
            let data = std::fs::read(format!("tests/images/animated/{file}.webp")).unwrap();
            let mut decoder = crate::WebPDecoder::new(std::io::Cursor::new(&data)).unwrap();
            let rows = decode_bands(decoder.band_reader().unwrap());
            assert!(rows == decode_whole(&data), "{file}");
        }
    }

    /// The bytes written so far and whether that is all, with a condvar signalling more.
    type PipeState = std::sync::Arc<(std::sync::Mutex<(Vec<u8>, bool)>, std::sync::Condvar)>;

    /// A reader over bytes another thread is still appending, blocking until they arrive.
    struct Pipe {
        shared: PipeState,
        pos: u64,
    }

    impl std::io::Read for Pipe {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let (lock, ready) = &*self.shared;
            let mut state = lock.lock().unwrap();
            while state.0.len() as u64 <= self.pos && !state.1 {
                state = ready.wait(state).unwrap();
            }
            let available = &state.0[(self.pos as usize).min(state.0.len())..];
            let n = available.len().min(buf.len());
            buf[..n].copy_from_slice(&available[..n]);
            self.pos += n as u64;
            Ok(n)
        }
    }

    impl std::io::Seek for Pipe {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            self.pos = match pos {
                std::io::SeekFrom::Start(p) => p,
                std::io::SeekFrom::Current(d) => self.pos.checked_add_signed(d).unwrap(),
                std::io::SeekFrom::End(d) => {
                    let (lock, ready) = &*self.shared;
                    let state = ready.wait_while(lock.lock().unwrap(), |s| !s.1).unwrap();
                    (state.0.len() as u64).checked_add_signed(d).unwrap()
                }
            };
            Ok(self.pos)
        }
    }

    #[test]
    fn band_reader_runs_on_consumer_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<crate::BandReader>();
        for (lossy, rgba) in [(true, false), (true, true), (false, true)] {
            let data = encode_test_image(301, 203, lossy, rgba);
            let shared: PipeState = Default::default();
            let (band_tx, band_rx) = std::sync::mpsc::sync_channel::<(u32, Vec<u8>)>(2);
            let rows = std::thread::scope(|scope| {
                let feed = shared.clone();
                let input = &data;
                scope.spawn(move || {
                    for chunk in input.chunks(1009) {
                        std::thread::yield_now();
                        let (lock, ready) = &*feed;
                        lock.lock().unwrap().0.extend_from_slice(chunk);
                        ready.notify_all();
                    }
                    let (lock, ready) = &*feed;
                    lock.lock().unwrap().1 = true;
                    ready.notify_all();
                });
                let pipe = std::io::BufReader::new(Pipe {
                    shared: shared.clone(),
                    pos: 0,
                });
                let mut decoder =
                    std::thread::spawn(move || crate::WebPDecoder::new(pipe).unwrap())
                        .join()
                        .unwrap();
                scope.spawn(move || {
                    let mut bands = decoder.band_reader().unwrap();
                    while let Some((y, band)) = bands.next_band().unwrap() {
                        band_tx.send((y, band.to_vec())).unwrap();
                    }
                });
                let mut rows = Vec::new();
                let mut count = 0;
                for (y, band) in band_rx {
                    assert_eq!(y as usize * 301 * if rgba { 4 } else { 3 }, rows.len());
                    rows.extend_from_slice(&band);
                    count += 1;
                }
                assert!(!lossy || count > 1);
                rows
            });
            assert!(rows == decode_whole(&data), "lossy {lossy} rgba {rgba}");
        }
    }

    fn roundtrip_libwebp_params(params: EncoderParams) {
        println!("Testing {params:?}");

        let mut img = vec![0; 256 * 256 * 4];
        rand::thread_rng().fill_bytes(&mut img);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder
            .encode(&img[..256 * 256 * 3], 256, 256, crate::ColorType::Rgb8)
            .unwrap();
        let decoded = libwebp_decode(&output);
        assert_eq!(img[..256 * 256 * 3], *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
            .unwrap();
        let decoded = libwebp_decode(&output);
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder.set_icc_profile(vec![0; 10]);
        encoder
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
            .unwrap();
        let decoded = libwebp_decode(&output);
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder.set_exif_metadata(vec![0; 10]);
        encoder
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
            .unwrap();
        let decoded = libwebp_decode(&output);
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params);
        encoder.set_xmp_metadata(vec![0; 7]);
        encoder.set_icc_profile(vec![0; 8]);
        encoder.set_icc_profile(vec![0; 9]);
        encoder
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
            .unwrap();
        let decoded = libwebp_decode(&output);
        assert_eq!(img, *decoded);
    }
}
