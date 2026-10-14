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
    }
}

const fn chunk_size(inner_bytes: usize) -> u32 {
    if inner_bytes % 2 == 1 {
        (inner_bytes + 1) as u32 + 8
    } else {
        inner_bytes as u32 + 8
    }
}

fn write_chunk<W: Write>(w: W, name: &[u8], data: &[u8]) -> io::Result<()> {
    write_chunk_parts(w, name, &[data])
}

fn write_chunk_parts<W: Write>(mut w: W, name: &[u8], parts: &[&[u8]]) -> io::Result<()> {
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
        let (parts, alpha) = encode_frame_lossy(
            width,
            height,
            color,
            self.params.lossy_quality,
            self.params.lossy_method,
            read_row,
        )?;
        self.write_container(b"VP8 ", parts, alpha, false, width, height)
    }

    /// Writes the RIFF container around a frame chunk of `frame_parts`, with the `ALPH` chunk
    /// of a lossy frame's `alpha_plane`.
    fn write_container(
        mut self,
        frame_chunk: &[u8; 4],
        frame_parts: Vec<Vec<u8>>,
        alpha_plane: Option<Vec<u8>>,
        lossless_with_alpha: bool,
        width: u32,
        height: u32,
    ) -> Result<(), EncodingError> {
        let frame: Vec<&[u8]> = frame_parts.iter().map(Vec::as_slice).collect();
        let frame_len: usize = frame.iter().map(|p| p.len()).sum();
        let lossy_with_alpha = alpha_plane.is_some();

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

            let alpha_chunk_data = if let Some(alpha) = &alpha_plane {
                let alpha_chunk =
                    vp8l::encode_alpha(alpha, width, height, self.params.lossy_method);

                total_bytes += chunk_size(alpha_chunk.len());
                Some(alpha_chunk)
            } else {
                None
            };

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

            if let Some(alpha_chunk) = alpha_chunk_data {
                write_chunk(&mut self.writer, b"ALPH", &alpha_chunk)?;
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

    #[allow(unsafe_code)]
    fn libwebp_encode(
        data: &[u8],
        width: u32,
        height: u32,
        rgba: bool,
        quality: f32,
        method: i32,
        lossless: bool,
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
