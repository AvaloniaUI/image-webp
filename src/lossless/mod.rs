mod decoder;
#[cfg(feature = "encoder")]
pub(crate) mod encoder;

pub(crate) use decoder::{AlphaBands, LosslessBands, LosslessDecoder};

#[cfg(feature = "encoder")]
/// Allows fine-tuning some encoder parameters.
///
/// Pass to [`WebPEncoder::set_params()`](crate::WebPEncoder::set_params).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct EncoderParams {
    /// Let lossless encoding use the predictor transform. Enabled by default.
    pub use_predictor_transform: bool,
    /// Use the lossy encoding to encode the image using the VP8 compression format.
    pub use_lossy: bool,
    /// A quality value for the lossy encoding that must be between 0 and 100. Defaults to 95.
    pub lossy_quality: u8,
    /// Lossy speed/size trade-off, 0 (fastest) to 6 (smallest), as libwebp's `method`.
    /// It also sets the effort spent on the alpha plane. Defaults to 4.
    pub lossy_method: u8,
    /// Lossless effort, 0 to 100, as libwebp's `quality` with `lossless = 1`. Defaults to 75.
    pub lossless_quality: u8,
    /// Lossless speed/size trade-off, 0 (fastest) to 6 (smallest), as libwebp's `method`.
    /// Defaults to 4.
    pub lossless_method: u8,
    /// Keep the RGB values of fully transparent pixels in lossless encoding, as libwebp's
    /// `exact`. Off by default, which lets them become black for better compression.
    pub lossless_exact: bool,
    /// Near-lossless level for lossless encoding, 0 (most change) to 100 (none), as libwebp's
    /// `near_lossless`: pixels move within a bound before the encode where that packs their
    /// residuals better. Defaults to 100.
    pub near_lossless: u8,
    /// Compress the alpha plane of lossy images, as libwebp's `alpha_compression`. When off,
    /// the `ALPH` chunk holds the plane uncompressed, which is faster and bigger. Defaults to on.
    pub alpha_compression: bool,
}

#[cfg(feature = "encoder")]
impl Default for EncoderParams {
    fn default() -> Self {
        Self {
            use_predictor_transform: true,
            use_lossy: false,
            lossy_quality: 95,
            lossy_method: 4,
            lossless_quality: 75,
            lossless_method: 4,
            lossless_exact: false,
            near_lossless: 100,
            alpha_compression: true,
        }
    }
}

const CODE_LENGTH_CODES: usize = 19;
const CODE_LENGTH_CODE_ORDER: [usize; CODE_LENGTH_CODES] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

#[rustfmt::skip]
const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),  (1, 0),  (1, 1),  (-1, 1), (0, 2),  (2, 0),  (1, 2),  (-1, 2),
    (2, 1),  (-2, 1), (2, 2),  (-2, 2), (0, 3),  (3, 0),  (1, 3),  (-1, 3),
    (3, 1),  (-3, 1), (2, 3),  (-2, 3), (3, 2),  (-3, 2), (0, 4),  (4, 0),
    (1, 4),  (-1, 4), (4, 1),  (-4, 1), (3, 3),  (-3, 3), (2, 4),  (-2, 4),
    (4, 2),  (-4, 2), (0, 5),  (3, 4),  (-3, 4), (4, 3),  (-4, 3), (5, 0),
    (1, 5),  (-1, 5), (5, 1),  (-5, 1), (2, 5),  (-2, 5), (5, 2),  (-5, 2),
    (4, 4),  (-4, 4), (3, 5),  (-3, 5), (5, 3),  (-5, 3), (0, 6),  (6, 0),
    (1, 6),  (-1, 6), (6, 1),  (-6, 1), (2, 6),  (-2, 6), (6, 2),  (-6, 2),
    (4, 5),  (-4, 5), (5, 4),  (-5, 4), (3, 6),  (-3, 6), (6, 3),  (-6, 3),
    (0, 7),  (7, 0),  (1, 7),  (-1, 7), (5, 5),  (-5, 5), (7, 1),  (-7, 1),
    (4, 6),  (-4, 6), (6, 4),  (-6, 4), (2, 7),  (-2, 7), (7, 2),  (-7, 2),
    (3, 7),  (-3, 7), (7, 3),  (-7, 3), (5, 6),  (-5, 6), (6, 5),  (-6, 5),
    (8, 0),  (4, 7),  (-4, 7), (7, 4),  (-7, 4), (8, 1),  (8, 2),  (6, 6),
    (-6, 6), (8, 3),  (5, 7),  (-5, 7), (7, 5),  (-7, 5), (8, 4),  (6, 7),
    (-6, 7), (7, 6),  (-7, 6), (8, 5),  (7, 7),  (-7, 7), (8, 6),  (8, 7)
];
