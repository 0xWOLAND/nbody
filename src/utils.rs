use image::{ImageBuffer, RgbImage};
use ndarray::Array3;

use crate::config::{N_CELLS, N_PARTICLES};

pub fn array_3_to_image(
    a: Array3<u8>,
    width: Option<usize>,
    height: Option<usize>,
) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    RgbImage::from_raw(
        width.unwrap_or(N_PARTICLES) as u32,
        height.unwrap_or(N_PARTICLES) as u32,
        a.into_raw_vec(),
    )
    .unwrap()
}
