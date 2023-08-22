use image::{ImageBuffer, RgbImage};
use ndarray::{s, Array, Array2, Array3, Axis};

use crate::config::{IMG_WIDTH, N_CELLS};

pub fn array_3_to_image(
    a: Array3<u8>,
    width: Option<usize>,
) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let x: Vec<u8> = a.into_raw_vec();
    RgbImage::from_raw(
        width.unwrap_or(IMG_WIDTH) as u32,
        width.unwrap_or(IMG_WIDTH) as u32,
        x,
    )
    .unwrap()
}

pub fn array_2_to_image(arr: Array2<f64>, width: usize) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let mut a: Array3<f64> = Array3::zeros((IMG_WIDTH, IMG_WIDTH, 3));
    let len = arr.len_of(Axis(1));

    for i in 0..len {
        let idx: Vec<usize> = arr
            .slice(s![.., i])
            .iter()
            .map(|x| ((*x * (IMG_WIDTH / N_CELLS) as f64) as usize + IMG_WIDTH) % IMG_WIDTH)
            .collect();
        a.slice_mut(s![idx[0], idx[1], ..]).fill(255.);
    }

    array_3_to_image(a.map(|x| *x as u8), Some(IMG_WIDTH))
}
