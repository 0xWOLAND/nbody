use image::{ImageBuffer, RgbImage};
use ndarray::{s, Array, Array2, Array3, Axis};

use crate::config::{IMG_WIDTH, N_CELLS, N_PARTICLES};

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
            .map(|x| ((*x * (IMG_WIDTH / N_CELLS) as f64) as usize).rem_euclid(IMG_WIDTH))
            .collect();
        a.slice_mut(s![idx[0], idx[1], ..]).fill(255.);
    }

    array_3_to_image(a.map(|x| *x as u8), Some(IMG_WIDTH))
}

use plotters::prelude::*;
pub fn hist(arr: Vec<u32>, title: Option<String>) {
    let title = title.unwrap_or_else(|| String::from("histogram"));
    let OUT_FILE_NAME = &format!("img/{}.png", title);
    let root = BitMapBackend::new(OUT_FILE_NAME, (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption(title, ("sans-serif", 50.0))
        .build_cartesian_2d(
            (0u32..(N_CELLS as u32 + 1)).into_segmented(),
            0u32..(2 * N_PARTICLES.pow(2) as u32),
        )
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("sans-serif", 15))
        .draw()
        .unwrap();

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(RED.mix(0.5).filled())
                .data(arr.iter().map(|x: &u32| (*x, 1))),
        )
        .unwrap();

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);
}

#[cfg(test)]
mod tests {
    use super::hist;

    #[test]
    fn test_hist() {
        hist((1..20).collect::<Vec<u32>>(), None);
    }
}
