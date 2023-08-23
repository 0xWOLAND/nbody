use ndarray::Array3;
use rustfft::num_complex::{Complex, Complex64};

use crate::config::{N_CELLS, OMEGA_M0};
use crate::fourier::*;
use crate::utils::array_3_to_image;

pub fn potential(density: Array3<f64>, fgrid: &Array3<f64>, t: f64) -> Array3<f64> {
    let grid: Array3<Complex64> = density_k(density);
    let grid: Array3<Complex64> = potential_k(grid, fgrid, t);
    let res: Array3<f64> = potential_real(grid);
    // let img = array_3_to_image(
    //     res.map(|x| ((*x) * 100.).min(u8::MAX as f64 - 1.) as u8),
    //     Some(N_CELLS),
    // );
    // let _ = img.save(format!("./img/positions_small/pot{}.png", t));
    res
}

fn density_k(density: Array3<f64>) -> Array3<Complex64> {
    forward(&density.map(|x| Complex { re: *x, im: 0. }))
}

pub fn potential_k(density: Array3<Complex64>, fgrid: &Array3<f64>, t: f64) -> Array3<Complex64> {
    let fgrid: Array3<Complex64> = fgrid.map(|x| Complex { re: *x, im: 0. });
    let c = Complex::new((-3. * OMEGA_M0 / 8.0) / t, 0.);
    let a: Array3<Complex<f64>> = density * fgrid;
    (a).map(|x| c * x)
}

fn potential_real(potential_k: Array3<Complex64>) -> Array3<f64> {
    inverse(&potential_k).map(|x| x.re)
}
