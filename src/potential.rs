use ndarray::Array3;
use rustfft::num_complex::{Complex, Complex64};

use crate::config::OMEGA_M0;
use crate::fourier::*;

pub fn potential(density: Array3<f64>, fgrid: &Array3<f64>, t: f64) -> Array3<f64> {
    let grid: Array3<Complex64> = density_k(density);
    let grid: Array3<Complex64> = potential_k(grid, fgrid, t);
    potential_real(grid)
}

fn density_k(density: Array3<f64>) -> Array3<Complex64> {
    forward(&density.map(|x| Complex { re: *x, im: 0. }))
}

pub fn potential_k(density: Array3<Complex64>, fgrid: &Array3<f64>, t: f64) -> Array3<Complex64> {
    let fgrid: Array3<Complex64> = fgrid.map(|x| Complex { re: *x, im: 0. });
    let c = Complex::new((-3. * OMEGA_M0 / 8.0) as f64 / t, 0.);
    (density * fgrid).map(|x| c * x)
}

fn potential_real(potential_k: Array3<Complex64>) -> Array3<f64> {
    inverse(potential_k).map(|x| x.re)
}
