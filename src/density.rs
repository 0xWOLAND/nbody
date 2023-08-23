use ndarray::{s, Array2, Array3, Axis};
use rand::Rng;

use crate::config::N_CELLS;

pub fn density(positions: &Array2<f64>, mass: f64) -> Array3<f64> {
    let mut grid = Array3::<f64>::zeros((N_CELLS, N_CELLS, N_CELLS));
    let len = positions.len_of(Axis(1));

    for i in 0..len {
        let x = positions.get((0, i)).unwrap();
        let y = positions.get((1, i)).unwrap();
        let z = positions.get((2, i)).unwrap();

        let x_c = (positions.get((0, i)).unwrap().floor() as usize).rem_euclid(N_CELLS);
        let y_c = (positions.get((1, i)).unwrap().floor() as usize).rem_euclid(N_CELLS);
        let z_c = (positions.get((2, i)).unwrap().floor() as usize).rem_euclid(N_CELLS);

        let d_x = positions.get((0, i)).unwrap() - (x_c as f64);
        let d_y = positions.get((1, i)).unwrap() - (y_c as f64);
        let d_z = positions.get((2, i)).unwrap() - (z_c as f64);

        let [t_x, t_y, t_z] = [d_x, d_y, d_z].map(|x| 1. - x);

        let [X, Y, Z] = [x_c, y_c, z_c].map(|x| (x + 1).rem_euclid(N_CELLS));

        grid.slice_mut(s![x_c, y_c, z_c])
            .iter_mut()
            .for_each(|x| *x += mass * t_x * t_y * t_z);

        grid.slice_mut(s![X, y_c, z_c])
            .iter_mut()
            .for_each(|x| *x += mass * d_x * t_y * t_z);
        grid.slice_mut(s![x_c, Y, z_c])
            .iter_mut()
            .for_each(|x| *x += mass * t_x * d_y * t_z);
        grid.slice_mut(s![x_c, y_c, Z])
            .iter_mut()
            .for_each(|x| *x += mass * t_x * t_y * d_z);

        grid.slice_mut(s![X, Y, z_c])
            .iter_mut()
            .for_each(|x| *x += mass * d_x * d_y * t_z);
        grid.slice_mut(s![X, y_c, Z])
            .iter_mut()
            .for_each(|x| *x += mass * d_x * t_y * d_z);
        grid.slice_mut(s![x_c, Y, Z])
            .iter_mut()
            .for_each(|x| *x += mass * t_x * d_y * d_z);

        grid.slice_mut(s![X, Y, Z])
            .iter_mut()
            .for_each(|x| *x += mass * d_x * d_y * d_z);
    }

    grid
}
