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

        let x_c = (x.floor() as usize).rem_euclid(N_CELLS);
        let y_c = (y.floor() as usize).rem_euclid(N_CELLS);
        let z_c = (z.floor() as usize).rem_euclid(N_CELLS);

        let d_x = x - (x_c as f64);
        let d_y = y - (y_c as f64);
        let d_z = z - (z_c as f64);

        let [t_x, t_y, t_z] = [d_x, d_y, d_z].map(|x| 1. - x);

        let [X, Y, Z] = [x_c, y_c, z_c].map(|x| (x + 1).rem_euclid(N_CELLS));

        grid.slice_mut(s![z_c, y_c, x_c])
            .iter_mut()
            .for_each(|x| *x += mass * t_x * t_y * t_z);

        grid.slice_mut(s![z_c, y_c, X])
            .iter_mut()
            .for_each(|x| *x += mass * d_x * t_y * t_z);
        grid.slice_mut(s![z_c, Y, x_c])
            .iter_mut()
            .for_each(|x| *x += mass * t_x * d_y * t_z);
        grid.slice_mut(s![Z, y_c, x_c])
            .iter_mut()
            .for_each(|x| *x += mass * t_x * t_y * d_z);

        grid.slice_mut(s![z_c, Y, X])
            .iter_mut()
            .for_each(|x| *x += mass * d_x * d_y * t_z);
        grid.slice_mut(s![Z, Y, x_c])
            .iter_mut()
            .for_each(|x| *x += mass * t_x * d_y * d_z);
        grid.slice_mut(s![Z, y_c, X])
            .iter_mut()
            .for_each(|x| *x += mass * d_x * t_y * d_z);

        grid.slice_mut(s![Z, Y, X])
            .iter_mut()
            .for_each(|x| *x += mass * d_x * d_y * d_z);
    }

    grid
}
