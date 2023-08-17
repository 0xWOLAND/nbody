use ndarray::{s, Array, Array1, Array2, Array3, ArrayView1, ArrayViewMut1, Axis};

use crate::{config::N_CELLS, cosmology::expansion_factor, potential::potential};

pub fn update(
    density: Array3<f64>,
    positions: Array2<f64>,
    velocities: Array2<f64>,
    fgrid: Array3<f64>,
    t: f64,
    dt: f64,
) {
    let potentials = potential(density, fgrid, t);
    let f_a = expansion_factor(t);
}

fn integrate(
    mut positions: Array2<f64>,
    mut velocities: Array2<f64>,
    potentials: Array3<f64>,
    t: f64,
    f_a: f64,
    dt: f64,
) {
    let centered_cells: Array2<f64> = positions.map(|x| x.floor());
    let weights = cic_weights(&positions, &centered_cells);

    for i in 0..=2 {
        let (next_position, next_velocity) = interpolate(
            &positions.slice_mut(s![i, ..]),
            &velocities.slice_mut(s![i, ..]),
            &centered_cells,
            &potentials,
            &weights,
            &f_a,
            &t,
            &dt,
            i,
        );
        positions.slice_mut(s![i, ..]).assign(&next_position);
        velocities.slice_mut(s![i, ..]).assign(&next_velocity);
    }
}

fn cic_weights(positions: &Array2<f64>, centered_cells: &Array2<f64>) -> Array2<f64> {
    let len = positions.len_of(Axis(0));
    let mut t: Array2<f64> = Array2::zeros((8, len));
    let diff: Array2<f64> = positions - centered_cells;
    for i in 0..len {
        let dx = diff.get((0, i)).unwrap();
        let dy = diff.get((1, i)).unwrap();
        let dz = diff.get((2, i)).unwrap();

        let [tx, ty, tz] = [dx, dy, dz].map(|x| 1. - x);
        t.slice_mut(s![0, i]).fill(tx * ty * tz);
        t.slice_mut(s![1, i]).fill(dx * ty * tz);
        t.slice_mut(s![2, i]).fill(tx * dy * tz);
        t.slice_mut(s![3, i]).fill(dx * dy * tz);
        t.slice_mut(s![4, i]).fill(tx * ty * tz);
        t.slice_mut(s![5, i]).fill(dx * ty * tz);
        t.slice_mut(s![6, i]).fill(dx * ty * tz);
        t.slice_mut(s![7, i]).fill(dx * ty * tz);
    }

    t
}

fn interpolate(
    mut positions: &ArrayViewMut1<f64>,
    mut velocities: &ArrayViewMut1<f64>,
    centered_cells: &Array2<f64>,
    potentials: &Array3<f64>,
    cic_weights: &Array2<f64>,
    f_a: &f64,
    t: &f64,
    dt: &f64,
    axis: i32,
) -> (Array1<f64>, Array1<f64>) {
    let mut n_next = centered_cells.clone();
    let mut n_prev = centered_cells.clone();
    let n_cells = N_CELLS as f64;

    n_next.slice_mut(s![axis, ..]).iter_mut().for_each(|x| {
        *x = (*x - 1.0 + n_cells) % n_cells;
    });

    n_prev.slice_mut(s![axis, ..]).iter_mut().for_each(|x| {
        *x = (*x + 1.0) % n_cells;
    });

    let len = positions.len_of(Axis(0));
    let mut next_velocities: Array1<f64> = Array1::zeros(len);
    let mut next_positions: Array1<f64> = Array1::zeros(len);

    for i in 0..len {
        let x1 = *(n_prev.get((0, i)).unwrap()) as usize;
        let y1 = *(n_prev.get((1, i)).unwrap()) as usize;
        let z1 = *(n_prev.get((2, i)).unwrap()) as usize;
        let x2 = *(n_next.get((0, i)).unwrap()) as usize;
        let y2 = *(n_next.get((1, i)).unwrap()) as usize;
        let z2 = *(n_next.get((2, i)).unwrap()) as usize;

        let [X1, Y1, Z1] = [x1, y1, z1].map(|x| (x + 1) % N_CELLS);
        let [X2, Y2, Z2] = [x2, y2, z2].map(|x| (x + 1) % N_CELLS);

        let weight = cic_weights.slice(s![.., i]);

        let g = potentials.get((x2, y2, z2)).unwrap() - potentials.get((x1, y1, z1)).unwrap();
        let g_x = potentials.get((X2, y2, z2)).unwrap() - potentials.get((X1, y1, z1)).unwrap();
        let g_y = potentials.get((x2, Y2, z2)).unwrap() - potentials.get((x1, Y1, z1)).unwrap();
        let g_z = potentials.get((x2, y2, Z2)).unwrap() - potentials.get((x1, y1, Z1)).unwrap();
        let g_xy = potentials.get((X2, Y2, z2)).unwrap() - potentials.get((X1, Y1, z1)).unwrap();
        let g_xz = potentials.get((X2, y2, Z2)).unwrap() - potentials.get((X1, y1, Z1)).unwrap();
        let g_yz = potentials.get((x2, Y2, Z2)).unwrap() - potentials.get((x1, Y1, Z1)).unwrap();
        let g_xyz = potentials.get((X2, Y2, Z2)).unwrap() - potentials.get((X1, Y1, Z1)).unwrap();
        let g_p = [g, g_x, g_y, g_z, g_xy, g_xz, g_yz, g_xyz]
            .iter()
            .zip(weight)
            .fold(0., |sum, x| sum + x.0 * x.1)
            / 2.;

        let v = velocities.get(i).unwrap();
        let x = positions.get(i).unwrap();
        next_velocities.slice_mut(s![i]).fill(v + dt * f_a * g_p);
        next_positions
            .slice_mut(s![i])
            .fill(x + dt * v * f_a * f64::sqrt(t + dt));
    }
    (next_positions, next_velocities)
}
