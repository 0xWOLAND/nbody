use crate::config::{MASS, N_CELLS, N_PARTICLES};
use ndarray::{s, Array1, Array2, Array3, AssignElem};
use rand::Rng;
use std::fmt;

#[derive(Debug)]
pub struct Particles {
    pub positions: Array2<f64>,
    pub velocities: Array2<f64>,
    pub density: Array3<f64>,
}

impl fmt::Display for Particles {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "x: {}\nv: {}\ndensity: {}",
            self.positions, self.velocities, self.density
        )
    }
}

impl Particles {
    #[warn(non_snake_case)]
    pub fn new() -> Particles {
        let mut rng = rand::thread_rng();
        let mut gen_particle_data = || {
            Array1::<f64>::zeros(3 * N_PARTICLES)
                .map(|_| N_CELLS as f64 * rng.gen::<f64>())
                .into_shape((3, N_PARTICLES))
                .unwrap()
        };

        let positions: Array2<f64> = gen_particle_data();
        let velocities: Array2<f64> = gen_particle_data();

        let mut density: Array3<f64> = Array3::<f64>::zeros((N_CELLS, N_CELLS, N_CELLS));
        for col in positions.columns() {
            let colc: Vec<i32> = col.map(|x| *x as i32 % N_CELLS as i32).to_vec().into();

            let x = colc[0];
            let y = colc[1];
            let z = colc[2];

            let dx = colc[0] as f64 - col[0];
            let dy = colc[1] as f64 - col[1];
            let dz = colc[2] as f64 - col[2];

            let [tx, ty, tz] = [dx, dy, dz].map(|x| 1. - x);
            let [X, Y, Z] = [x, y, z].map(|x| (x + 1) % N_CELLS as i32);

            density.slice_mut(s![x, y, z,]).fill(MASS * tx * ty * tz);
            density.slice_mut(s![X, y, z,]).fill(MASS * dx * ty * tz);
            density.slice_mut(s![x, Y, z,]).fill(MASS * tx * dy * tz);
            density.slice_mut(s![X, Y, z,]).fill(MASS * dx * dy * tz);
            density.slice_mut(s![x, y, Z,]).fill(MASS * tx * ty * dz);
            density.slice_mut(s![X, y, Z,]).fill(MASS * dx * ty * dz);
            density.slice_mut(s![x, Y, Z,]).fill(MASS * tx * dy * dz);
            density.slice_mut(s![X, Y, Z,]).fill(MASS * dx * dy * dz);
        }

        Particles {
            positions,
            velocities,
            density,
        }
    }
}

pub fn test() -> String {
    format!("{}", Particles::new())
}
