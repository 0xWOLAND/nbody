pub fn test() -> String {
    "".to_string()
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3};

    use crate::config::{A_INIT, N_CELLS, N_PARTICLES};
    use crate::density::density;
    use crate::fourier::ksq_inv;
    use crate::integrate::update;
    use crate::utils::array_2_to_image;
    use crate::{ic::initial_conditions, random_field::gaussian_random_field};

    #[test]
    pub fn simulate() {
        let average_density = (N_CELLS / N_PARTICLES).pow(3) as f64;
        let rho: Array3<f64> = gaussian_random_field(0.845, 3.685);
        let (positions, velocities): (Array2<f64>, Array2<f64>) = initial_conditions(rho.clone());

        let img = array_2_to_image(positions.clone(), N_CELLS);
        img.save("positions.png");
        let fgrid = ksq_inv();
        let t = A_INIT;
        let dt = 100.;
        let den = density(&positions, average_density);
        println!("p1 {:?}", positions);
        let (positions, velocities): (Array2<f64>, Array2<f64>) =
            update(den, positions, velocities, fgrid, t, dt);
        println!("p2 {:?}", positions);
        let img = array_2_to_image(positions.clone(), N_CELLS);
        img.save("positions2.png");
    }
}
