use ndarray::{Array, Array2, Array3, ShapeBuilder};

#[derive(Debug)]
pub struct Meshgrid2 {
    grid: Vec<(f64, f64)>,
    dim: (usize, usize),
}

impl Meshgrid2 {
    pub fn new(xs: &Vec<f64>, ys: &Vec<f64>) -> Meshgrid2 {
        let grid: Vec<(f64, f64)> = ys
            .clone()
            .iter()
            .flat_map(move |y| xs.clone().iter().map(move |x| (*x, *y)).collect::<Vec<_>>())
            .collect();
        Meshgrid2 {
            grid,
            dim: (xs.len(), ys.len()),
        }
    }

    pub fn x(&self) -> Array2<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.0).collect()).unwrap()
    }

    pub fn y(&self) -> Array2<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.1).collect()).unwrap()
    }

    pub fn get(&self) -> (Array2<f64>, Array2<f64>) {
        (self.x(), self.y())
    }
}

#[derive(Debug)]
pub struct Meshgrid3 {
    grid: Vec<(f64, f64, f64)>,
    dim: (usize, usize, usize),
}

impl Meshgrid3 {
    pub fn new(xs: &Vec<f64>, ys: &Vec<f64>, zs: &Vec<f64>) -> Meshgrid3 {
        let grid: Vec<(f64, f64, f64)> = zs
            .clone()
            .iter()
            .flat_map(move |z| {
                ys.clone()
                    .iter()
                    .flat_map(move |y| {
                        xs.clone()
                            .iter()
                            .map(move |x| (*x, *y, *z))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        Meshgrid3 {
            grid,
            dim: (xs.len(), ys.len(), zs.len()),
        }
    }

    pub fn x(&self) -> Array3<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.0).collect()).unwrap()
    }

    pub fn y(&self) -> Array3<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.1).collect()).unwrap()
    }

    pub fn z(&self) -> Array3<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.2).collect()).unwrap()
    }

    pub fn get(&self) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        (self.x(), self.y(), self.z())
    }
}

#[cfg(test)]
mod Tests {
    use super::Meshgrid2;

    #[test]
    fn test() {
        let m = Meshgrid2::new(&vec![1., 2.], &vec![3., 4.]);
        println!("{:?}", m);
        println!("{:?}", m.x());
        println!("{:?}", m.y());
    }
}
