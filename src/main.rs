use ndarray::prelude::*;
use ndarray::stack;
use ndarray_npy::{read_npy, write_npy, NpzWriter, NpzReader};
use std::fs::File;

fn index_periodic(i: isize, N: isize) -> usize {
    if i > N-1 {
        (i - N) as usize
    } else if i < 0 {
        (N + i) as usize
    } else {
        i as usize
    }
}

fn fpx_iph(f: Array2<f64>, N: isize) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((N as usize,N as usize).f());
    for yi in 0..f.shape()[0] {
        for xi in 0..f.shape()[1] {
            result[[yi,xi]] = (1.0 / 60.0) * (2.0 * f[[yi, index_periodic(xi as isize - 2, N)]] 
                                           - 13.0 * f[[yi, index_periodic(xi as isize - 1, N)]] 
                                           + 47.0 * f[[yi, index_periodic(xi as isize    , N)]] 
                                           + 27.0 * f[[yi, index_periodic(xi as isize + 1, N)]] 
                                           -  3.0 * f[[yi, index_periodic(xi as isize + 2, N)]])
        };
    };
    result
}

fn fmx_imh(f: Array2<f64>, N: isize) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((N as usize,N as usize).f());
    for yi in 0..f.shape()[0] {
        for xi in 0..f.shape()[1] {
            result[[yi,xi]] = (1.0 / 60.0) * (2.0 * f[[yi, index_periodic(xi as isize + 2, N)]]
                                           - 13.0 * f[[yi, index_periodic(xi as isize + 1, N)]]
                                           + 47.0 * f[[yi, index_periodic(xi as isize    , N)]]
                                           + 27.0 * f[[yi, index_periodic(xi as isize - 1, N)]]
                                           -  3.0 * f[[yi, index_periodic(xi as isize - 2, N)]])
        };
    };
    result
}

fn fpy_iph(f: Array2<f64>, N: isize) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((N as usize,N as usize).f());
    for yi in 0..f.shape()[0] {
        for xi in 0..f.shape()[1] {
            result[[yi,xi]] = (1.0 / 60.0) * (2.0 * f[[index_periodic(yi as isize - 2, N), yi]]
                                           - 13.0 * f[[index_periodic(yi as isize - 1, N), yi]]
                                           + 47.0 * f[[index_periodic(yi as isize    , N), yi]]
                                           + 27.0 * f[[index_periodic(yi as isize + 1, N), yi]]
                                           -  3.0 * f[[index_periodic(yi as isize + 2, N), yi]])
        };
    };
    result
}

fn fmy_imh(f: Array2<f64>, N: isize) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((N as usize,N as usize).f());
    for yi in 0..f.shape()[0] {
        for xi in 0..f.shape()[1] {
            result[[yi,xi]] = (1.0 / 60.0) * (2.0 * f[[index_periodic(yi as isize + 2, N), yi]]
                                           - 13.0 * f[[index_periodic(yi as isize + 1, N), yi]]
                                           + 47.0 * f[[index_periodic(yi as isize    , N), yi]]
                                           + 27.0 * f[[index_periodic(yi as isize - 1, N), yi]]
                                           -  3.0 * f[[index_periodic(yi as isize - 2, N), yi]])
        };
    };
    result
}

fn main() {
    let N: isize = 4;
    let dx = 0.1;
    let xx = Array::from_shape_fn((N as usize, N as usize), |(_i, j)| (j as f64) * dx - ((N as f64) - 1.0) * dx / 2.0);
    let yy =-Array::from_shape_fn((N as usize, N as usize), |( i,_j)| (i as f64) * dx - ((N as f64) - 1.0) * dx / 2.0);
    let rr = (&xx * &xx + &yy * &yy).mapv(|a| a.powf(0.5));
    println!("{:?}", xx.slice(s![0,..]));
    println!("{:?}", yy.slice(s![..,0]));
    println!("{:?}", rr);
    fpx_iph(rr, N);

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_add() {
        let ones = Array::from_elem((2,2),1.);
        let twos = Array::from_elem((2,2),2.);
        let added = &twos + &ones;
        assert_eq!(added, Array::from_elem((2,2), 3.));
    }

    #[test]
    fn test_ndarray_mul() {
        let ones = Array::from_elem((2,2),1.);
        let twos = Array::from_elem((2,2),2.);
        let multd = &twos * &ones;
        assert_eq!(multd, Array::from_elem((2,2), 2.));
    }

    #[test]
    fn test_ndarray_npy_read_write() {
        let zeros = Array::from_elem((5,5),0.);
        for i in zeros.axis_iter(Axis(0)) {
            println!("{}",i);
        }
        write_npy("test.npy", &zeros);
        let recover: Array2<f64> = read_npy("test.npy").unwrap();
        assert_eq!(zeros, recover);
    }

    #[test]
    fn test_ndarray_npz_read_write() {
        let zeros = Array::from_elem((2,2),0.);
        let time = Array::from_elem(1,0.5);
        let mut npz = NpzWriter::new(File::create("test.npz").unwrap());
        npz.add_array("data", &zeros);
        npz.add_array("time", &time);
        npz.finish();
        let mut npzread = NpzReader::new(File::open("test.npz").unwrap()).unwrap();
        let timeread: Array1<f64> = npzread.by_name("time").unwrap();
        let dataread: Array2<f64> = npzread.by_name("data").unwrap();
        assert_eq!(zeros, dataread);
        assert_eq!(time,  timeread);
    }
}
