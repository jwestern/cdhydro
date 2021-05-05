use ndarray::prelude::*;
use ndarray_npy::{read_npy, write_npy, NpzWriter, NpzReader};
use std::fs::File;

fn main() {
    let zeros = Array::from_elem((2,2),0.);
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
        let zeros = Array::from_elem((2,2),0.);
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
