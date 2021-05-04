use ndarray::prelude::*;
use ndarray_npy::{read_npy, write_npy, WriteNpyError, ReadNpyError, ReadNpyExt};

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
}
