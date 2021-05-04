use ndarray::prelude::*;

fn main() {
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_add() {
        let ones = Array::from_elem((2,2),1.);
        let twos = Array::from_elem((2,2),2.);
        let added = &twos + &ones;
        assert_eq!(added, Array::from_elem((2,2),3.));
    }

    #[test]
    fn test_ndarray_mul() {
        let ones = Array::from_elem((2,2),1.);
        let twos = Array::from_elem((2,2),2.);
        let multd = &twos * &ones;
        assert_eq!(multd, Array::from_elem((2,2),2.));
    }
}
