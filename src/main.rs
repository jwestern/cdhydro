use ndarray::prelude::*;

fn main() {
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_add_and_mul() {
        let ones = Array::from_elem((2,2),1.);
        let twos = Array::from_elem((2,2),2.);
        let added = &twos + &ones;
        let multd = &twos * &ones;
        assert_eq!(added, Array::from_elem((2,2),3.));
        assert_eq!(multd, Array::from_elem((2,2),2.));
    }
}
