use ndarray::{Array, Array1, Array2, ArrayView, AsArray, Axis, Dimension, ScalarOperand, Slice, Zip};
use num_traits::{Float, NumOps};

pub fn diff<'a, T: 'a, D, V>(data: V, axis: Option<Axis>) -> Array<T, D>
where
	T: NumOps + ScalarOperand,
	D: Dimension,
	V: AsArray<'a, T, D>
{
	let data_view = data.into();
	let axis = axis.unwrap_or_else(|| Axis(data_view.ndim() - 1));

	let head = data_view.slice_axis(axis, Slice::from(..-1));
	let tail = data_view.slice_axis(axis, Slice::from(1..));

	&tail - &head
}

pub fn subtract_outer<'a, T: 'a>(x: &Array1<T>, y: &Array1<T>) -> Array2<T>
where
	T: NumOps + ScalarOperand + Copy
{
	let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
	Array2::from_shape_fn((size_x, size_y), |(i, j)| x[i] - y[j])
}

pub fn maximum<'a, T: 'a, D, V>(x: V, y: V) -> Array<T, D>
where
	T: NumOps + ScalarOperand + Float + Sync + Send,
	D: Dimension,
	V: AsArray<'a, T, D>
{
	let y = y.into();
	Zip::from(x.into()).and::<ArrayView<T, D>>(y).par_map_collect(|x: &T, y: &T| x.max(*y))
}

pub fn minimum<'a, T: 'a, D, V>(x: V, y: V) -> Array<T, D>
where
	T: NumOps + ScalarOperand + Float + Sync + Send,
	D: Dimension,
	V: AsArray<'a, T, D>
{
	let y = y.into();
	Zip::from(x.into()).and::<ArrayView<T, D>>(y).par_map_collect(|x: &T, y: &T| x.min(*y))
}

#[cfg(test)]
mod tests {
	use ndarray::array;

	use super::*;

	#[test]
	fn test_diff() {
		let x = array![[1, 3, 6, 10], [0, 5, 6, 8]];
		let y = diff(&x, None);
		assert_eq!(y, array![[2, 3, 4], [5, 1, 2]]);
	}

	#[test]
	fn test_subtract_outer() {
		let x = array![2., 3., 4.];
		let y = array![1., 5., 2.];
		let z = subtract_outer(&x, &y);
		assert_eq!(z, array![[1., -3., 0.], [2., -2., 1.], [3., -1., 2.]]);
	}

	#[test]
	fn test_maximum() {
		let max = maximum(&array![2., 3., 4.], &array![1., 5., 2.]);
		assert_eq!(max, array![2., 5., 4.]);
	}

	#[test]
	fn test_minimum() {
		let min = minimum(&array![2., 3., 4.], &array![1., 5., 2.]);
		assert_eq!(min, array![1., 3., 2.]);
	}
}
