use std::f64::consts::PI;

use ndarray::{Array1, ScalarOperand};
use num_traits::{Float, FromPrimitive, NumOps};

#[inline]
fn cosine_window<'a, T: 'a>(a: f64, b: f64, c: f64, d: f64, size: usize) -> Array1<T>
where
	T: NumOps + ScalarOperand + Float + Sync + Send + FromPrimitive
{
	Array1::from_shape_fn(size, |i| {
		let x = (PI * i as f64) / (size - 1) as f64;
		let b_ = b * (2.0f64 * x).cos();
		let c_ = c * (4.0f64 * x).cos();
		let d_ = d * (6.0f64 * x).cos();
		T::from_f64((a - b_) + (c_ - d_)).unwrap()
	})
}

#[inline]
fn triangular_window<'a, T: 'a>(l: f64, size: usize) -> Array1<T>
where
	T: NumOps + ScalarOperand + Float + Sync + Send + FromPrimitive
{
	Array1::from_shape_fn(size, |i| {
		let x = 1.0f64 - ((i as f64 - (size - 1) as f64 / 2.0f64) / (l as f64 / 2.0f64)).abs();
		T::from_f64(x).unwrap()
	})
}

#[inline]
pub fn hanning<'a, T: 'a>(size: usize) -> Array1<T>
where
	T: NumOps + ScalarOperand + Float + Sync + Send + FromPrimitive
{
	cosine_window(0.5, 0.5, 0.0, 0.0, size)
}

#[inline]
pub fn hamming<'a, T: 'a>(size: usize) -> Array1<T>
where
	T: NumOps + ScalarOperand + Float + Sync + Send + FromPrimitive
{
	cosine_window(0.54, 0.46, 0.0, 0.0, size)
}

#[inline]
pub fn blackman<'a, T: 'a>(size: usize) -> Array1<T>
where
	T: NumOps + ScalarOperand + Float + Sync + Send + FromPrimitive
{
	cosine_window(0.35875, 0.48829, 0.14128, 0.01168, size)
}

#[inline]
pub fn nuttall<'a, T: 'a>(size: usize) -> Array1<T>
where
	T: NumOps + ScalarOperand + Float + Sync + Send + FromPrimitive
{
	cosine_window(0.355768, 0.487396, 0.144232, 0.012604, size)
}

#[inline]
pub fn triangular<'a, T: 'a>(size: usize) -> Array1<T>
where
	T: NumOps + ScalarOperand + Float + Sync + Send + FromPrimitive
{
	triangular_window(size as f64, size)
}
