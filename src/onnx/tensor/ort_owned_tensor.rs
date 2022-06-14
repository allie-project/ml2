use std::{fmt::Debug, ops::Deref};

use ndarray::{Array, ArrayView};
use tracing::debug;

use super::{ndarray_tensor::NdArrayTensor, TensorDataToType};
use crate::onnx::{memory::MemoryInfo, ort, sys};

/// Tensor containing data owned by the ONNX Runtime C library, used to return values from inference.
///
/// This tensor type is returned by the [`Session::run()`](../session/struct.Session.html#method.run) method.
/// It is not meant to be created directly.
///
/// The tensor hosts an [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
/// of the data on the C side. This allows manipulation on the Rust side using `ndarray` without copying the data.
///
/// `OrtOwnedTensor` implements the [`std::deref::Deref`](#impl-Deref) trait for ergonomic access to
/// the underlying [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).
#[derive(Debug)]
pub struct OrtOwnedTensor<'t, 'm, T, D>
where
	T: TensorDataToType,
	D: ndarray::Dimension,
	'm: 't // 'm outlives 't
{
	pub(crate) tensor_ptr: *mut sys::OrtValue,
	array_view: ArrayView<'t, T, D>,
	#[allow(dead_code)]
	memory_info: &'m MemoryInfo
}

impl<'t, 'm, T, D> Deref for OrtOwnedTensor<'t, 'm, T, D>
where
	T: TensorDataToType,
	D: ndarray::Dimension
{
	type Target = ArrayView<'t, T, D>;

	fn deref(&self) -> &Self::Target {
		&self.array_view
	}
}

impl<'t, 'm, T, D> OrtOwnedTensor<'t, 'm, T, D>
where
	T: TensorDataToType,
	D: ndarray::Dimension
{
	pub(crate) fn new(tensor_ptr: *mut sys::OrtValue, array_view: ArrayView<'t, T, D>, memory_info: &'m MemoryInfo) -> OrtOwnedTensor<'t, 'm, T, D> {
		OrtOwnedTensor { tensor_ptr, array_view, memory_info }
	}

	/// Apply a softmax on the specified axis
	pub fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
	where
		D: ndarray::RemoveAxis,
		T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign
	{
		self.array_view.softmax(axis)
	}
}

impl<'t, 'm, T, D> Drop for OrtOwnedTensor<'t, 'm, T, D>
where
	T: TensorDataToType,
	D: ndarray::Dimension,
	'm: 't // 'm outlives 't
{
	#[tracing::instrument]
	fn drop(&mut self) {
		debug!("Dropping OrtOwnedTensor.");
		unsafe { ort().ReleaseValue.unwrap()(self.tensor_ptr) }

		self.tensor_ptr = std::ptr::null_mut();
	}
}
