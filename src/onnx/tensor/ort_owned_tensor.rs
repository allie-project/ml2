use std::{fmt::Debug, ops::Deref, ptr, rc};

use ndarray::{Array, ArrayView};
use tracing::debug;

use super::{ndarray_tensor::NdArrayTensor, TensorDataToType, TensorElementDataType};
use crate::onnx::{memory::MemoryInfo, ortsys, sys, OrtError, OrtResult};

/// A wrapper around a tensor produced by onnxruntime inference.
///
/// Since different outputs for the same model can have different types, this type is used to allow
/// the user to dynamically query each output's type and extract the appropriate tensor type with
/// [try_extract].
#[derive(Debug)]
pub struct DynOrtTensor<'m, D>
where
	D: ndarray::Dimension
{
	tensor_ptr_holder: rc::Rc<TensorPointerDropper>,
	memory_info: &'m MemoryInfo,
	shape: D,
	data_type: TensorElementDataType
}

impl<'m, D> DynOrtTensor<'m, D>
where
	D: ndarray::Dimension
{
	pub(crate) fn new(tensor_ptr: *mut sys::OrtValue, memory_info: &'m MemoryInfo, shape: D, data_type: TensorElementDataType) -> DynOrtTensor<'m, D> {
		DynOrtTensor {
			tensor_ptr_holder: rc::Rc::from(TensorPointerDropper { tensor_ptr }),
			memory_info,
			shape,
			data_type
		}
	}

	/// The ONNX data type this tensor contains.
	pub fn data_type(&self) -> TensorElementDataType {
		self.data_type
	}

	/// Extract a tensor containing `T`.
	///
	/// Where the type permits it, the tensor will be a view into existing memory.
	///
	/// # Errors
	///
	/// An error will be returned if `T`'s ONNX type doesn't match this tensor's type, or if an
	/// onnxruntime error occurs.
	pub fn try_extract<'t, T>(&self) -> OrtResult<OrtOwnedTensor<'t, T, D>>
	where
		T: TensorDataToType + Clone + Debug,
		'm: 't // mem info outlives tensor
	{
		if self.data_type != T::tensor_element_data_type() {
			Err(OrtError::DataTypeMismatch {
				actual: self.data_type,
				requested: T::tensor_element_data_type()
			})
		} else {
			// Note: Both tensor and array will point to the same data, nothing is copied.
			// As such, there is no need to free the pointer used to create the ArrayView.
			assert_ne!(self.tensor_ptr_holder.tensor_ptr, ptr::null_mut());

			let mut is_tensor = 0;
			ortsys![unsafe IsTensor(self.tensor_ptr_holder.tensor_ptr, &mut is_tensor) -> OrtError::FailedTensorCheck];
			assert_eq!(is_tensor, 1);

			let array_view = T::extract_array(self.shape.clone(), self.tensor_ptr_holder.tensor_ptr)?;

			Ok(OrtOwnedTensor::new(self.tensor_ptr_holder.clone(), array_view))
		}
	}
}

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
pub struct OrtOwnedTensor<'t, T, D>
where
	T: TensorDataToType,
	D: ndarray::Dimension
{
	tensor_ptr_holder: rc::Rc<TensorPointerDropper>,
	array_view: ArrayView<'t, T, D>
}

impl<'t, T, D> Deref for OrtOwnedTensor<'t, T, D>
where
	T: TensorDataToType,
	D: ndarray::Dimension
{
	type Target = ArrayView<'t, T, D>;

	fn deref(&self) -> &Self::Target {
		&self.array_view
	}
}

impl<'t, T, D> OrtOwnedTensor<'t, T, D>
where
	T: TensorDataToType,
	D: ndarray::Dimension
{
	pub(crate) fn new(tensor_ptr_holder: rc::Rc<TensorPointerDropper>, array_view: ArrayView<'t, T, D>) -> OrtOwnedTensor<'t, T, D> {
		OrtOwnedTensor { tensor_ptr_holder, array_view }
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
/// Holds on to a tensor pointer until dropped.
///
/// This allows for creating an [`OrtOwnedTensor`] from a [`DynOrtTensor`] without consuming `self`, which would prevent
/// retrying extraction and avoids awkward interaction with the outputs `Vec`. It also avoids requiring `OrtOwnedTensor`
/// to keep a reference to `DynOrtTensor`, which would be inconvenient.
#[derive(Debug)]
pub(crate) struct TensorPointerDropper {
	tensor_ptr: *mut sys::OrtValue
}

impl Drop for TensorPointerDropper {
	#[tracing::instrument]
	fn drop(&mut self) {
		debug!("Dropping OrtOwnedTensor.");
		ortsys![unsafe ReleaseValue(self.tensor_ptr)];

		self.tensor_ptr = ptr::null_mut();
	}
}
