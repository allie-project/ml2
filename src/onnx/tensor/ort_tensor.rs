use std::{ffi, fmt::Debug, ops::Deref};

use ndarray::Array;
use tracing::{debug, error};

use crate::onnx::{
	error::{assert_non_null_pointer, call_ort, status_to_result},
	memory::MemoryInfo,
	ort, sys,
	tensor::{ndarray_tensor::NdArrayTensor, IntoTensorElementDataType, TensorElementDataType},
	OrtError, OrtResult
};

/// Owned tensor, backed by an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
///
/// This tensor bounds the ONNX Runtime to `ndarray`; it is used to copy an
/// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) to the runtime's memory.
///
/// **NOTE**: The type is not meant to be used directly, use an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
/// instead.
#[derive(Debug)]
pub struct OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	pub(crate) c_ptr: *mut sys::OrtValue,
	array: Array<T, D>,
	#[allow(dead_code)]
	memory_info: &'t MemoryInfo
}

impl<'t, T, D> OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	fn from_array_numeric<'m>(
		memory_info: &'m MemoryInfo,
		array: &'m mut Array<T, D>,
		tensor_ptr: &'m *mut sys::OrtValue,
		tensor_ptr_ptr: *mut *mut sys::OrtValue,
		shape_ptr: *const i64,
		shape_len: usize
	) -> OrtResult<()>
	where
		'm: 't // 'm outlives 't
	{
		// primitive data is already suitably laid out in memory; provide it to
		// onnxruntime as is
		let tensor_values_ptr: *mut std::ffi::c_void = array.as_mut_ptr() as *mut std::ffi::c_void;
		assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

		unsafe {
			call_ort(|ort| {
				ort.CreateTensorWithDataAsOrtValue.unwrap()(
					memory_info.ptr,
					tensor_values_ptr,
					array.len() * std::mem::size_of::<T>(),
					shape_ptr,
					shape_len,
					T::tensor_element_data_type().into(),
					tensor_ptr_ptr
				)
			})
		}
		.map_err(OrtError::CreateTensorWithData)?;
		assert_non_null_pointer(tensor_ptr, "Tensor")?;

		let mut is_tensor = 0;
		let status = unsafe { ort().IsTensor.unwrap()(*tensor_ptr, &mut is_tensor) };
		status_to_result(status).map_err(OrtError::IsTensor)?;
		Ok(())
	}

	pub(crate) fn from_array<'m>(memory_info: &'m MemoryInfo, allocator_ptr: *mut sys::OrtAllocator, mut array: Array<T, D>) -> OrtResult<OrtTensor<'t, T, D>>
	where
		'm: 't // 'm outlives 't
	{
		// where onnxruntime will write the tensor data to
		let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
		let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;

		let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = array.shape().len();

		match T::tensor_element_data_type() {
			TensorElementDataType::Float32
			| TensorElementDataType::Uint8
			| TensorElementDataType::Int8
			| TensorElementDataType::Uint16
			| TensorElementDataType::Int16
			| TensorElementDataType::Int32
			| TensorElementDataType::Int64
			| TensorElementDataType::Float64
			| TensorElementDataType::Uint32
			| TensorElementDataType::Uint64 => OrtTensor::from_array_numeric(memory_info, &mut array, &tensor_ptr, tensor_ptr_ptr, shape_ptr, shape_len)?,
			#[cfg(feature = "half")]
			TensorElementDataType::Bfloat16 | TensorElementDataType::Float16 => {
				OrtTensor::from_array_numeric(memory_info, &mut array, &tensor_ptr, tensor_ptr_ptr, shape_ptr, shape_len)?
			}
			TensorElementDataType::String => {
				// create tensor without data -- data is filled in later
				unsafe {
					call_ort(|ort| {
						ort.CreateTensorAsOrtValue.unwrap()(allocator_ptr, shape_ptr, shape_len, T::tensor_element_data_type().into(), tensor_ptr_ptr)
					})
				}
				.map_err(OrtError::CreateTensor)?;

				// create null-terminated copies of each string, as per `FillStringTensor` docs
				let null_terminated_copies: Vec<ffi::CString> = array
					.iter()
					.map(|elt| {
						let slice = elt.try_utf8_bytes().expect("String data type must provide utf8 bytes");
						ffi::CString::new(slice)
					})
					.collect::<std::result::Result<Vec<_>, _>>()
					.map_err(OrtError::CStringNulError)?;

				let string_pointers = null_terminated_copies.iter().map(|cstring| cstring.as_ptr()).collect::<Vec<_>>();

				unsafe { call_ort(|ort| ort.FillStringTensor.unwrap()(tensor_ptr, string_pointers.as_ptr(), string_pointers.len())) }
					.map_err(OrtError::FillStringTensor)?;
			}
			_ => unimplemented!("Tensor element data type {:?} not yet implemented", T::tensor_element_data_type())
		}

		assert_non_null_pointer(tensor_ptr, "Tensor")?;

		Ok(OrtTensor {
			c_ptr: tensor_ptr,
			array,
			memory_info
		})
	}
}

impl<'t, T, D> Deref for OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	type Target = Array<T, D>;

	fn deref(&self) -> &Self::Target {
		&self.array
	}
}

impl<'t, T, D> Drop for OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	#[tracing::instrument]
	fn drop(&mut self) {
		// We need to let the C part free
		debug!("Dropping Tensor.");
		if self.c_ptr.is_null() {
			error!("Null pointer, not calling free.");
		} else {
			unsafe { ort().ReleaseValue.unwrap()(self.c_ptr) }
		}

		self.c_ptr = std::ptr::null_mut();
	}
}

impl<'t, T, D> OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	/// Apply a softmax on the specified axis
	pub fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
	where
		D: ndarray::RemoveAxis,
		T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign
	{
		self.array.softmax(axis)
	}
}

#[cfg(test)]
mod tests {
	use std::ptr;

	use ndarray::{arr0, arr1, arr2, arr3};
	use test_log::test;

	use super::*;
	use crate::onnx::{AllocatorType, MemType};

	#[test]
	fn orttensor_from_array_0d_i32() {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
		let array = arr0::<i32>(123);
		let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array).unwrap();
		let expected_shape: &[usize] = &[];
		assert_eq!(tensor.shape(), expected_shape);
	}

	#[test]
	fn orttensor_from_array_1d_i32() {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
		let array = arr1(&[1_i32, 2, 3, 4, 5, 6]);
		let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array).unwrap();
		let expected_shape: &[usize] = &[6];
		assert_eq!(tensor.shape(), expected_shape);
	}

	#[test]
	fn orttensor_from_array_2d_i32() {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
		let array = arr2(&[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]);
		let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array).unwrap();
		assert_eq!(tensor.shape(), &[2, 6]);
	}

	#[test]
	fn orttensor_from_array_3d_i32() {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
		let array = arr3(&[
			[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
			[[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
			[[25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]]
		]);
		let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array).unwrap();
		assert_eq!(tensor.shape(), &[3, 2, 6]);
	}

	#[test]
	fn orttensor_from_array_1d_string() {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
		let array = arr1(&[String::from("foo"), String::from("bar"), String::from("baz")]);
		let tensor = OrtTensor::from_array(&memory_info, ort_default_allocator(), array).unwrap();
		assert_eq!(tensor.shape(), &[3]);
	}

	#[test]
	fn orttensor_from_array_3d_str() {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
		let array = arr3(&[[["1", "2", "3"], ["4", "5", "6"]], [["7", "8", "9"], ["10", "11", "12"]]]);
		let tensor = OrtTensor::from_array(&memory_info, ort_default_allocator(), array).unwrap();
		assert_eq!(tensor.shape(), &[2, 2, 3]);
	}

	fn ort_default_allocator() -> *mut sys::OrtAllocator {
		let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
		unsafe {
			// this default non-arena allocator doesn't need to be deallocated
			call_ort(|ort| ort.GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr))
		}
		.unwrap();
		allocator_ptr
	}
}
