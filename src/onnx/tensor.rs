//! Module containing tensor types.
//!
//! Two main types of tensors are available.
//!
//! The first one, [`Tensor`](struct.Tensor.html),
//! is an _owned_ tensor that is backed by [`ndarray`](https://crates.io/crates/ndarray).
//! This kind of tensor is used to pass input data for the inference.
//!
//! The second one, [`OrtOwnedTensor`](struct.OrtOwnedTensor.html), is used
//! internally to pass to the ONNX Runtime inference execution to place
//! its output values. It is built using a [`OrtOwnedTensorExtractor`](struct.OrtOwnedTensorExtractor.html)
//! following the builder pattern.
//!
//! Once "extracted" from the runtime environment, this tensor will contain an
//! [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
//! containing _a view_ of the data. When going out of scope, this tensor will free the required
//! memory on the C side.
//!
//! **NOTE**: Tensors are not meant to be built directly. When performing inference,
//! the [`Session::run()`](../session/struct.Session.html#method.run) method takes
//! an `ndarray::Array` as input (taking ownership of it) and will convert it internally
//! to a [`Tensor`](struct.Tensor.html). After inference, a [`OrtOwnedTensor`](struct.OrtOwnedTensor.html)
//! will be returned by the method which can be derefed into its internal
//! [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).

pub mod ndarray_tensor;
pub mod ort_owned_tensor;
pub mod ort_tensor;

use std::{fmt, ptr};

pub use ort_owned_tensor::OrtOwnedTensor;
pub use ort_tensor::OrtTensor;

use super::{
	error::assert_non_null_pointer,
	sys::{self as sys, OnnxEnumInt},
	OrtError, OrtResult
};

/// Enum mapping ONNX Runtime's supported tensor data types.
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum TensorElementDataType {
	/// 32-bit floating point number, equivalent to Rust's `f32`.
	Float32 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT as OnnxEnumInt,
	/// Unsigned 8-bit integer, equivalent to Rust's `u8`.
	Uint8 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 as OnnxEnumInt,
	/// Signed 8-bit integer, equivalent to Rust's `i8`.
	Int8 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 as OnnxEnumInt,
	/// Unsigned 16-bit integer, equivalent to Rust's `u16`.
	Uint16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 as OnnxEnumInt,
	/// Signed 16-bit integer, equivalent to Rust's `i16`.
	Int16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 as OnnxEnumInt,
	/// Signed 32-bit integer, equivalent to Rust's `i32`.
	Int32 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 as OnnxEnumInt,
	/// Signed 64-bit integer, equivalent to Rust's `i64`.
	Int64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 as OnnxEnumInt,
	/// String, equivalent to Rust's `String`.
	String = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING as OnnxEnumInt,
	/// Boolean, equivalent to Rust's `bool`.
	Bool = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL as OnnxEnumInt,
	#[cfg(feature = "half")]
	/// 16-bit floating point number, equivalent to `half::f16` (requires the `half` crate).
	Float16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 as OnnxEnumInt,
	/// 64-bit floating point number, equivalent to Rust's `f64`. Also known as `double`.
	Float64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE as OnnxEnumInt,
	/// Unsigned 32-bit integer, equivalent to Rust's `u32`.
	Uint32 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 as OnnxEnumInt,
	/// Unsigned 64-bit integer, equivalent to Rust's `u64`.
	Uint64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 as OnnxEnumInt,
	// /// Complex 64-bit floating point number, equivalent to Rust's `num_complex::Complex<f64>`.
	// Complex64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 as OnnxEnumInt,
	// TODO: `num_complex` crate doesn't support i128 provided by the `decimal` crate.
	// /// Complex 128-bit floating point number, equivalent to Rust's `num_complex::Complex<f128>`.
	// Complex128 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 as OnnxEnumInt,
	/// Brain 16-bit floating point number, equivalent to `half::bf16` (requires the `half` crate).
	#[cfg(feature = "half")]
	Bfloat16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 as OnnxEnumInt
}

impl From<TensorElementDataType> for sys::ONNXTensorElementDataType {
	fn from(val: TensorElementDataType) -> Self {
		match val {
			TensorElementDataType::Float32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			TensorElementDataType::Uint8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
			TensorElementDataType::Int8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
			TensorElementDataType::Uint16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
			TensorElementDataType::Int16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
			TensorElementDataType::Int32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
			TensorElementDataType::Int64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
			TensorElementDataType::String => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
			TensorElementDataType::Bool => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
			#[cfg(feature = "half")]
			TensorElementDataType::Float16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
			TensorElementDataType::Float64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
			TensorElementDataType::Uint32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
			TensorElementDataType::Uint64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
			// TensorElementDataType::Complex64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
			// TensorElementDataType::Complex128 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
			#[cfg(feature = "half")]
			TensorElementDataType::Bfloat16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
		}
	}
}

/// Trait used to map Rust types (for example `f32`) to ONNX tensor element data types (for example `Float`).
pub trait IntoTensorElementDataType {
	/// Returns the ONNX tensor element data type corresponding to the given Rust type.
	fn tensor_element_data_type() -> TensorElementDataType;

	/// If the type is `String`, returns `Some` with UTF-8 contents, else `None`.
	fn try_utf8_bytes(&self) -> Option<&[u8]>;
}

macro_rules! impl_type_trait {
	($type_:ty, $variant:ident) => {
		impl IntoTensorElementDataType for $type_ {
			fn tensor_element_data_type() -> TensorElementDataType {
				TensorElementDataType::$variant
			}

			fn try_utf8_bytes(&self) -> Option<&[u8]> {
				None
			}
		}
	};
}

impl_type_trait!(f32, Float32);
impl_type_trait!(u8, Uint8);
impl_type_trait!(i8, Int8);
impl_type_trait!(u16, Uint16);
impl_type_trait!(i16, Int16);
impl_type_trait!(i32, Int32);
impl_type_trait!(i64, Int64);
impl_type_trait!(bool, Bool);
#[cfg(feature = "half")]
impl_type_trait!(half::f16, Float16);
impl_type_trait!(f64, Float64);
impl_type_trait!(u32, Uint32);
impl_type_trait!(u64, Uint64);
// impl_type_trait!(num_complex::Complex<f64>, Complex64);
// impl_type_trait!(num_complex::Complex<f128>, Complex128);
#[cfg(feature = "half")]
impl_type_trait!(half::bf16, Bfloat16);

/// Adapter for common Rust string types to ONNX strings.
///
/// It should be easy to use both `String` and `&str` as [TensorElementDataType::String] data, but
/// we can't define an automatic implementation for anything that implements `AsRef<str>` as it
/// would conflict with the implementations of [IntoTensorElementDataType] for primitive numeric
/// types (which might implement `AsRef<str>` at some point in the future).
pub trait Utf8Data {
	fn utf8_bytes(&self) -> &[u8];
}

impl Utf8Data for String {
	fn utf8_bytes(&self) -> &[u8] {
		self.as_bytes()
	}
}

impl<'a> Utf8Data for &'a str {
	fn utf8_bytes(&self) -> &[u8] {
		self.as_bytes()
	}
}

impl<T: Utf8Data> IntoTensorElementDataType for T {
	fn tensor_element_data_type() -> TensorElementDataType {
		TensorElementDataType::String
	}

	fn try_utf8_bytes(&self) -> Option<&[u8]> {
		Some(self.utf8_bytes())
	}
}

/// Trait used to map ONNX Runtime types to Rust types.
pub trait TensorDataToType: Sized + fmt::Debug + Clone {
	/// The tensor element type that this type can extract from.
	fn tensor_element_data_type() -> TensorElementDataType;

	/// Extract an `ArrayView` from the ORT-owned tensor.
	fn extract_array<'t, D>(shape: D, tensor: *mut sys::OrtValue) -> OrtResult<ndarray::ArrayView<'t, Self, D>>
	where
		D: ndarray::Dimension;
}

/// Implements `TensorDataToType` for primitives which can use `GetTensorMutableData`.
macro_rules! impl_prim_type_from_ort_trait {
	($type_: ty, $variant: ident) => {
		impl TensorDataToType for $type_ {
			fn tensor_element_data_type() -> TensorElementDataType {
				TensorElementDataType::$variant
			}

			fn extract_array<'t, D>(shape: D, tensor: *mut sys::OrtValue) -> OrtResult<ndarray::ArrayView<'t, Self, D>>
			where
				D: ndarray::Dimension
			{
				extract_primitive_array(shape, tensor)
			}
		}
	};
}

/// Construct an [`ndarray::ArrayView`] for an ORT tensor.
///
/// Only to be used on types whose Rust in-memory representation matches ONNX Runtime's (e.g. primitive numeric types
/// like u32)
fn extract_primitive_array<'t, D, T: TensorDataToType>(shape: D, tensor: *mut sys::OrtValue) -> OrtResult<ndarray::ArrayView<'t, T, D>>
where
	D: ndarray::Dimension
{
	// Get pointer to output tensor values
	let mut output_array_ptr: *mut T = ptr::null_mut();
	let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
	let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr as *mut *mut std::ffi::c_void;
	unsafe { super::error::call_ort(|ort| ort.GetTensorMutableData.unwrap()(tensor, output_array_ptr_ptr_void)) }.map_err(OrtError::GetTensorMutableData)?;
	assert_non_null_pointer(output_array_ptr, "GetTensorMutableData")?;

	let array_view = unsafe { ndarray::ArrayView::from_shape_ptr(shape, output_array_ptr) };
	Ok(array_view)
}

impl_prim_type_from_ort_trait!(f32, Float32);
impl_prim_type_from_ort_trait!(f64, Float64);
impl_prim_type_from_ort_trait!(u8, Uint8);
impl_prim_type_from_ort_trait!(u16, Uint16);
impl_prim_type_from_ort_trait!(u32, Uint32);
impl_prim_type_from_ort_trait!(u64, Uint64);
impl_prim_type_from_ort_trait!(i8, Int8);
impl_prim_type_from_ort_trait!(i16, Int16);
impl_prim_type_from_ort_trait!(i32, Int32);
impl_prim_type_from_ort_trait!(i64, Int64);
