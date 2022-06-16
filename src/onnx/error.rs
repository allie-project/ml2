use std::{io, path::PathBuf};

use thiserror::Error;

use super::sys;
use super::{char_p_to_string, ort};

/// Type alias for the result returned by ORT functions.
pub type Result<T> = std::result::Result<T, OrtError>;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum OrtError {
	#[error("Failed to construct String")]
	StringConversion(OrtApiError),
	/// An error occurred while creating an ONNX environment.
	#[error("Failed to create ONNX environment: {0}")]
	Environment(OrtApiError),
	/// Error occurred when creating ONNX session options.
	#[error("Failed to create ONNX session options: {0}")]
	SessionOptions(OrtApiError),
	/// Error occurred when creating an ONNX session.
	#[error("Failed to create ONNX session: {0}")]
	Session(OrtApiError),
	/// Error occurred when creating an ONNX allocator.
	#[error("Failed to get ONNX allocator: {0}")]
	Allocator(OrtApiError),
	/// Error occurred when counting ONNX session input/output count.
	#[error("Failed to get input or output count: {0}")]
	InOutCount(OrtApiError),
	/// Error occurred when getting ONNX input name.
	#[error("Failed to get input name: {0}")]
	InputName(OrtApiError),
	/// Error occurred when getting ONNX type information
	#[error("Failed to get type info: {0}")]
	GetTypeInfo(OrtApiError),
	/// Error occurred when casting ONNX type information to tensor information
	#[error("Failed to cast type info to tensor info: {0}")]
	CastTypeInfoToTensorInfo(OrtApiError),
	/// Error occurred when getting tensor elements type
	#[error("Failed to get tensor element type: {0}")]
	TensorElementType(OrtApiError),
	/// Error occurred when getting ONNX dimensions count
	#[error("Failed to get dimensions count: {0}")]
	GetDimensionsCount(OrtApiError),
	/// Error occurred when getting ONNX dimensions
	#[error("Failed to get dimensions: {0}")]
	GetDimensions(OrtApiError),
	/// Error occurred when creating CPU memory information
	#[error("Failed to create CPU memory info: {0}")]
	CreateCpuMemoryInfo(OrtApiError),
	/// Error occurred when creating ONNX tensor
	#[error("Failed to create tensor: {0}")]
	CreateTensor(OrtApiError),
	/// Error occurred when creating ONNX tensor with specific data
	#[error("Failed to create tensor with data: {0}")]
	CreateTensorWithData(OrtApiError),
	/// Error occurred when filling a tensor with string data
	#[error("Failed to fill string tensor: {0}")]
	FillStringTensor(OrtApiError),
	/// Error occurred when checking if ONNX tensor was properly initialized
	#[error("Failed to check if tensor is a tensor or was properly initialized: {0}")]
	IsTensor(OrtApiError),
	/// Error occurred when getting tensor type and shape
	#[error("Failed to get tensor type and shape: {0}")]
	GetTensorTypeAndShape(OrtApiError),
	/// Error occurred when ONNX inference operation was called
	#[error("Failed to run: {0}")]
	Run(OrtApiError),
	/// Error occurred when extracting data from an ONNX tensor into an C array to be used as an `ndarray::ArrayView`
	#[error("Failed to get tensor data: {0}")]
	GetTensorMutableData(OrtApiError),
	/// Error occurred when downloading a pre-trained ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models)
	#[error("Failed to download ONNX model: {0}")]
	DownloadError(#[from] OrtDownloadError),
	/// Dimensions of input data and ONNX model loaded from file do not match
	#[error("Dimensions do not match: {0:?}")]
	NonMatchingDimensions(NonMatchingDimensionsError),
	/// File does not exist
	#[error("File {filename:?} does not exist")]
	FileDoesNotExists {
		/// Path which does not exists
		filename: PathBuf
	},
	/// Path is invalid UTF-8
	#[error("Path {path:?} cannot be converted to UTF-8")]
	NonUtf8Path {
		/// Path with invalid UTF-8
		path: PathBuf
	},
	/// Attempt to build a Rust `CString` from a null pointer
	#[error("Failed to build CString when original contains null: {0}")]
	CStringNulError(#[from] std::ffi::NulError),
	#[error("{0} pointer should be null")]
	/// ORT pointer should have been null
	PointerShouldBeNull(String),
	/// ORT pointer should not have been null
	#[error("{0} pointer should not be null")]
	PointerShouldNotBeNull(String),
	/// ONNX Model has invalid dimensions.
	#[error("Invalid dimensions")]
	InvalidDimensions,
	/// The runtime type was undefined.
	#[error("Undefined tensor element type")]
	UndefinedTensorElementType,
	/// Error occurred when checking if ONNX tensor was properly initialized
	#[error("Failed to check if tensor is a tensor or was properly initialized")]
	IsTensorCheck,
	/// Could not retrieve model metadata.
	#[error("Failed to retrieve model metadata: {0}")]
	GetModelMetadata(OrtApiError)
}

/// Error used when input dimensions defined in the model and passed from an inference call do not match.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum NonMatchingDimensionsError {
	/// Number of inputs from model does not match number of inputs from inference call
	#[error(
		"Non-matching number of inputs: {inference_input_count:?} provided vs {model_input_count:?} for model (inputs: {inference_input:?}, model: {model_input:?})"
	)]
	InputsCount {
		/// Number of input dimensions used by inference call
		inference_input_count: usize,
		/// Number of input dimensions defined in model
		model_input_count: usize,
		/// Input dimensions used by inference call
		inference_input: Vec<Vec<usize>>,
		/// Input dimensions defined in model
		model_input: Vec<Vec<Option<u32>>>
	},
	/// Inputs length from model does not match the expected input from inference call
	#[error("Different input lengths: Expected input: {model_input:?}, received input: {inference_input:?}")]
	InputsLength {
		/// Input dimensions used by inference call
		inference_input: Vec<Vec<usize>>,
		/// Input dimensions defined in model
		model_input: Vec<Vec<Option<u32>>>
	}
}

/// Error details when ONNX C API returns an error.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum OrtApiError {
	/// Details about the error.
	#[error("Error calling ONNX: {0}")]
	Msg(String),
	/// Details as reported by the ONNX C API in case the conversion to UTF-8 failed.
	#[error("Error calling ONNX function; failed to convert the error message to UTF-8")]
	IntoStringError(std::ffi::IntoStringError)
}

/// Error from downloading pre-trained model from the [ONNX Model Zoo](https://github.com/onnx/models).
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum OrtDownloadError {
	/// Generic input/output error
	#[error("Error reading file: {0}")]
	IoError(#[from] io::Error),
	/// Download error by ureq
	#[cfg(feature = "onnx-fetch-models")]
	#[error("Error downloading to file: {0}")]
	UreqError(#[from] Box<ureq::Error>),
	/// Error getting Content-Length from HTTP GET request.
	#[error("Error getting Content-Length from HTTP GET")]
	ContentLengthError,
	/// Mismatch between amount of downloaded and expected bytes.
	#[error("Error copying data to file: expected {expected} length, but got {io}")]
	CopyError {
		/// Expected amount of bytes to download
		expected: u64,
		/// Number of bytes read from network and written to file
		io: u64
	}
}

/// Wrapper type around ONNX's `OrtStatus` pointer.
///
/// This wrapper exists to facilitate conversion from C raw pointers to Rust error types.
pub struct OrtStatusWrapper(*mut sys::OrtStatus);

impl From<*mut sys::OrtStatus> for OrtStatusWrapper {
	fn from(status: *mut sys::OrtStatus) -> Self {
		OrtStatusWrapper(status)
	}
}

pub(crate) fn assert_null_pointer<T>(ptr: *const T, name: &str) -> Result<()> {
	ptr.is_null().then(|| ()).ok_or_else(|| OrtError::PointerShouldBeNull(name.to_owned()))
}

pub(crate) fn assert_non_null_pointer<T>(ptr: *const T, name: &str) -> Result<()> {
	(!ptr.is_null()).then(|| ()).ok_or_else(|| OrtError::PointerShouldBeNull(name.to_owned()))
}

impl From<OrtStatusWrapper> for std::result::Result<(), OrtApiError> {
	fn from(status: OrtStatusWrapper) -> Self {
		if status.0.is_null() {
			Ok(())
		} else {
			let raw: *const i8 = unsafe { ort().GetErrorMessage.unwrap()(status.0) };
			match char_p_to_string(raw) {
				Ok(msg) => Err(OrtApiError::Msg(msg)),
				Err(err) => match err {
					OrtError::StringConversion(OrtApiError::IntoStringError(e)) => Err(OrtApiError::IntoStringError(e)),
					_ => unreachable!()
				}
			}
		}
	}
}

impl Drop for OrtStatusWrapper {
	fn drop(&mut self) {
		unsafe { ort().ReleaseStatus.unwrap()(self.0) }
	}
}

pub(crate) fn status_to_result(status: *mut sys::OrtStatus) -> std::result::Result<(), OrtApiError> {
	let status_wrapper: OrtStatusWrapper = status.into();
	status_wrapper.into()
}

/// A wrapper around an OrtApi function that maps the status code into an [`OrtApiError`].
///
/// [`OrtApiError`]: enum.OrtApiError.html
pub(crate) unsafe fn call_ort<F>(mut f: F) -> std::result::Result<(), OrtApiError>
where
	F: FnMut(sys::OrtApi) -> *mut sys::OrtStatus
{
	status_to_result(f(ort()))
}
