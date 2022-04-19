pub mod download;
pub mod environment;
pub mod error;
pub mod memory;
pub mod session;
pub mod sys;
pub mod tensor;

use std::{
	ffi::CStr,
	sync::{atomic::AtomicPtr, Arc, Mutex}
};

pub use error::{OrtApiError, OrtError, Result as OrtResult};
use lazy_static::lazy_static;

use self::sys::OnnxEnumInt;

#[macro_export]
#[doc(hidden)]
macro_rules! extern_system_fn {
	($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "C" fn $($tt)*);
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "C" fn $($tt)*);
}

lazy_static! {
	pub(crate) static ref G_ORT_API: Arc<Mutex<AtomicPtr<sys::OrtApi>>> = {
		let base: *const sys::OrtApiBase = unsafe { sys::OrtGetApiBase() };
		assert_ne!(base, std::ptr::null());
		let get_api: extern_system_fn! { unsafe fn(u32) -> *const sys::OrtApi } = unsafe { (*base).GetApi.unwrap() };
		let api: *const sys::OrtApi = unsafe { get_api(sys::ORT_API_VERSION) };
		Arc::new(Mutex::new(AtomicPtr::new(api as *mut sys::OrtApi)))
	};
}

pub fn ort() -> sys::OrtApi {
	let mut api_ref = G_ORT_API.lock().expect("failed to acquire OrtApi lock; another thread panicked?");
	let api_ref_mut: &mut *mut sys::OrtApi = api_ref.get_mut();
	let api_ptr_mut: *mut sys::OrtApi = *api_ref_mut;

	assert_ne!(api_ptr_mut, std::ptr::null_mut());

	unsafe { *api_ptr_mut }
}

pub(crate) fn char_p_to_string(raw: *const i8) -> OrtResult<String> {
	let c_string = unsafe { std::ffi::CStr::from_ptr(raw as *mut i8).to_owned() };
	match c_string.into_string() {
		Ok(string) => Ok(string),
		Err(e) => Err(OrtApiError::IntoStringError(e))
	}
	.map_err(OrtError::StringConversion)
}

/// ONNX's logger sends the code location where the log occurred, which will be parsed into this struct.
#[derive(Debug)]
struct CodeLocation<'a> {
	file: &'a str,
	line: &'a str,
	function: &'a str
}

impl<'a> From<&'a str> for CodeLocation<'a> {
	fn from(code_location: &'a str) -> Self {
		let mut splitter = code_location.split(' ');
		let file_and_line = splitter.next().unwrap_or("<unknown file>:<unknown line>");
		let function = splitter.next().unwrap_or("<unknown function>");
		let mut file_and_line_splitter = file_and_line.split(':');
		let file = file_and_line_splitter.next().unwrap_or("<unknown file>");
		let line = file_and_line_splitter.next().unwrap_or("<unknown line>");

		CodeLocation { file, line, function }
	}
}

extern_system_fn! {
	/// Callback from C that will handle ONNX logging, forwarding ONNX's logs to the `tracing` crate.
	pub(crate) fn custom_logger(_params: *mut std::ffi::c_void, severity: sys::OrtLoggingLevel, category: *const i8, log_id: *const i8, code_location: *const i8, message: *const i8) {
		use tracing::{span, Level, trace, debug, warn, info, error};

		let log_level = match severity {
			sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE => Level::TRACE,
			sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO => Level::DEBUG,
			sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING => Level::INFO,
			sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR => Level::WARN,
			sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL => Level::ERROR
		};

		assert_ne!(category, std::ptr::null());
		let category = unsafe { CStr::from_ptr(category) };
		assert_ne!(code_location, std::ptr::null());
		let code_location = unsafe { CStr::from_ptr(code_location) }.to_str().unwrap_or("unknown");
		assert_ne!(message, std::ptr::null());
		let message = unsafe { CStr::from_ptr(message) };
		assert_ne!(log_id, std::ptr::null());
		let log_id = unsafe { CStr::from_ptr(log_id) };

		let code_location = CodeLocation::from(code_location);
		let span = span!(
			Level::TRACE,
			"onnxruntime",
			category = category.to_str().unwrap_or("<unknown>"),
			file = code_location.file,
			line = code_location.line,
			function = code_location.function,
			log_id = log_id.to_str().unwrap_or("<unknown>")
		);
		let _enter = span.enter();

		match log_level {
			Level::TRACE => trace!("{:?}", message),
			Level::DEBUG => debug!("{:?}", message),
			Level::INFO => info!("{:?}", message),
			Level::WARN => warn!("{:?}", message),
			Level::ERROR => error!("{:?}", message)
		}
	}
}

/// ONNX Runtime logging level.
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum LoggingLevel {
	Verbose = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE as OnnxEnumInt,
	Info = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO as OnnxEnumInt,
	Warning = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING as OnnxEnumInt,
	Error = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR as OnnxEnumInt,
	Fatal = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL as OnnxEnumInt
}

impl From<LoggingLevel> for sys::OrtLoggingLevel {
	fn from(logging_level: LoggingLevel) -> Self {
		match logging_level {
			LoggingLevel::Verbose => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
			LoggingLevel::Info => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
			LoggingLevel::Warning => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
			LoggingLevel::Error => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
			LoggingLevel::Fatal => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL
		}
	}
}

/// ONNX Runtime provides various graph optimizations to improve performance. Graph optimizations are essentially
/// graph-level transformations, ranging from small graph simplifications and node eliminations to more complex node
/// fusions and layout optimizations.
///
/// Graph optimizations are divided in several categories (or levels) based on their complexity and functionality. They
/// can be performed either online or offline. In online mode, the optimizations are done before performing the
/// inference, while in offline mode, the runtime saves the optimized graph to disk (most commonly used when converting
/// an ONNX model to an ONNX Runtime model).
///
/// The optimizations belonging to one level are performed after the optimizations of the previous level have been
/// applied (e.g., extended optimizations are applied after basic optimizations have been applied).
///
/// **All optimizations are enabled by default.**
///
/// # Online/offline mode
/// All optimizations can be performed either online or offline. In online mode, when initializing an inference session,
/// we also apply all enabled graph optimizations before performing model inference. Applying all optimizations each
/// time we initiate a session can add overhead to the model startup time (especially for complex models), which can be
/// critical in production scenarios. This is where the offline mode can bring a lot of benefit. In offline mode, after
/// performing graph optimizations, ONNX Runtime serializes the resulting model to disk. Subsequently, we can reduce
/// startup time by using the already optimized model and disabling all optimizations.
///
/// ## Notes:
/// - When running in offline mode, make sure to use the exact same options (e.g., execution providers, optimization
///   level) and hardware as the target machine that the model inference will run on (e.g., you cannot run a model
///   pre-optimized for a GPU execution provider on a machine that is equipped only with CPU).
/// - When layout optimizations are enabled, the offline mode can only be used on compatible hardware to the environment
///   when the offline model is saved. For example, if model has layout optimized for AVX2, the offline model would
///   require CPUs that support AVX2.
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum GraphOptimizationLevel {
	Disable = sys::GraphOptimizationLevel::ORT_DISABLE_ALL as OnnxEnumInt,
	/// Level 1 includes semantics-preserving graph rewrites which remove redundant nodes and redundant computation.
	/// They run before graph partitioning and thus apply to all the execution providers. Available basic/level 1 graph
	/// optimizations are as follows:
	///
	/// - Constant Folding: Statically computes parts of the graph that rely only on constant initializers. This
	///   eliminates the need to compute them during runtime.
	/// - Redundant node eliminations: Remove all redundant nodes without changing the graph structure. The following
	///   such optimizations are currently supported:
	///   * Identity Elimination
	///   * Slice Elimination
	///   * Unsqueeze Elimination
	///   * Dropout Elimination
	/// - Semantics-preserving node fusions : Fuse/fold multiple nodes into a single node. For example, Conv Add fusion
	///   folds the Add operator as the bias of the Conv operator. The following such optimizations are currently
	///   supported:
	///   * Conv Add Fusion
	///   * Conv Mul Fusion
	///   * Conv BatchNorm Fusion
	///   * Relu Clip Fusion
	///   * Reshape Fusion
	Level1 = sys::GraphOptimizationLevel::ORT_ENABLE_BASIC as OnnxEnumInt,
	#[rustfmt::skip]
	/// Level 2 optimizations include complex node fusions. They are run after graph partitioning and are only applied to
	/// the nodes assigned to the CPU or CUDA execution provider. Available extended/level 2 graph optimizations are as follows:
	///
	/// | Optimization                    | EPs       | Comments                                                                       |
	/// |:------------------------------- |:--------- |:------------------------------------------------------------------------------ |
	/// | GEMM Activation Fusion          | CPU       |                                                                                |
	/// | Matmul Add Fusion               | CPU       |                                                                                |
	/// | Conv Activation Fusion          | CPU       |                                                                                |
	/// | GELU Fusion                     | CPU, CUDA |                                                                                |
	/// | Layer Normalization Fusion      | CPU, CUDA |                                                                                |
	/// | BERT Embedding Layer Fusion     | CPU, CUDA | Fuses BERT embedding layers, layer normalization, & attention mask length      |
	/// | Attention Fusion*               | CPU, CUDA |                                                                                |
	/// | Skip Layer Normalization Fusion | CPU, CUDA | Fuse bias of fully connected layers, skip connections, and layer normalization |
	/// | Bias GELU Fusion                | CPU, CUDA | Fuse bias of fully connected layers & GELU activation                          |
	/// | GELU Approximation*             | CUDA      | Disabled by default; enable with `OrtSessionOptions::EnableGeluApproximation`  |
	///
	/// > **NOTE**: To optimize performance of the BERT model, approximation is used in GELU Approximation and Attention
	/// Fusion for the CUDA execution provider. The impact on accuracy is negligible based on our evaluation; F1 score
	/// for a BERT model on SQuAD v1.1 is almost the same (87.05 vs 87.03).
	Level2 = sys::GraphOptimizationLevel::ORT_ENABLE_EXTENDED as OnnxEnumInt,
	/// Level 3 optimizations include memory layout optimizations, which may optimize the graph to use the NCHWc memory
	/// layout rather than NCHW to improve spatial locality for some targets.
	Level3 = sys::GraphOptimizationLevel::ORT_ENABLE_ALL as OnnxEnumInt
}

impl From<GraphOptimizationLevel> for sys::GraphOptimizationLevel {
	fn from(val: GraphOptimizationLevel) -> Self {
		match val {
			GraphOptimizationLevel::Disable => sys::GraphOptimizationLevel::ORT_DISABLE_ALL,
			GraphOptimizationLevel::Level1 => sys::GraphOptimizationLevel::ORT_ENABLE_BASIC,
			GraphOptimizationLevel::Level2 => sys::GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
			GraphOptimizationLevel::Level3 => sys::GraphOptimizationLevel::ORT_ENABLE_ALL
		}
	}
}

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

/// Allocator type
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum AllocatorType {
	// Invalid = sys::OrtAllocatorType::Invalid as i32,
	/// Device allocator
	Device = sys::OrtAllocatorType::OrtDeviceAllocator as i32,
	/// Arena allocator
	Arena = sys::OrtAllocatorType::OrtArenaAllocator as i32
}

impl From<AllocatorType> for sys::OrtAllocatorType {
	fn from(val: AllocatorType) -> Self {
		use AllocatorType::*;
		match val {
			// Invalid => sys::OrtAllocatorType::Invalid,
			Device => sys::OrtAllocatorType::OrtDeviceAllocator,
			Arena => sys::OrtAllocatorType::OrtArenaAllocator
		}
	}
}

/// Memory type
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum MemType {
	CPUInput = sys::OrtMemType::OrtMemTypeCPUInput as i32,
	CPUOutput = sys::OrtMemType::OrtMemTypeCPUOutput as i32,
	/// Default memory type
	Default = sys::OrtMemType::OrtMemTypeDefault as i32
}

impl MemType {
	pub const CPU: MemType = MemType::CPUOutput;
}

impl From<MemType> for sys::OrtMemType {
	fn from(val: MemType) -> Self {
		use MemType::*;
		match val {
			CPUInput => sys::OrtMemType::OrtMemTypeCPUInput,
			CPUOutput => sys::OrtMemType::OrtMemTypeCPUOutput,
			Default => sys::OrtMemType::OrtMemTypeDefault
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_char_p_to_string() {
		let s = std::ffi::CString::new("foo").unwrap();
		let ptr = s.as_c_str().as_ptr();
		assert_eq!("foo", char_p_to_string(ptr).unwrap());
	}
}
