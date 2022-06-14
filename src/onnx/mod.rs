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
