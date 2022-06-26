#![allow(clippy::tabs_in_doc_comments)]

#[cfg(feature = "onnx-fetch-models")]
use std::env;
#[cfg(not(target_family = "windows"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;
use std::{
	ffi::CString,
	fmt::{self, Debug},
	os::raw::c_char,
	path::Path
};

use ndarray::Array;
use tracing::{debug, error};

use super::{
	char_p_to_string,
	environment::Environment,
	error::{assert_non_null_pointer, assert_null_pointer, status_to_result, NonMatchingDimensionsError, OrtApiError, OrtError, OrtResult},
	execution_providers::apply_execution_providers,
	extern_system_fn,
	memory::MemoryInfo,
	metadata::Metadata,
	ort, ortsys, sys,
	tensor::{ort_owned_tensor::OrtOwnedTensor, IntoTensorElementDataType, OrtTensor, TensorDataToType, TensorElementDataType},
	AllocatorType, ExecutionProviderOptions, GraphOptimizationLevel, MemType
};
#[cfg(feature = "onnx-fetch-models")]
use super::{download::OnnxModel, error::OrtDownloadError};

/// Type used to create a session using the _builder pattern_.
///
/// A `SessionBuilder` is created by calling the [`Environment::session()`] method on the environment.
///
/// Once created, you can use the different methods to configure the session.
///
/// Once configured, use the [`SessionBuilder::with_model_from_file()`] method to "commit" the builder configuration
/// into a [`Session`].
///
/// # Example
///
/// ```no_run
/// # use std::error::Error;
/// # use ml2::onnx::{Environment, LoggingLevel, GraphOptimizationLevel};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let environment = Environment::builder().with_name("test").with_log_level(LoggingLevel::Verbose).build()?;
/// let mut session = environment
/// 	.session()?
/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
/// 	.with_intra_threads(1)?
/// 	.with_model_from_file("squeezenet.onnx")?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder<'a, 'b> {
	env: &'a Environment<'b>,
	session_options_ptr: *mut sys::OrtSessionOptions,

	allocator: AllocatorType,
	memory_type: MemType,
	execution_providers: Option<ExecutionProviderOptions<'a>>
}

impl<'a, 'b> Debug for SessionBuilder<'a, 'b> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
		f.debug_struct("SessionBuilder")
			.field("env", &self.env.name())
			.field("allocator", &self.allocator)
			.field("memory_type", &self.memory_type)
			.finish()
	}
}

impl<'a, 'b> Drop for SessionBuilder<'a, 'b> {
	#[tracing::instrument]
	fn drop(&mut self) {
		if self.session_options_ptr.is_null() {
			error!("Session options pointer is null, not dropping");
		} else {
			debug!("Dropping the session options.");
			ortsys![unsafe ReleaseSessionOptions(self.session_options_ptr)];
		}
	}
}

impl<'a, 'b> SessionBuilder<'a, 'b> {
	pub(crate) fn new(env: &'a Environment<'b>) -> OrtResult<SessionBuilder<'a, 'b>> {
		let mut session_options_ptr: *mut sys::OrtSessionOptions = std::ptr::null_mut();
		ortsys![unsafe CreateSessionOptions(&mut session_options_ptr) -> OrtError::CreateSessionOptions; nonNull(session_options_ptr)];

		Ok(SessionBuilder {
			env,
			session_options_ptr,
			allocator: AllocatorType::Arena,
			memory_type: MemType::Default,
			execution_providers: None
		})
	}

	/// Configures a list of execution providers to attempt to use for the session.
	///
	/// Execution providers are loaded in the order they are provided until a suitable execution provider is found. Most
	/// execution providers will silently fail if they are unavailable or misconfigured (see notes below), however, some
	/// may log to the console, which is sadly unavoidable. The CPU execution provider is always available, so always
	/// put it last in the list (though it is not required).
	///
	/// Execution providers will only work if the corresponding `onnxep-*` feature is enabled and ONNX Runtime was built
	/// with support for the corresponding execution provider. Execution providers that do not have their corresponding
	/// feature enabled are currently ignored.
	///
	/// Execution provider options can be specified in the second argument. Refer to ONNX Runtime's
	/// [execution provider docs](https://onnxruntime.ai/docs/execution-providers/) for configuration options. In most
	/// cases, passing `None` to configure with no options is suitable.
	///
	/// It is recommended to enable the `cuda` EP for x86 platforms and the `acl` EP for ARM platforms for the best
	/// performance, though this does mean you'll have to build ONNX Runtime for these targets. Microsoft's prebuilt
	/// binaries are built with CUDA and TensorRT support, if you built ml2 with the `onnxep-cuda` or `onnxep-tensorrt`
	/// features enabled.
	///
	/// Supported execution providers:
	/// - `cpu`: Default CPU/MLAS execution provider. Available on all platforms.
	/// - `acl`: Arm Compute Library
	/// - `cuda`: NVIDIA CUDA/cuDNN
	/// - `tensorrt`: NVIDIA TensorRT
	///
	/// ## Notes
	///
	/// - **Use of [`SessionBuilder::with_execution_providers`] in a library is discouraged.** Execution providers
	///   should always be configurable by the user, in case an execution provider is misconfigured and causes the
	///   application to crash (see notes below). Instead, your library should accept an [`Environment`] from the user
	///   rather than creating its own. This way, the user can configure execution providers for **all** modules that
	///   use it.
	/// - Using the CUDA/TensorRT execution providers **can terminate the process if the CUDA/TensorRT installation is
	///   misconfigured**. Configuring the execution provider will seem to work, but when you attempt to run a session,
	///   it will hard crash the process with a "stack buffer overrun" error. This can occur when CUDA/TensorRT is
	///   missing a DLL such as `zlibwapi.dll`. To prevent your app from crashing, you can check to see if you can load
	///   `zlibwapi.dll` before enabling the CUDA/TensorRT execution providers.
	pub fn with_execution_providers(mut self, execution_providers: ExecutionProviderOptions<'a>) -> OrtResult<SessionBuilder<'a, 'b>> {
		self.execution_providers.replace(execution_providers);
		Ok(self)
	}

	/// Configure the session to use a number of threads to parallelize the execution within nodes. If ONNX Runtime was
	/// built with OpenMP (as is the case with Microsoft's prebuilt binaries), this will have no effect on the number of
	/// threads used. Instead, you can configure the number of threads OpenMP uses via the `OMP_NUM_THREADS` environment
	/// variable.
	///
	/// For configuring the number of threads used when the session execution mode is set to `Parallel`, see
	/// [`SessionBuilder::with_inter_threads()`].
	pub fn with_intra_threads(self, num_threads: i16) -> OrtResult<SessionBuilder<'a, 'b>> {
		// We use a u16 in the builder to cover the 16-bits positive values of a i32.
		let num_threads = num_threads as i32;
		ortsys![unsafe SetIntraOpNumThreads(self.session_options_ptr, num_threads) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Configure the session to use a number of threads to parallelize the execution of the graph. If nodes can be run
	/// in parallel, this sets the maximum number of threads to use to run them in parallel.
	///
	/// This has no effect when the session execution mode is set to `Sequential`.
	///
	/// For configuring the number of threads used to parallelize the execution within nodes, see
	/// [`SessionBuilder::with_intra_threads()`].
	pub fn with_inter_threads(self, num_threads: i16) -> OrtResult<SessionBuilder<'a, 'b>> {
		// We use a u16 in the builder to cover the 16-bits positive values of a i32.
		let num_threads = num_threads as i32;
		ortsys![unsafe SetInterOpNumThreads(self.session_options_ptr, num_threads) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Enable/disable the parallel execution mode for this session. By default, this is disabled.
	///
	/// Parallel execution can improve performance for models with many branches, at the cost of higher memory usage.
	/// You can configure the amount of threads used to parallelize the execution of the graph via
	/// [`SessionBuilder::with_inter_threads()`].
	pub fn with_parallel_execution(self, parallel_execution: bool) -> OrtResult<SessionBuilder<'a, 'b>> {
		let execution_mode = if parallel_execution {
			sys::ExecutionMode::ORT_PARALLEL
		} else {
			sys::ExecutionMode::ORT_SEQUENTIAL
		};
		ortsys![unsafe SetSessionExecutionMode(self.session_options_ptr, execution_mode) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Set the session's optimization level. See [`GraphOptimizationLevel`] for more information on the different
	/// optimization levels.
	pub fn with_optimization_level(self, opt_level: GraphOptimizationLevel) -> OrtResult<SessionBuilder<'a, 'b>> {
		ortsys![unsafe SetSessionGraphOptimizationLevel(self.session_options_ptr, opt_level.into()) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Set the session's allocator. Defaults to [`AllocatorType::Arena`].
	pub fn with_allocator(mut self, allocator: AllocatorType) -> OrtResult<SessionBuilder<'a, 'b>> {
		self.allocator = allocator;
		Ok(self)
	}

	/// Set the session's memory type. Defaults to [`MemType::Default`].
	pub fn with_memory_type(mut self, memory_type: MemType) -> OrtResult<SessionBuilder<'a, 'b>> {
		self.memory_type = memory_type;
		Ok(self)
	}

	/// Downloads a pre-trained ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models) and builds the session.
	#[cfg(feature = "onnx-fetch-models")]
	pub fn with_model_downloaded<M>(self, model: M) -> OrtResult<Session<'a, 'b>>
	where
		M: Into<OnnxModel>
	{
		self.with_model_downloaded_monomorphized(model.into())
	}

	#[cfg(feature = "onnx-fetch-models")]
	fn with_model_downloaded_monomorphized(self, model: OnnxModel) -> OrtResult<Session<'a, 'b>> {
		let download_dir = env::current_dir().map_err(OrtDownloadError::IoError)?;
		let downloaded_path = model.download_to(download_dir)?;
		self.with_model_from_file(downloaded_path)
	}

	// TODO: Add all functions changing the options.
	//       See all OrtApi methods taking a `options: *mut OrtSessionOptions`.

	/// Loads an ONNX model from a file and builds the session.
	pub fn with_model_from_file<P>(self, model_filepath_ref: P) -> OrtResult<Session<'a, 'b>>
	where
		P: AsRef<Path> + 'a
	{
		let model_filepath = model_filepath_ref.as_ref();
		if !model_filepath.exists() {
			return Err(OrtError::FileDoesNotExist {
				filename: model_filepath.to_path_buf()
			});
		}

		// Build an OsString, then a vector of bytes to pass to C
		let model_path = std::ffi::OsString::from(model_filepath);
		#[cfg(target_family = "windows")]
		let model_path: Vec<u16> = model_path
            .encode_wide()
            .chain(std::iter::once(0)) // Make sure we have a null terminated string
            .collect();
		#[cfg(not(target_family = "windows"))]
		let model_path: Vec<std::os::raw::c_char> = model_path
            .as_bytes()
            .iter()
            .chain(std::iter::once(&b'\0')) // Make sure we have a null terminated string
            .map(|b| *b as std::os::raw::c_char)
            .collect();

		let execution_providers = (&self.env.execution_providers).as_ref().or(self.execution_providers.as_ref());
		if let Some(execution_providers) = execution_providers {
			apply_execution_providers(self.session_options_ptr, execution_providers);
		}

		let env_ptr: *const sys::OrtEnv = self.env.env_ptr();

		let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();
		ortsys![unsafe CreateSession(env_ptr, model_path.as_ptr(), self.session_options_ptr, &mut session_ptr) -> OrtError::CreateSession; nonNull(session_ptr)];

		let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
		ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr) -> OrtError::GetAllocator; nonNull(allocator_ptr)];

		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default)?;

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Output>>>()?;

		Ok(Session {
			env: self.env,
			session_ptr,
			allocator_ptr,
			memory_info,
			inputs,
			outputs
		})
	}

	/// Load an ONNX graph from memory and commit the session
	pub fn with_model_from_memory<B>(self, model_bytes: B) -> OrtResult<Session<'a, 'b>>
	where
		B: AsRef<[u8]>
	{
		self.with_model_from_memory_monomorphized(model_bytes.as_ref())
	}

	fn with_model_from_memory_monomorphized(self, model_bytes: &[u8]) -> OrtResult<Session<'a, 'b>> {
		let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

		let env_ptr: *const sys::OrtEnv = self.env.env_ptr();

		let execution_providers = (&self.env.execution_providers).as_ref().or(self.execution_providers.as_ref());
		if let Some(execution_providers) = execution_providers {
			apply_execution_providers(self.session_options_ptr, execution_providers);
		}

		let model_data = model_bytes.as_ptr() as *const std::ffi::c_void;
		let model_data_length = model_bytes.len();
		ortsys![
			unsafe CreateSessionFromArray(env_ptr, model_data, model_data_length, self.session_options_ptr, &mut session_ptr) -> OrtError::CreateSession;
			nonNull(session_ptr)
		];

		let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
		ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr) -> OrtError::GetAllocator; nonNull(allocator_ptr)];

		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default)?;

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Output>>>()?;

		Ok(Session {
			env: self.env,
			session_ptr,
			allocator_ptr,
			memory_info,
			inputs,
			outputs
		})
	}
}

/// Type storing the session information, built from an [`Environment`](environment/struct.Environment.html)
#[derive(Debug)]
pub struct Session<'a, 'b> {
	#[allow(dead_code)]
	env: &'a Environment<'b>,
	session_ptr: *mut sys::OrtSession,
	allocator_ptr: *mut sys::OrtAllocator,
	memory_info: MemoryInfo,
	/// Information about the ONNX's inputs as stored in loaded file
	pub inputs: Vec<Input>,
	/// Information about the ONNX's outputs as stored in loaded file
	pub outputs: Vec<Output>
}

/// Information about an ONNX's input as stored in loaded file
#[derive(Debug)]
pub struct Input {
	/// Name of the input layer
	pub name: String,
	/// Type of the input layer's elements
	pub input_type: TensorElementDataType,
	/// Shape of the input layer
	///
	/// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
	pub dimensions: Vec<Option<u32>>
}

/// Information about an ONNX's output as stored in loaded file
#[derive(Debug)]
pub struct Output {
	/// Name of the output layer
	pub name: String,
	/// Type of the output layer's elements
	pub output_type: TensorElementDataType,
	/// Shape of the output layer
	///
	/// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
	pub dimensions: Vec<Option<u32>>
}

impl Input {
	/// Return an iterator over the shape elements of the input layer
	///
	/// Note: The member [`Input::dimensions`](struct.Input.html#structfield.dimensions)
	/// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
	/// iterator converts to `usize`.
	pub fn dimensions(&self) -> impl Iterator<Item = Option<usize>> + '_ {
		self.dimensions.iter().map(|d| d.map(|d2| d2 as usize))
	}
}

impl Output {
	/// Return an iterator over the shape elements of the output layer
	///
	/// Note: The member [`Output::dimensions`](struct.Output.html#structfield.dimensions)
	/// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
	/// iterator converts to `usize`.
	pub fn dimensions(&self) -> impl Iterator<Item = Option<usize>> + '_ {
		self.dimensions.iter().map(|d| d.map(|d2| d2 as usize))
	}
}

impl<'a, 'b> Drop for Session<'a, 'b> {
	#[tracing::instrument]
	fn drop(&mut self) {
		debug!("Dropping the session.");
		if self.session_ptr.is_null() {
			error!("Session pointer is null, not dropping.");
		} else {
			ortsys![unsafe ReleaseSession(self.session_ptr)];
		}

		self.session_ptr = std::ptr::null_mut();
		self.allocator_ptr = std::ptr::null_mut();
	}
}

impl<'a, 'b> Session<'a, 'b> {
	/// Run the input data through the ONNX graph, performing inference.
	///
	/// Note that ONNX models can have multiple inputs; a `Vec<_>` is thus
	/// used for the input data here.
	pub fn run<'s, 't, 'm, TIn, TOut, D>(&'s mut self, input_arrays: Vec<Array<TIn, D>>) -> OrtResult<Vec<OrtOwnedTensor<'t, 'm, TOut, ndarray::IxDyn>>>
	where
		TIn: IntoTensorElementDataType + Debug + Clone,
		TOut: TensorDataToType,
		D: ndarray::Dimension,
		'm: 't, // 'm outlives 't (memory info outlives tensor)
		's: 'm  // 's outlives 'm (session outlives memory info)
	{
		self.validate_input_shapes(&input_arrays)?;

		// Build arguments to Run()

		let input_names_ptr: Vec<*const c_char> = self
			.inputs
			.iter()
			.map(|input| input.name.clone())
			.map(|n| CString::new(n).unwrap())
			.map(|n| n.into_raw() as *const c_char)
			.collect();

		let output_names_cstring: Vec<CString> = self
			.outputs
			.iter()
			.map(|output| output.name.clone())
			.map(|n| CString::new(n).unwrap())
			.collect();
		let output_names_ptr: Vec<*const c_char> = output_names_cstring.iter().map(|n| n.as_ptr() as *const c_char).collect();

		let mut output_tensor_extractors_ptrs: Vec<*mut sys::OrtValue> = vec![std::ptr::null_mut(); self.outputs.len()];

		// The C API expects pointers for the arrays (pointers to C-arrays)
		let input_ort_tensors: Vec<OrtTensor<TIn, D>> = input_arrays
			.into_iter()
			.map(|input_array| OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array))
			.collect::<OrtResult<Vec<OrtTensor<TIn, D>>>>()?;
		let input_ort_values: Vec<*const sys::OrtValue> = input_ort_tensors
			.iter()
			.map(|input_array_ort| input_array_ort.c_ptr as *const sys::OrtValue)
			.collect();

		let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();

		ortsys![
			unsafe Run(
				self.session_ptr,
				run_options_ptr,
				input_names_ptr.as_ptr(),
				input_ort_values.as_ptr(),
				input_ort_values.len(),
				output_names_ptr.as_ptr(),
				output_names_ptr.len(),
				output_tensor_extractors_ptrs.as_mut_ptr()
			) -> OrtError::SessionRun
		];

		let memory_info_ref = &self.memory_info;
		let outputs: OrtResult<Vec<OrtOwnedTensor<TOut, ndarray::Dim<ndarray::IxDynImpl>>>> = output_tensor_extractors_ptrs
			.into_iter()
			.map(|tensor_ptr| {
				let dims = unsafe {
					call_with_tensor_info(tensor_ptr, |tensor_info_ptr| {
						get_tensor_dimensions(tensor_info_ptr).map(|dims| dims.iter().map(|&n| n as usize).collect::<Vec<_>>())
					})
				}?;

				// NOTE: Both tensor and array will point to the same data, nothing is copied.
				// As such, there is no need to free the pointer used to create the ArrayView.
				assert_non_null_pointer(tensor_ptr, "Run")?;

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(tensor_ptr, &mut is_tensor) -> OrtError::FailedTensorCheck];
				assert_eq!(is_tensor, 1);

				let array_view = TOut::extract_array(ndarray::IxDyn(&dims), tensor_ptr)?;

				Ok(OrtOwnedTensor::new(tensor_ptr, array_view, memory_info_ref))
			})
			.collect();

		// Reconvert to CString so drop impl is called and memory is freed
		let cstrings: OrtResult<Vec<CString>> = input_names_ptr
			.into_iter()
			.map(|p| {
				assert_non_null_pointer(p, "c_char for CString")?;
				unsafe { Ok(CString::from_raw(p as *mut c_char)) }
			})
			.collect();
		cstrings?;

		outputs
	}

	// pub fn tensor_from_array<'a, 'b, T, D>(&'a self, array: Array<T, D>) -> Tensor<'b, T, D>
	// where
	//     'a: 'b, // 'a outlives 'b
	// {
	//     Tensor::from_array(self, array)
	// }

	fn validate_input_shapes<TIn, D>(&mut self, input_arrays: &[Array<TIn, D>]) -> OrtResult<()>
	where
		TIn: IntoTensorElementDataType + Debug + Clone,
		D: ndarray::Dimension
	{
		// ******************************************************************
		// FIXME: Properly handle errors here
		// Make sure all dimensions match (except dynamic ones)

		// Verify length of inputs
		if input_arrays.len() != self.inputs.len() {
			error!("Non-matching number of inputs: {} (inference) vs {} (model)", input_arrays.len(), self.inputs.len());
			return Err(OrtError::NonMatchingDimensions(NonMatchingDimensionsError::InputsCount {
				inference_input_count: 0,
				model_input_count: 0,
				inference_input: input_arrays.iter().map(|input_array| input_array.shape().to_vec()).collect(),
				model_input: self.inputs.iter().map(|input| input.dimensions.clone()).collect()
			}));
		}

		// Verify length of each individual inputs
		let inputs_different_length = input_arrays
			.iter()
			.zip(self.inputs.iter())
			.any(|(l, r)| l.shape().len() != r.dimensions.len());
		if inputs_different_length {
			error!("Different input lengths: {:?} vs {:?}", self.inputs, input_arrays);
			return Err(OrtError::NonMatchingDimensions(NonMatchingDimensionsError::InputsLength {
				inference_input: input_arrays.iter().map(|input_array| input_array.shape().to_vec()).collect(),
				model_input: self.inputs.iter().map(|input| input.dimensions.clone()).collect()
			}));
		}

		// Verify shape of each individual inputs
		let inputs_different_shape = input_arrays.iter().zip(self.inputs.iter()).any(|(l, r)| {
			let l_shape = l.shape();
			let r_shape = r.dimensions.as_slice();
			l_shape.iter().zip(r_shape.iter()).any(|(l2, r2)| match r2 {
				Some(r3) => *r3 as usize != *l2,
				None => false // None means dynamic size; in that case shape always match
			})
		});
		if inputs_different_shape {
			error!("Different input lengths: {:?} vs {:?}", self.inputs, input_arrays);
			return Err(OrtError::NonMatchingDimensions(NonMatchingDimensionsError::InputsLength {
				inference_input: input_arrays.iter().map(|input_array| input_array.shape().to_vec()).collect(),
				model_input: self.inputs.iter().map(|input| input.dimensions.clone()).collect()
			}));
		}

		Ok(())
	}

	pub fn metadata(&self) -> OrtResult<Metadata> {
		let mut metadata_ptr: *mut sys::OrtModelMetadata = std::ptr::null_mut();
		ortsys![unsafe SessionGetModelMetadata(self.session_ptr, &mut metadata_ptr) -> OrtError::GetModelMetadata; nonNull(metadata_ptr)];
		Ok(Metadata::new(metadata_ptr, self.allocator_ptr))
	}
}

unsafe fn get_tensor_dimensions(tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo) -> OrtResult<Vec<i64>> {
	let mut num_dims = 0;
	ortsys![GetDimensionsCount(tensor_info_ptr, &mut num_dims) -> OrtError::GetDimensionsCount];
	assert_ne!(num_dims, 0);

	let mut node_dims: Vec<i64> = vec![0; num_dims as usize];
	ortsys![GetDimensions(tensor_info_ptr, node_dims.as_mut_ptr(), num_dims) -> OrtError::GetDimensions];
	Ok(node_dims)
}

unsafe fn extract_data_type(tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo) -> OrtResult<TensorElementDataType> {
	let mut type_sys = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![GetTensorElementType(tensor_info_ptr, &mut type_sys) -> OrtError::GetTensorElementType];
	assert_ne!(type_sys, sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
	// This transmute should be safe since its value is read from GetTensorElementType, which we must trust
	Ok(std::mem::transmute(type_sys))
}

/// Calls the provided closure with the result of `GetTensorTypeAndShape`, deallocating the
/// resulting `*OrtTensorTypeAndShapeInfo` before returning.
unsafe fn call_with_tensor_info<F, T>(tensor_ptr: *const sys::OrtValue, mut f: F) -> OrtResult<T>
where
	F: FnMut(*const sys::OrtTensorTypeAndShapeInfo) -> OrtResult<T>
{
	let mut tensor_info_ptr: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
	ortsys![GetTensorTypeAndShape(tensor_ptr, &mut tensor_info_ptr) -> OrtError::GetTensorTypeAndShape];

	let res = f(tensor_info_ptr);
	ortsys![ReleaseTensorTypeAndShapeInfo(tensor_info_ptr)];
	res
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::with_model_from_file()` method.
mod dangerous {
	use super::*;
	use crate::onnx::{ortfree, tensor::TensorElementDataType};

	pub(super) fn extract_inputs_count(session_ptr: *mut sys::OrtSession) -> OrtResult<usize> {
		let f = ort().SessionGetInputCount.unwrap();
		extract_io_count(f, session_ptr)
	}

	pub(super) fn extract_outputs_count(session_ptr: *mut sys::OrtSession) -> OrtResult<usize> {
		let f = ort().SessionGetOutputCount.unwrap();
		extract_io_count(f, session_ptr)
	}

	fn extract_io_count(
		f: extern_system_fn! { unsafe fn(*const sys::OrtSession, *mut usize) -> *mut sys::OrtStatus },
		session_ptr: *mut sys::OrtSession
	) -> OrtResult<usize> {
		let mut num_nodes: usize = 0;
		let status = unsafe { f(session_ptr, &mut num_nodes) };
		status_to_result(status).map_err(OrtError::GetInOutCount)?;
		assert_null_pointer(status, "SessionStatus")?;
		(num_nodes != 0)
			.then(|| ())
			.ok_or_else(|| OrtError::GetInOutCount(OrtApiError::Msg("No nodes in model".to_owned())))?;
		Ok(num_nodes)
	}

	fn extract_input_name(session_ptr: *mut sys::OrtSession, allocator_ptr: *mut sys::OrtAllocator, i: usize) -> OrtResult<String> {
		let f = ort().SessionGetInputName.unwrap();
		extract_io_name(f, session_ptr, allocator_ptr, i)
	}

	fn extract_output_name(session_ptr: *mut sys::OrtSession, allocator_ptr: *mut sys::OrtAllocator, i: usize) -> OrtResult<String> {
		let f = ort().SessionGetOutputName.unwrap();
		extract_io_name(f, session_ptr, allocator_ptr, i)
	}

	fn extract_io_name(
		f: extern_system_fn! { unsafe fn(
			*const sys::OrtSession,
			usize,
			*mut sys::OrtAllocator,
			*mut *mut c_char,
		) -> *mut sys::OrtStatus },
		session_ptr: *mut sys::OrtSession,
		allocator_ptr: *mut sys::OrtAllocator,
		i: usize
	) -> OrtResult<String> {
		let mut name_bytes: *mut c_char = std::ptr::null_mut();

		let status = unsafe { f(session_ptr, i, allocator_ptr, &mut name_bytes) };
		status_to_result(status).map_err(OrtError::GetInputName)?;
		assert_non_null_pointer(name_bytes, "InputName")?;

		let name = char_p_to_string(name_bytes)?;
		ortfree!(unsafe allocator_ptr, name_bytes);
		Ok(name)
	}

	pub(super) fn extract_input(session_ptr: *mut sys::OrtSession, allocator_ptr: *mut sys::OrtAllocator, i: usize) -> OrtResult<Input> {
		let input_name = extract_input_name(session_ptr, allocator_ptr, i)?;
		let f = ort().SessionGetInputTypeInfo.unwrap();
		let (input_type, dimensions) = extract_io(f, session_ptr, i)?;
		Ok(Input {
			name: input_name,
			input_type,
			dimensions
		})
	}

	pub(super) fn extract_output(session_ptr: *mut sys::OrtSession, allocator_ptr: *mut sys::OrtAllocator, i: usize) -> OrtResult<Output> {
		let output_name = extract_output_name(session_ptr, allocator_ptr, i)?;
		let f = ort().SessionGetOutputTypeInfo.unwrap();
		let (output_type, dimensions) = extract_io(f, session_ptr, i)?;
		Ok(Output {
			name: output_name,
			output_type,
			dimensions
		})
	}

	fn extract_io(
		f: extern_system_fn! { unsafe fn(
			*const sys::OrtSession,
			usize,
			*mut *mut sys::OrtTypeInfo,
		) -> *mut sys::OrtStatus },
		session_ptr: *mut sys::OrtSession,
		i: usize
	) -> OrtResult<(TensorElementDataType, Vec<Option<u32>>)> {
		let mut typeinfo_ptr: *mut sys::OrtTypeInfo = std::ptr::null_mut();

		let status = unsafe { f(session_ptr, i, &mut typeinfo_ptr) };
		status_to_result(status).map_err(OrtError::GetTypeInfo)?;
		assert_non_null_pointer(typeinfo_ptr, "TypeInfo")?;

		let mut tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
		ortsys![unsafe CastTypeInfoToTensorInfo(typeinfo_ptr, &mut tensor_info_ptr) -> OrtError::CastTypeInfoToTensorInfo; nonNull(tensor_info_ptr)];

		let io_type: TensorElementDataType = unsafe { extract_data_type(tensor_info_ptr)? };
		let node_dims = unsafe { get_tensor_dimensions(tensor_info_ptr)? };

		ortsys![unsafe ReleaseTypeInfo(typeinfo_ptr)];

		Ok((io_type, node_dims.into_iter().map(|d| if d == -1 { None } else { Some(d as u32) }).collect()))
	}
}
