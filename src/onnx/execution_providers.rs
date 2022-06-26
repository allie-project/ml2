#![allow(unused_imports)]

use std::{collections::HashMap, ffi::CString, os::raw::c_char};

use super::{error::status_to_result, ortsys, sys};

extern "C" {
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_CPU(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "onnxep-acl")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
}

pub type ExecutionProviderOptions<'a> = Vec<(&'a str, Option<HashMap<&'a str, &'a str>>)>;

pub(crate) fn apply_execution_providers(options: *mut sys::OrtSessionOptions, execution_providers: &ExecutionProviderOptions) {
	for (name, init_args) in execution_providers {
		let init_args = if let Some(init_args) = init_args { init_args.clone() } else { HashMap::new() };
		let init_args: HashMap<&str, &str> = init_args.iter().map(|(k, v)| (*k, *v)).collect();
		match *name {
			#[cfg(feature = "onnxep-acl")]
			"acl" | "ACL" | "AclExecutionProvider" => {
				let use_arena = init_args.get("use_arena").map(|s| s.parse::<bool>().unwrap_or(false)).unwrap_or(false);
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_ACL(options, use_arena.into()) };
				if status_to_result(status).is_ok() {
					return; // EP found
				}
			}
			"cpu" | "CPU" | "CPUExecutionProvider" => {
				let use_arena = init_args.get("use_arena").map(|s| s.parse::<bool>().unwrap_or(false)).unwrap_or(false);
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena.into()) };
				if status_to_result(status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "onnxep-cuda")]
			"cuda" | "CUDA" | "CUDAExecutionProvider" => {
				let mut cuda_options: *mut sys::OrtCUDAProviderOptionsV2 = std::ptr::null_mut();
				if status_to_result(ortsys![unsafe CreateCUDAProviderOptions(&mut cuda_options)]).is_err() {
					continue; // next EP
				}
				let keys: Vec<CString> = init_args.keys().map(|k| CString::new(*k).unwrap()).collect();
				let values: Vec<CString> = init_args.values().map(|v| CString::new(*v).unwrap()).collect();
				assert_eq!(keys.len(), values.len()); // sanity check
				let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
				let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
				let status = ortsys![unsafe UpdateCUDAProviderOptions(cuda_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), keys.len())];
				if status_to_result(status).is_err() {
					ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
					continue; // next EP
				}
				let status = ortsys![unsafe SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda_options)];
				ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
				if status_to_result(status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "onnxep-tensorrt")]
			"tensorrt" | "TensorRT" | "TensorRTExecutionProvider" => {
				let mut tensorrt_options: *mut sys::OrtTensorRTProviderOptionsV2 = std::ptr::null_mut();
				if status_to_result(ortsys![unsafe CreateTensorRTProviderOptions(&mut tensorrt_options)]).is_err() {
					continue; // next EP
				}
				let keys: Vec<CString> = init_args.keys().map(|k| CString::new(*k).unwrap()).collect();
				let values: Vec<CString> = init_args.values().map(|v| CString::new(*v).unwrap()).collect();
				assert_eq!(keys.len(), values.len()); // sanity check
				let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
				let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
				let status = ortsys![unsafe UpdateTensorRTProviderOptions(tensorrt_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), keys.len())];
				if status_to_result(status).is_err() {
					ortsys![unsafe ReleaseTensorRTProviderOptions(tensorrt_options)];
					continue; // next EP
				}
				let status = ortsys![unsafe SessionOptionsAppendExecutionProvider_TensorRT_V2(options, tensorrt_options)];
				ortsys![unsafe ReleaseTensorRTProviderOptions(tensorrt_options)];
				if status_to_result(status).is_ok() {
					return; // EP found
				}
			}
			_ => {}
		};
		panic!("Applying execution provider {}", name);
	}
}
