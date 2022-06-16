#![allow(clippy::tabs_in_doc_comments)]

use std::{
	ffi::{c_void, CString},
	os::raw::c_char
};

use super::{
	char_p_to_string,
	error::{assert_non_null_pointer, status_to_result, Result as OrtResult},
	ort, sys, OrtError
};

pub struct Metadata {
	metadata_ptr: *mut sys::OrtModelMetadata,
	allocator_ptr: *mut sys::OrtAllocator
}

impl Metadata {
	pub(crate) fn new(metadata_ptr: *mut sys::OrtModelMetadata, allocator_ptr: *mut sys::OrtAllocator) -> Self {
		Metadata { metadata_ptr, allocator_ptr }
	}

	pub fn description(&self) -> OrtResult<String> {
		let mut str_bytes: *mut i8 = std::ptr::null_mut();
		let status = unsafe { ort().ModelMetadataGetDescription.unwrap()(self.metadata_ptr, self.allocator_ptr, &mut str_bytes) };
		status_to_result(status).map_err(OrtError::Allocator)?;
		assert_non_null_pointer(str_bytes, "ModelMetadataGetDescription")?;

		let value = char_p_to_string(str_bytes)?;
		unsafe { (*self.allocator_ptr).Free.unwrap()(self.allocator_ptr, str_bytes as *mut std::ffi::c_void) };
		Ok(value)
	}

	pub fn producer(&self) -> OrtResult<String> {
		let mut str_bytes: *mut i8 = std::ptr::null_mut();
		let status = unsafe { ort().ModelMetadataGetProducerName.unwrap()(self.metadata_ptr, self.allocator_ptr, &mut str_bytes) };
		status_to_result(status).map_err(OrtError::Allocator)?;
		assert_non_null_pointer(str_bytes, "ModelMetadataGetProducerName")?;

		let value = char_p_to_string(str_bytes)?;
		unsafe { (*self.allocator_ptr).Free.unwrap()(self.allocator_ptr, str_bytes as *mut std::ffi::c_void) };
		Ok(value)
	}

	pub fn name(&self) -> OrtResult<String> {
		let mut str_bytes: *mut i8 = std::ptr::null_mut();
		let status = unsafe { ort().ModelMetadataGetGraphName.unwrap()(self.metadata_ptr, self.allocator_ptr, &mut str_bytes) };
		status_to_result(status).map_err(OrtError::Allocator)?;
		assert_non_null_pointer(str_bytes, "ModelMetadataGetGraphName")?;

		let value = char_p_to_string(str_bytes)?;
		unsafe { (*self.allocator_ptr).Free.unwrap()(self.allocator_ptr, str_bytes as *mut std::ffi::c_void) };
		Ok(value)
	}

	pub fn version(&self) -> OrtResult<i64> {
		let mut ver = 0i64;
		let status = unsafe { ort().ModelMetadataGetVersion.unwrap()(self.metadata_ptr, &mut ver) };
		status_to_result(status).map_err(OrtError::Allocator)?;

		Ok(ver)
	}

	pub fn custom(&self, key: &str) -> OrtResult<Option<String>> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		let key_str = CString::new(key)?;
		let status = unsafe { ort().ModelMetadataLookupCustomMetadataMap.unwrap()(self.metadata_ptr, self.allocator_ptr, key_str.as_ptr(), &mut str_bytes) };
		status_to_result(status).map_err(OrtError::Allocator)?;
		if !str_bytes.is_null() {
			unsafe {
				let value = char_p_to_string(str_bytes)?;
				(*self.allocator_ptr).Free.unwrap()(self.allocator_ptr, str_bytes as *mut c_void);
				Ok(Some(value))
			}
		} else {
			Ok(None)
		}
	}
}

impl Drop for Metadata {
	fn drop(&mut self) {
		unsafe { ort().ReleaseModelMetadata.unwrap()(self.metadata_ptr) };
	}
}
