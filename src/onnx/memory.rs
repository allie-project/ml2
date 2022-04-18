use tracing::{debug, error};

use super::{
	error::{assert_non_null_pointer, status_to_result, OrtError, Result},
	ort, sys, AllocatorType, MemType
};

#[derive(Debug)]
pub(crate) struct MemoryInfo {
	pub ptr: *mut sys::OrtMemoryInfo
}

impl MemoryInfo {
	#[tracing::instrument]
	pub fn new(allocator: AllocatorType, memory_type: MemType) -> Result<Self> {
		debug!("Creating new OrtMemoryInfo.");

		let mut memory_info_ptr: *mut sys::OrtMemoryInfo = std::ptr::null_mut();
		let status = unsafe { ort().CreateCpuMemoryInfo.unwrap()(allocator.into(), memory_type.into(), &mut memory_info_ptr) };
		status_to_result(status).map_err(OrtError::CreateCpuMemoryInfo)?;
		assert_non_null_pointer(memory_info_ptr, "MemoryInfo")?;

		Ok(Self { ptr: memory_info_ptr })
	}
}

impl Drop for MemoryInfo {
	#[tracing::instrument]
	fn drop(&mut self) {
		if self.ptr.is_null() {
			error!("OrtMemoryInfo pointer is null, not dropping.");
		} else {
			debug!("Dropping OrtMemoryInfo");
			unsafe { ort().ReleaseMemoryInfo.unwrap()(self.ptr) };
		}

		self.ptr = std::ptr::null_mut();
	}
}

#[cfg(test)]
mod tests {
	use test_log::test;

	use super::*;

	#[test]
	fn create_memory_info() {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
		std::mem::drop(memory_info);
	}
}
