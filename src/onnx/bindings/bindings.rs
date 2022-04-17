#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/onnx/bindings/linux/x86_64/bindings.rs"));

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/onnx/bindings/linux/aarch64/bindings.rs"));

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/onnx/bindings/macos/x86_64/bindings.rs"));

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/onnx/bindings/macos/aarch64/bindings.rs"));

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/onnx/bindings/windows/x86_64/bindings.rs"));

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/onnx/bindings/windows/aarch64/bindings.rs"));
