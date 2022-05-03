<div align=center>
	<img src=".github/banner.png" width="1000" alt="libglide">
	<h1>libglide - AI Utilities</h1>
</div>

libglide is the primary AI library used by [Allie Project](https://github.com/allie-project/allie). It provides simplified interfaces for a few ONNX Runtime models, a machine learning framework based on [linfa](https://github.com/rust-ml/linfa), and ONNX Runtime bindings based on [onnxruntime-rs](https://github.com/nbigaouette/onnxruntime-rs).

**Included model interfaces**:
- ❌ [GLIDE](https://arxiv.org/abs/2112.10741)
- ❌ [T5](https://arxiv.org/abs/1910.10683) & [mT5](https://arxiv.org/abs/2010.11934)
- ❌ [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- ❌ [Glow-TTS](https://arxiv.org/abs/2005.11129)
- ❌ [HiFi-GAN](https://arxiv.org/abs/2010.05646)
- ❌ [StyleGAN](https://arxiv.org/abs/1812.04948)

> **NOTE**: libglide is still a work in progress; we are currently working on porting the codebase from C++ to Rust.

## Usage
`libglide` is not published on Crates.io at the moment. You are welcome to use the code behind the model implementations provided in this repo in your projects; all code is under the Apache 2.0 license. You may use libglide as a Git crate, but note that the API is not guaranteed to be stable and may have breaking changes at any time.

## Cargo features
- **Linear Algebra**
    - `linalg-pure`: Enables modules using linear algebra without a BLAS backend. Operations like matrix multiplication will be slower, but the library compiles faster and does not require a Fortran compiler.
    - `linalg-blas`: Enables modules using linear algebra, enabling linking to a BLAS library. Linear algebra modules will work without a BLAS library (see `linalg-pure`), but enabling BLAS is recommended for release builds for the best possible speed.
        - `blas-netlib`, `blas-netlib-static` (**Unix**): Links to the Netlib BLAS library.
        - `blas-openblas`, `blas-openblas-static` (**All**): Links to the OpenBLAS library. Static linking is not allowed on Windows; consider Intel MKL instead.
        - `blas-intel-mkl`, `blas-intel-mkl-static` (**All (x86 only)**): Links to the Intel MKL library. Only supported for x86 and x64 targets.
        - `blas-blis`, `blas-blis-static` (**Unix**, recommended): Links to the BLIS library. Recommended for ease of use and speed on Unix targets.
        - `blas-accelerate` (**macOS**): Links to the Accelerate framework for BLAS.
- **Core Modules**: Core modules implementing machine learning tools similar to parts of Python's scikit-learn.
    - `all-modules`: Enable all modules. **Enabled by default, requires a linear algebra implementation.**
    - `logistic`: Provides pure Rust implementations of two-class (binomial) and multinomial logistic regression models. **Requires a linear algebra implementation.**
- `onnx`: Enables building ONNX Runtime Bindings. Must be enabled to use ONNX models (GLIDE, T5, GPT-2, Glow-TTS, HiFi-GAN, & StyleGAN).
    - `onnx-fetch-models`: Enables fetching models from the ONNX Model Zoo. Not recommended for production.
    - `onnx-generate-bindings`: Update/generate ONNX Runtime bindings with `bindgen`. Requires [libclang](https://clang.llvm.org/doxygen/group__CINDEX.html).
    - `onnx-copy-dylibs`: Copy dynamic libraries to the Cargo `target` folder. Heavily recommended on Windows, where the operating system may have an older version of ONNX Runtime bundled.
    - `onnx-prefer-system-strategy`: Uses the `system` compile strategy by default. This will require users to provide their own ONNX Runtime dylibs.
        - `onnx-prefer-dynamic-libs`: If the path pointed to by `ORT_LIB_LOCATION` contains static libraries, `libglide` will link to them rather than dynamic libraries. This feature prefers linking to dynamic libraries.
    - `onnx-prefer-compile-strategy`: Uses the `compile` strategy by default. This will take a *very* long time and is currently unfinished, but allows for static linking, avoiding DLL hell.
        - `onnx-compile-static`: Compiles ONNX Runtime as a static library.
        - `onnx-mimalloc`: Uses the (usually) faster mimalloc memory allocation library instead of the platform default.
        - `onnx-experimental`: Compiles Microsoft experimental operators.
        - `onnx-training`: Enables training via ONNX Runtime. Currently unavailable through high-level bindings.
        - `onnx-minimal-build`: Builds ONNX Runtime without ONNX model loading. Drastically reduces size. Recommended for release builds.
        - **Execution providers**
            - `onnxep-cuda`: Enables the CUDA execution provider for Maxwell (7xx) NVIDIA GPUs and above. Requires CUDA v11.4+.
            - `onnxep-tensorrt`: Enables the TensorRT execution provider for GeForce 9xx series NVIDIA GPUs and above. Requires CUDA v11.4+ and TensorRT v8.4+.
            - `onnxep-dnnl`: Enables the oneDNN execution provider for x86/x64 targets.
            - `onnxep-coreml`: Enables the CoreML execution provider for macOS/iOS targets.
            - `onnxep-armnn`: Enables the ArmNN execution provider for ARM v8 targets.
            - `onnxep-acl`: Enables the ARM Compute Library execution provider for multi-core ARM v8 processors.
            - `onnxep-directml`: Enables the DirectML execution provider for Windows x86/x64 targets with dedicated GPUs supporting DirectX 12.
            - `onnxep-migraphx`: Enables the MIGraphX execution provider for Windows x86/x64 targets with dedicated AMD GPUs.
            - `onnxep-nnapi`: Enables the Android Neural Networks API execution provider for Android targets.
            - `onnxep-tvm`: Enables the **preview** Apache TVM execution provider.
            - `onnxep-openvino`: Enables the OpenVINO execution provider for 6th+ generation Intel Core CPUs.
            - `onnxep-vitis-ai`: Enables Xilinx's Vitis-AI execution provider for U200/U250 accelerators.
- **Miscellaneous**
    - `half`: Builds support for `float16`/`bfloat16` ONNX tensors.
    - `use-half-intrinsics`: Use intrinsics in the `half` crate for faster operations when dealing with `float16`/`bfloat16` ONNX tensors.

## Shared library hell
Because building ONNX Runtime takes so long (and static linking is not recommended by Microsoft), it may be easier to compile ONNX Runtime as a shared library or use prebuilt DLLs. However, this can cause some issues with library paths and load orders.

### Windows
Some versions of Windows come bundled with `onnxruntime.dll` in the System32 folder. On build 22598.1, `onnxruntime.dll` is on version 1.10.0. libglide requires 1.11.0, so everything failed because system DLLs take precedence. Luckily, DLLs in the same folder as the application have higher priority; libglide can automatically copy the DLLs to the Cargo target folder when the `onnx-copy-dylibs` feature is enabled.

### Linux
You'll either have to copy `libonnxruntime.so` to a known lib location (e.g. `/usr/lib`) or use rpath.

In `Cargo.toml`:
```toml
[profile.dev]
rpath = true

[profile.release]
rpath = true

# do this for all profiles
```

In `.cargo/config.toml`:
```toml
[target.x86_64-unknown-linux-gnu]
rustflags = [ "-Clink-args=-Wl,-rpath,\\$ORIGIN" ]

# do this for all targets as well
```

### macOS
macOS follows the same procedure as Linux, except the rpath should point to `@loader_path` rather than `$ORIGIN`:

```toml
# .cargo/config.toml
[target.x86_64-apple-darwin]
rustflags = [ "-Clink-args=-Wl,-rpath,@loader_path" ]
```
