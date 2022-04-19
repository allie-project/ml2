<div align=center>
	<img src=".github/banner.png" width="1000" alt="libglide">
	<h1>Neural Network Utilities</h1>
</div>

libglide is the primary AI library used by [Allie Project](https://github.com/allie-project/allie). It provides simplified interfaces for a few ONNX Runtime models, a neural network framework based on [linfa](https://github.com/rust-ml/linfa), and ONNX Runtime bindings based on [onnxruntime-rs](https://github.com/nbigaouette/onnxruntime-rs).

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

## Shared library hell
Because building ONNX Runtime takes so long (and static linking is not recommended by Microsoft), it may be easier to compile ONNX Runtime as a shared library or use prebuilt DLLs. However, this can cause some issues with library paths and load orders.

### Windows
Some versions of Windows come bundled with `onnxruntime.dll` in the System32 folder. On build 22598.1, `onnxruntime.dll` is on version 1.10.0. libglide requires 1.11.0, so everything failed because system DLLs take precedence. Luckily, DLLs in the same folder as the application take even more precedence; libglide automatically copies the DLLs to the Cargo target folder when the `copy-dylibs` feature is enabled.

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
