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
