<div align=center>
	<img src="docs/assets/libglide-banner.png" width="1000" alt="libglide">
	<h1>Neural Network Utilities</h1>
</div>

libglide implements some algorithms used by certain AI models used in [Allie Project](https://github.com/allie-project/allie), as well as a framework for training & inferring simple neural networks based on [TinyDNN](https://github.com/tiny-dnn). There is a Node.js addon available [in the Allie monorepo](https://github.com/allie-project/allie/tree/master/packages/glide), but it is not currently published, you'll have to build it yourself.

**Included model utils**:
- ✅ [GLIDE](https://arxiv.org/abs/2112.10741)
- ❌ [T5](https://arxiv.org/abs/1910.10683) & [mT5](https://arxiv.org/abs/2010.11934)
- ❌ [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- ❌ [Glow-TTS](https://arxiv.org/abs/2005.11129)
- ❌ [HiFi-GAN](https://arxiv.org/abs/2010.05646)
- ❌ [StyleGAN](https://arxiv.org/abs/1812.04948)

libglide as a neural network framework is not intended to be used in any serious capacity. It does well only with simple classifiers and is very unoptimized. We have no intention of expanding it in the future because TinyDNN's architecture isn't that great (and besides, there's much more established and supported libs, like [ONNX Runtime](https://onnxruntime.ai), [TFLite](https://www.tensorflow.org/lite/), [Caffe2](https://caffe2.ai), [PyTorch](https://pytorch.org), etc). We will only deliver occasional optimizations and small additions.

## Building
First, make sure submodules are synced and updated by running `git submodule sync --recursive` and `git submodule update --recursive`.

Simply run:

```shell
# Configure CMake
> cmake -S . -B build
# Build libglide & tests
> cmake --build build -j
```

> ℹ️ **NOTE**: If you're using Ninja, or another non-MSBuild generator on Windows, use the provided `vcvars.ps1` script to initialize Visual Studio environment variables. Otherwise, Ninja/etc won't find the correct include paths for standard headers/libraries.