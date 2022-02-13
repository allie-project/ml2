# GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models
This repository implements the sampling parts of [OpenAI GLIDE](https://arxiv.org/abs/2112.10741) in C++. Code is adapted from [glide-text2im](https://github.com/openai/glide-text2im). You can convert the existing models in Python like so:

This library **is not an implementation of the model**, but rather a simple interface for both the regular & upsampling models to easily encode the prompt, run sampling, & get an output image in regular RGB format. GLIDE is tested with an ONNX Runtime version of the model, however, you can use any neural network library you want.

More documentation, benchmarks, simple API, & pre-converted ONNX & ONNX Runtime models coming soon.
