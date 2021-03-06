#[cfg(feature = "linalg-blas")]
extern crate blas_src;

pub mod audio;
#[cfg(feature = "bayes")]
pub mod bayes;
pub mod core;
#[cfg(all(feature = "kernel", feature = "nn"))]
pub mod kernel;
#[cfg(all(feature = "_linalg", feature = "logistic"))]
pub mod logistic;
pub mod ndarray;
#[cfg(feature = "nn")]
pub mod nn;
#[cfg(feature = "onnx")]
pub mod onnx;
