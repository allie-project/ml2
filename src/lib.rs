#[cfg(feature = "linalg-blas")]
extern crate blas_src;

pub mod audio;
#[cfg(feature = "bayes")]
pub mod bayes;
pub mod core;
#[cfg(all(feature = "_linalg", feature = "logistic"))]
pub mod logistic;
pub mod ndarray;
#[cfg(feature = "onnx")]
pub mod onnx;
