pub mod composing;
pub mod correlation;
pub mod dataset;
pub mod error;
mod metrics_classification;
mod metrics_clustering;
mod metrics_regression;
pub mod param_guard;
pub mod prelude;
pub mod traits;

pub use composing::*;
pub use dataset::{Dataset, DatasetBase, DatasetPr, DatasetView, Float, Label};
pub use error::Error;
#[cfg(all(feature = "linalg-pure", not(feature = "linalg-blas")))]
pub use linfa_linalg as linalg;
#[cfg(feature = "linalg-blas")]
pub use ndarray_linalg as linalg;
pub use param_guard::ParamGuard;

/// Common metrics functions for classification and regression
pub mod metrics {
	pub use crate::core::metrics_classification::{BinaryClassification, ConfusionMatrix, ReceiverOperatingCharacteristic, ToConfusionMatrix};
	pub use crate::core::metrics_clustering::SilhouetteScore;
	pub use crate::core::metrics_regression::{MultiTargetRegression, SingleTargetRegression};
}
