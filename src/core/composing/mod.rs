//! This module contains three composition models:
//!  * `MultiClassModel`: combines multiple binary decision models into a single multi-class model
//!  * `MultiTargetModel`: combines multiple univariate models into a single multi-target model
//!  * `Platt`: calibrates a classifier (i.e. SVC) with predicted posterior probabilities
mod multi_class_model;
mod multi_target_model;
pub mod platt_scaling;

pub use multi_class_model::MultiClassModel;
pub use multi_target_model::MultiTargetModel;
pub use platt_scaling::{Platt, PlattError, PlattParams};
