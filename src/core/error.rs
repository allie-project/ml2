use ndarray::ShapeError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum Error {
	#[error("invalid parameter {0}")]
	Parameters(String),
	#[error("invalid prior {0}")]
	Priors(String),
	#[error("algorithm not converged {0}")]
	NotConverged(String),
	// ShapeError doesn't implement serde traits, and deriving them remotely on a complex error
	// type isn't really feasible, so we skip this variant.
	#[serde(skip)]
	#[error("invalid ndarray shape {0}")]
	NdShape(#[from] ShapeError),
	#[error("not enough samples")]
	NotEnoughSamples,
	#[error("The number of samples do not match: {0} - {1}")]
	MismatchedShapes(usize, usize)
}
