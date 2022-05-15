use argmin::prelude::*;
use ndarray::{Array, ArrayBase, Data, Dimension, Zip};
use serde_crate::{Deserialize, Serialize};

use super::float::Float;

pub fn elem_dot<F: crate::core::Float, A1: Data<Elem = F>, A2: Data<Elem = F>, D: Dimension>(a: &ArrayBase<A1, D>, b: &ArrayBase<A2, D>) -> F {
	Zip::from(a).and(b).fold(F::zero(), |acc, &a, &b| acc + a * b)
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(crate = "serde_crate")]
pub struct ArgminParam<F, D: Dimension>(pub Array<F, D>);

impl<F, D: Dimension> ArgminParam<F, D> {
	#[inline]
	pub fn as_array(&self) -> &Array<F, D> {
		&self.0
	}
}

impl<F: Float, D: Dimension> ArgminSub<ArgminParam<F, D>, ArgminParam<F, D>> for ArgminParam<F, D> {
	fn sub(&self, other: &ArgminParam<F, D>) -> ArgminParam<F, D> {
		ArgminParam(&self.0 - &other.0)
	}
}

impl<F: Float, D: Dimension> ArgminAdd<ArgminParam<F, D>, ArgminParam<F, D>> for ArgminParam<F, D> {
	fn add(&self, other: &ArgminParam<F, D>) -> ArgminParam<F, D> {
		ArgminParam(&self.0 + &other.0)
	}
}

impl<F: Float, D: Dimension> ArgminDot<ArgminParam<F, D>, F> for ArgminParam<F, D> {
	fn dot(&self, other: &ArgminParam<F, D>) -> F {
		elem_dot(&self.0, &other.0)
	}
}

impl<F: Float, D: Dimension> ArgminNorm<F> for ArgminParam<F, D> {
	fn norm(&self) -> F {
		num_traits::Float::sqrt(elem_dot(&self.0, &self.0))
	}
}

impl<F: Float, D: Dimension> ArgminMul<F, ArgminParam<F, D>> for ArgminParam<F, D> {
	fn mul(&self, other: &F) -> ArgminParam<F, D> {
		ArgminParam(&self.0 * *other)
	}
}

impl<F: Float, D: Dimension> ArgminMul<ArgminParam<F, D>, ArgminParam<F, D>> for ArgminParam<F, D> {
	fn mul(&self, other: &ArgminParam<F, D>) -> ArgminParam<F, D> {
		ArgminParam(&self.0 * &other.0)
	}
}
