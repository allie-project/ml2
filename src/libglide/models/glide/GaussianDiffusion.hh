/**
 * MIT License
 * 
 * Copyright (c) 2022 pyke.io
 *               2021 OpenAI (ported from https://github.com/openai/glide-text2im)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>

namespace glide {

struct PMeanVarianceResult {
	xt::xarray<double> mean;
	xt::xarray<double> variance;
	xt::xarray<double> logVariance;
	xt::xarray<double> predXStart;
};

class GaussianDiffusion {
	public:
		GaussianDiffusion(const xt::xtensor<double, 1> betas);

		/**
		 * Compute the mean and variance of the diffusion posterior:
		 *
		 *    q(x_{t-1} | x_t, x_0)
		 */
		auto qPosteriorMeanVariance(xt::xtensor<double, 4> xStart, xt::xtensor<double, 4> x_t, xt::xtensor<double, 1> t);

		/**
		 * Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x, x_0.
		 *
		 * @param model the model, which takes a signal and a batch of timesteps as input.
		 * @param x the [N x C x ...] tensor at time t.
		 * @param t a 1-D Tensor of timesteps.
		 * @param clipDenoised if true, clip the denoised signal into [-1, 1]
		 * @param denoisedFn if not null, a function which applies to the x_start prediction before it is used to sample. Applies before clip_denoised.
		 * @param modelArgs additional arguments to pass to the model. This can be used for conditioning.
		 */
		template<typename Model, typename ...ModelArgs>
		auto pMeanVariance(
			Model const &model,
			xt::xtensor<double, 4> x,
			xt::xtensor<int, 1> t,
			bool clipDenoised = true,
			std::function<xt::xtensor<double, 4>(xt::xtensor<double, 4>)> denoisedFn = nullptr,
			ModelArgs&&... modelArgs
		);

		/**
		 * Compute the mean for the previous step, given a function cond_fn that
		 * computes the gradient of a conditional log probability with respect to
		 * x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
		 * condition on y.
		 *
		 * This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
		 */
		template<typename CondFn, typename ...ModelArgs>
		auto conditionMean(
			CondFn condFn,
			PMeanVarianceResult pMeanVar,
			xt::xtensor<double, 4> x,
			xt::xtensor<int, 1> t,
			ModelArgs&&... modelArgs
		);

		/**
		 * Compute what the p_mean_variance output would have been, should the model's
		 * score function be conditioned by cond_fn.
		 *
		 * See `conditionMean()` for details on cond_fn.
		 *
		 * Unlike `conditionMean()`, this instead uses the conditioning strategy from
		 * Song et al (2020).
		 */
		template<typename CondFn, typename ...ModelArgs>
		auto conditionScore(
			CondFn condFn,
			PMeanVarianceResult pMeanVar,
			xt::xtensor<double, 4> x,
			xt::xtensor<int, 1> t,
			ModelArgs&&... modelArgs
		);

		xt::xtensor<double, 1> alphasCumulative;
		xt::xtensor<double, 1> alphasCumulativePrev;
		xt::xtensor<double, 1> alphasCumulativeNext;

		xt::xtensor<double, 1> sqrtAlphasCumulative;
		xt::xtensor<double, 1> sqrtOneMinusAlphasCumulative;
		xt::xtensor<double, 1> sqrtLogOneMinusAlphasCumulative;
		xt::xtensor<double, 1> sqrtRecipAlphasCumulative;
		xt::xtensor<double, 1> sqrtRecipm1AlphasCumulative;

		xt::xtensor<double, 1> posteriorVariance;
		xt::xtensor<double, 1> posteriorLogVarianceClipped;
		xt::xtensor<double, 1> posteriorMeanCoef1;
		xt::xtensor<double, 1> posteriorMeanCoef2;

	protected:
		auto predictXStartFromEps(xt::xtensor<double, 4> x, xt::xtensor<int, 1> t, xt::xtensor<double, 4> eps);
		auto predictEpsFromXStart(xt::xtensor<double, 4> x, xt::xtensor<int, 1> t, xt::xtensor<double, 4> predXStart);

		const xt::xtensor<double, 1> betas;
		int numTimesteps;
};

}