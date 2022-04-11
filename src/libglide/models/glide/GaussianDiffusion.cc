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

#include <assert.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "libglide/core/operators.hh"
#include "libglide/models/glide/betaSchedule.hh"
#include "libglide/models/glide/GaussianDiffusion.hh"

namespace glide {

namespace {
	/**
	 * Extract values from a 1-D array for a batch of indices.
	 *
	 * @param arr the 1-D array.
	 * @param timesteps a tensor of indices into the array to extract.
	 * @param broadcastShape a larger shape of K dimensions with the batch dimension equal to the length of timesteps
	 * @returns a tensor of shape [batchSize, 1, ...] where the shape has K dims.
	 */
	inline xt::xarray<double> extractIntoTensor(const xt::xtensor<double, 1> arr, const xt::xtensor<int, 1> timesteps, const std::array<size_t, 4> broadcastShape) {
		// res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
		// while len(res.shape) < len(broadcast_shape):
		// 	res = res[..., None]
		// return res + th.zeros(broadcast_shape, device=timesteps.device)

		// if timesteps is of length 2 (as is the case in the GLIDE main model), we need to make sure that both timesteps are identical
		// otherwise this very, very hacky code will break
		assert(timesteps.size() > 1 ? timesteps[0] == timesteps[1] : true);
		return arr.flat(timesteps[0]) * xt::ones<double>(broadcastShape);
	}
}

GaussianDiffusion::GaussianDiffusion(const xt::xtensor<double, 1> betas) : betas(betas) {
	assert(betas.shape().size() == 1);
	assert(xt::all(betas > 0 && betas <= 1));

	this->numTimesteps = betas.shape()[0];

	auto alphas = 1.0 - betas;
	this->alphasCumulative = xt::cumprod<double>(xt::eval(alphas), 0);
	assert(this->alphasCumulative.shape()[0] == this->numTimesteps);
	// self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
	this->alphasCumulativePrev = xt::concatenate(xt::xtuple(xt::xtensor<double, 1>({ 1.0 }), xt::view(this->alphasCumulative, xt::range(0, -1))));
	// self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
	this->alphasCumulativeNext = xt::concatenate(xt::xtuple(xt::view(this->alphasCumulative, xt::range(1, xt::placeholders::_)), xt::xtensor<double, 1>({ 0.0 })));
	assert(this->alphasCumulative.shape()[0] == this->alphasCumulativePrev.shape()[0]);

	// calculations for diffusion q(x_t | x_{t-1}) and others
	this->sqrtAlphasCumulative = xt::sqrt(this->alphasCumulative);
	this->sqrtOneMinusAlphasCumulative = xt::sqrt(1.0 - this->alphasCumulative);
	assert(this->sqrtAlphasCumulative.shape()[0] == this->sqrtOneMinusAlphasCumulative.shape()[0]);
	this->sqrtLogOneMinusAlphasCumulative = xt::log(1.0 - this->alphasCumulative);
	this->sqrtRecipAlphasCumulative = xt::sqrt(1.0 / this->alphasCumulative);
	this->sqrtRecipm1AlphasCumulative = xt::sqrt(1.0 / this->alphasCumulative - 1);

	// calculations for posterior q(x_{t-1} | x_t, x_0)
	this->posteriorVariance = (betas * (1.0 - this->alphasCumulativePrev) / (1.0 - this->alphasCumulative));
	// below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
	this->posteriorLogVarianceClipped = xt::log(xt::concatenate(xt::xtuple(xt::xtensor<double, 1>({ this->posteriorVariance[1] }), xt::view(this->posteriorVariance, xt::range(1, xt::placeholders::_)))));
	this->posteriorMeanCoef1 = (betas * xt::sqrt(this->alphasCumulativePrev) / (1.0 - this->alphasCumulative));
	this->posteriorMeanCoef2 = ((1.0 - this->alphasCumulativePrev) * xt::sqrt(alphas) / (1.0 - this->alphasCumulative));
}

auto GaussianDiffusion::predictXStartFromEps(xt::xtensor<double, 4> x, xt::xtensor<int, 1> t, xt::xtensor<double, 4> eps) {
	assert(x.shape() == eps.shape());
	return xt::eval(
			extractIntoTensor(this->sqrtRecipAlphasCumulative, t, x.shape()) * x
		- extractIntoTensor(this->sqrtRecipm1AlphasCumulative, t, x.shape()) * eps
	);
}

auto GaussianDiffusion::predictEpsFromXStart(xt::xtensor<double, 4> x, xt::xtensor<int, 1> t, xt::xtensor<double, 4> predXStart) {
	return xt::eval(
			(extractIntoTensor(this->sqrtRecipAlphasCumulative, t, x.shape()) * x - predXStart)
		/ extractIntoTensor(this->sqrtRecipm1AlphasCumulative, t, x.shape())
	);
}

/**
 * Compute the mean and variance of the diffusion posterior:
 *
 *    q(x_{t-1} | x_t, x_0)
 */
auto GaussianDiffusion::qPosteriorMeanVariance(xt::xtensor<double, 4> xStart, xt::xtensor<double, 4> x_t, xt::xtensor<double, 1> t) {
	assert(xStart.shape() == x_t.shape());
	auto posteriorMean = (
			extractIntoTensor(this->posteriorMeanCoef1, t, x_t.shape()) * xStart
		+ extractIntoTensor(this->posteriorMeanCoef2, t, x_t.shape()) * x_t
	);
	auto posteriorVariance = extractIntoTensor(this->posteriorVariance, t, x_t.shape());
	auto posteriorLogVarianceClipped = extractIntoTensor(this->posteriorLogVarianceClipped, t, x_t.shape());
	assert(
		posteriorMean.shape()[0] == posteriorVariance.shape()[0] &&
		posteriorVariance.shape()[0] == posteriorLogVarianceClipped.shape()[0] &&
		posteriorLogVarianceClipped.shape()[0] == xStart.shape()[0]
	);
	return xt::xtuple(xt::eval(posteriorMean), posteriorVariance, posteriorLogVarianceClipped);
}

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
auto GaussianDiffusion::pMeanVariance(
	Model const &model,
	xt::xtensor<double, 4> x,
	xt::xtensor<int, 1> t,
	bool clipDenoised,
	std::function<xt::xtensor<double, 4>(xt::xtensor<double, 4>)> denoisedFn,
	ModelArgs&&... modelArgs
) {
	auto B /* batches? */ = x.shape()[0];
	auto C /* channels? */ = x.shape()[1];
	assert(t.shape().size() == 1 && t.shape()[0] == B);
	auto modelOutput = model(x, t, modelArgs...);
	assert(
		modelOutput.shape()[0] == B &&
		modelOutput.shape()[1] == C * 2 &&
		modelOutput.shape()[2] == x.shape()[2] &&
		modelOutput.shape()[3] == x.shape()[3]);
	
	// PERF: this is slow apparently (<= 14ms)
	auto outputTuple = xt::split(modelOutput, 2, 1);
	auto output = outputTuple[0], varValues = outputTuple[1];
	auto minLog = extractIntoTensor(this->posteriorLogVarianceClipped, t, x.shape());
	auto maxLog = extractIntoTensor(xt::eval(xt::log(this->betas)), t, x.shape());
	auto frac = (varValues + 1) / 2;
	auto modelLogVariance = frac * maxLog + (1 - frac) * minLog;
	auto modelVariance = xt::exp(modelLogVariance);

	auto xStart = this->predictXStartFromEps(x, t, output);
	if (denoisedFn)
		xStart = denoisedFn(xStart);
	if (clipDenoised)
		xStart = xt::clip(x, -1, 1);

	auto modelMean = std::get<0>(this->qPosteriorMeanVariance(xStart, x, t));

	assert(modelMean.shape() == modelLogVariance.shape());
	assert(modelLogVariance.shape() == xStart.shape());
	assert(xStart.shape() == x.shape());

	return PMeanVarianceResult {
		modelMean,
		modelVariance,
		modelLogVariance,
		xStart
	};
}

/**
 * Compute the mean for the previous step, given a function cond_fn that
 * computes the gradient of a conditional log probability with respect to
 * x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
 * condition on y.
 *
 * This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
 */
template<typename CondFn, typename ...ModelArgs>
auto GaussianDiffusion::conditionMean(
	CondFn condFn,
	PMeanVarianceResult pMeanVar,
	xt::xtensor<double, 4> x,
	xt::xtensor<int, 1> t,
	ModelArgs&&... modelArgs
) {
	auto gradient = condFn(x, t, modelArgs...);
	return pMeanVar.mean + pMeanVar.variance * gradient;
}

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
auto GaussianDiffusion::conditionScore(
	CondFn condFn,
	PMeanVarianceResult pMeanVar,
	xt::xtensor<double, 4> x,
	xt::xtensor<int, 1> t,
	ModelArgs&&... modelArgs
) {
	auto alphaBar = extractIntoTensor(this->alphasCumulative, t, x.shape());

	auto eps = this->predictEpsFromXStart(x, t, pMeanVar.predXStart);
	eps = eps - xt::sqrt(1 - alphaBar) * condFn(x, t, modelArgs...);

	return PMeanVarianceResult {
		std::get<0>(this->qPosteriorMeanVariance(pMeanVar.predXStart, x, t)),
		pMeanVar.variance,
		pMeanVar.logVariance,
		this->predictXStartFromEps(x, t, eps)
	};
}

}
