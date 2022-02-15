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
#include <algorithm>
#include <functional>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>

#include <xtensor/xio.hpp>
#include <iostream>
#include <chrono>

#include "../../core/operators.hh"
#include "betaSchedule.hh"

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
	template<typename T = double>
	xt::xarray<T> _extractIntoTensor(const xt::xtensor<T, 1> arr, const xt::xtensor<int, 1> timesteps, const std::array<size_t, 4> broadcastShape) {
		// res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
		// while len(res.shape) < len(broadcast_shape):
		// 	res = res[..., None]
		// return res + th.zeros(broadcast_shape, device=timesteps.device)

		// if timesteps is of length 2 (as the case of the GLIDE main model), we need to make sure that both timesteps are identical
		// otherwise this very, very hacky code will break
		assert(timesteps.size() > 1 ? timesteps[0] == timesteps[1] : true);
		return arr.flat(timesteps[0] /* this is gonna fucking break */) * xt::ones<T>(broadcastShape);
	}
}

template<typename T = double>
class GaussianDiffusion {
	public:
		GaussianDiffusion(xt::xtensor<T, 1> betas) : betas(betas) {
			assert(betas.shape().size() == 1);
			assert(xt::all(betas > 0 && betas <= 1));

			this->numTimesteps = betas.shape()[0];

			//std::cout << betas << std::endl;
			//std::cout << this->numTimesteps << std::endl;

			auto alphas = 1.0 - betas;
			this->alphasCumulative = xt::cumprod<T>(xt::eval(alphas), 0);
			assert(this->alphasCumulative.shape()[0] == this->numTimesteps);
			// self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
			this->alphasCumulativePrev = xt::concatenate(xt::xtuple(xt::xtensor<T, 1>({ 1.0 }), xt::view(this->alphasCumulative, xt::range(0, -1))));
			// self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
			this->alphasCumulativeNext = xt::concatenate(xt::xtuple(xt::view(this->alphasCumulative, xt::range(1, xt::placeholders::_)), xt::xtensor<T, 1>({ 0.0 })));
			assert(this->alphasCumulative.shape()[0] == this->alphasCumulativePrev.shape()[0]);

			//std::cout << this->alphasCumulative << std::endl;
			//std::cout << this->alphasCumulativePrev << std::endl;
			//std::cout << this->alphasCumulativeNext << std::endl;

			// calculations for diffusion q(x_t | x_{t-1}) and others
			this->sqrtAlphasCumulative = xt::sqrt(this->alphasCumulative);
			this->sqrtOneMinusAlphasCumulative = xt::sqrt(1.0 - this->alphasCumulative);
			assert(this->sqrtAlphasCumulative.shape()[0] == this->sqrtOneMinusAlphasCumulative.shape()[0]);
			this->sqrtLogOneMinusAlphasCumulative = xt::log(1.0 - this->alphasCumulative);
			this->sqrtRecipAlphasCumulative = xt::sqrt(1.0 / this->alphasCumulative);
			this->sqrtRecipm1AlphasCumulative = xt::sqrt(1.0 / this->alphasCumulative - 1);

			//std::cout << this->sqrtAlphasCumulative << std::endl;
			//std::cout << this->sqrtOneMinusAlphasCumulative << std::endl;
			//std::cout << this->sqrtLogOneMinusAlphasCumulative << std::endl;
			//std::cout << this->sqrtRecipAlphasCumulative << std::endl;
			//std::cout << this->sqrtRecipm1AlphasCumulative << std::endl;
 
			// calculations for posterior q(x_{t-1} | x_t, x_0)
			this->posteriorVariance = (betas * (1.0 - this->alphasCumulativePrev) / (1.0 - this->alphasCumulative));
			// below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
			this->posteriorLogVarianceClipped = xt::log(xt::concatenate(xt::xtuple(xt::xtensor<T, 1>({ this->posteriorVariance[1] }), xt::view(this->posteriorVariance, xt::range(1, xt::placeholders::_)))));
			this->posteriorMeanCoef1 = (betas * xt::sqrt(this->alphasCumulativePrev) / (1.0 - this->alphasCumulative));
			this->posteriorMeanCoef2 = ((1.0 - this->alphasCumulativePrev) * xt::sqrt(alphas) / (1.0 - this->alphasCumulative));

			//std::cout << this->posteriorVariance << std::endl;
			//std::cout << this->posteriorLogVarianceClipped << std::endl;
			//std::cout << this->posteriorMeanCoef1 << std::endl;
			//std::cout << this->posteriorMeanCoef2 << std::endl;
		}

		/**
		 * Compute the mean and variance of the diffusion posterior:
		 *
		 *    q(x_{t-1} | x_t, x_0)
		 */
		auto qPosteriorMeanVariance(xt::xtensor<T, 4> xStart, xt::xtensor<T, 4> x_t, xt::xtensor<T, 1> t) {
			assert(xStart.shape() == x_t.shape());
			auto posteriorMean = (
				  _extractIntoTensor(this->posteriorMeanCoef1, t, x_t.shape()) * xStart
				+ _extractIntoTensor(this->posteriorMeanCoef2, t, x_t.shape()) * x_t
			);
			auto posteriorVariance = _extractIntoTensor(this->posteriorVariance, t, x_t.shape());
			auto posteriorLogVarianceClipped = _extractIntoTensor(this->posteriorLogVarianceClipped, t, x_t.shape());
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
		 * @return a tuple of (mean, variance, log_variance, pred_xstart)
		 */
		template<typename Model, typename ...ModelArgs>
		auto pMeanVariance(
			Model const &model,
			xt::xtensor<T, 4> x,
			xt::xtensor<int, 1> t,
			bool clipDenoised = true,
			std::function<xt::xtensor<T, 4>(xt::xtensor<T, 4>)> denoisedFn = nullptr,
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
			
			// PERF: this is slow (<= 14ms)
			auto outputTuple = xt::split(modelOutput, 2, 1);
			auto output = outputTuple[0], varValues = outputTuple[1];
			auto minLog = _extractIntoTensor(this->posteriorLogVarianceClipped, t, x.shape());
			auto maxLog = _extractIntoTensor(xt::eval(xt::log(this->betas)), t, x.shape());
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

			return modelMean;
		}

	protected:
		auto predictXStartFromEps(xt::xtensor<T, 4> x, xt::xtensor<int, 1> t, xt::xtensor<T, 4> eps) {
			assert(x.shape() == eps.shape());
			return xt::eval(
				  _extractIntoTensor(this->sqrtRecipAlphasCumulative, t, x.shape()) * x
				- _extractIntoTensor(this->sqrtRecipm1AlphasCumulative, t, x.shape()) * eps
			);
		}

		auto predictEpsFromXStart(xt::xtensor<T, 4> x, xt::xtensor<int, 1> t, xt::xtensor<T, 4> predXStart) {
			return xt::eval(
				 (_extractIntoTensor(this->sqrtRecipAlphasCumulative, t, x.shape()) * x - predXStart)
				/ _extractIntoTensor(this->sqrtRecipm1AlphasCumulative, t, x.shape())
			);
		}

		xt::xtensor<T, 1> betas;
		int numTimesteps;

		// TODO: these should be xfunction for optimizations
		xt::xtensor<T, 1> alphasCumulative;
		xt::xtensor<T, 1> alphasCumulativePrev;
		xt::xtensor<T, 1> alphasCumulativeNext;

		xt::xtensor<T, 1> sqrtAlphasCumulative;
		xt::xtensor<T, 1> sqrtOneMinusAlphasCumulative;
		xt::xtensor<T, 1> sqrtLogOneMinusAlphasCumulative;
		xt::xtensor<T, 1> sqrtRecipAlphasCumulative;
		xt::xtensor<T, 1> sqrtRecipm1AlphasCumulative;

		xt::xtensor<T, 1> posteriorVariance;
		xt::xtensor<T, 1> posteriorLogVarianceClipped;
		xt::xtensor<T, 1> posteriorMeanCoef1;
		xt::xtensor<T, 1> posteriorMeanCoef2;
};

template class GaussianDiffusion<double>;

}

// int main() {
// 	auto x = xt::eval(xt::random::randn<double>({ 1, 3, 256, 256 }));
// 	auto t = xt::eval(xt::random::randint<int>({ 1 }, 0, 1000));
// 	using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
// 	auto start = Clock::now();

// 	for (int i = 0; i < 1; i++) {
// 		auto betas = glide::getNamedBetaSchedule(glide::BetaScheduleTypeV2::BETA_SCHEDULE_SQUAREDCOS_CAP_V2, 1000);
// 		glide::GaussianDiffusion<double> gd(betas);
// 		gd.pMeanVariance(
// 			[](xt::xtensor<double, 4> x, xt::xtensor<int, 1> t) {
// 				return xt::eval(xt::concatenate(xt::xtuple(x, x), 1));
// 			},
// 			x,
// 			t
// 		);
// 	}
	
// 	auto end = Clock::now();
// 	auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
// 	double seconds = diff / 1000000.0;
// 	std::cout << "C++ took " << seconds << " seconds" << std::endl;

// 	return 0;
// }
