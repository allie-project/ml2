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

#include "betaSchedule.hh"

#define M_PI 3.14159265358979323846

namespace glide {

namespace {
	// https://github.com/openai/glide-text2im/blob/9cc8e563851bd38f5ddb3e305127192cb0f02f5c/glide_text2im/gaussian_diffusion.py#L11-L15
	xt::xtensor<double, 1> _warmupBeta(double betaStart, double betaEnd, int numDiffusionTimesteps, float warmupFrac = 0.5) {
		xt::xtensor<double, 1> betas = betaEnd * xt::ones<double>({ numDiffusionTimesteps });
		int warmupTime = (int)(numDiffusionTimesteps * warmupFrac);
		xt::view(betas, xt::range(0, warmupTime)) = xt::linspace<double>(betaStart, betaEnd, warmupTime);
		return betas;
	}
}

// https://github.com/openai/glide-text2im/blob/9cc8e563851bd38f5ddb3e305127192cb0f02f5c/glide_text2im/gaussian_diffusion.py#L18-L49
xt::xtensor<double, 1> getBetaSchedule(BetaScheduleType betaSchedule, double betaStart, double betaEnd, int numDiffusionTimesteps) {
	xt::xtensor<double, 1> betas;
	switch (betaSchedule) {
		case BetaScheduleType::BETA_SCHEDULE_QUAD:
			betas = xt::pow(xt::linspace<double>(pow(betaStart, 0.5), pow(betaEnd, 0.5), numDiffusionTimesteps), 2);
			break;
		case BetaScheduleType::BETA_SCHEDULE_WARMUP_10:
			betas = _warmupBeta(betaStart, betaEnd, numDiffusionTimesteps, 0.1f);
			break;
		case BetaScheduleType::BETA_SCHEDULE_WARMUP_50:
			betas = _warmupBeta(betaStart, betaEnd, numDiffusionTimesteps, 0.5f);
			break;
		case BetaScheduleType::BETA_SCHEDULE_CONST:
			betas = betaEnd * xt::ones<double>({ numDiffusionTimesteps });
			break;
		case BetaScheduleType::BETA_SCHEDULE_JSD:
			// 1/T, 1/(T-1), 1/(T-2), ..., 1
			betas = 1.0 / xt::linspace(numDiffusionTimesteps, 1, numDiffusionTimesteps);
			break;
		default:
		case BetaScheduleType::BETA_SCHEDULE_LINEAR:
			betas = xt::linspace<double>(betaStart, betaEnd, numDiffusionTimesteps);
			break;
	}

	assert(betas.shape()[0] == numDiffusionTimesteps);

	return betas;
}

xt::xtensor<double, 1> getNamedBetaSchedule(BetaScheduleTypeV2 scheduleType, int numDiffusionTimesteps) {
	switch (scheduleType) {
		default:
		case BetaScheduleTypeV2::BETA_SCHEDULE_LINEAR_V2: {
			// Linear schedule from Ho et al, extended to work for any number of diffusion steps.
			float scale = 1000.0f / numDiffusionTimesteps;
			return getBetaSchedule(
				BetaScheduleType::BETA_SCHEDULE_LINEAR,
				scale * 0.0001f,
				scale * 0.02f,
				numDiffusionTimesteps
			);
		}
		case BetaScheduleTypeV2::BETA_SCHEDULE_SQUAREDCOS_CAP_V2:
			return betasForAlphaBar(numDiffusionTimesteps, [](double t) {
				return pow(cos((t + 0.008) / 1.008 * M_PI / 2.0), 2);
			});
	}
}

xt::xtensor<double, 1> betasForAlphaBar(int numDiffusionTimesteps, std::function<double(double)> alphaBarFunc, double maxBeta) {
	xt::xtensor<double, 1> betas = xt::zeros<double>({ numDiffusionTimesteps });
	for (int i = 0; i < numDiffusionTimesteps; i++) {
		auto t1 = (double)i / (double)numDiffusionTimesteps;
		auto t2 = (double)(i + 1) / (double)numDiffusionTimesteps;
		betas[i] = std::min(1 - alphaBarFunc(t2) / alphaBarFunc(t1), maxBeta);
	}
	return betas;
}

}