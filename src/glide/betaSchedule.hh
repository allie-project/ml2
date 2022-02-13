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

#include <functional>
#include <xtensor/xtensor.hpp>

namespace glide {

enum BetaScheduleType {
	BETA_SCHEDULE_QUAD,
	BETA_SCHEDULE_LINEAR,
	BETA_SCHEDULE_WARMUP_10,
	BETA_SCHEDULE_WARMUP_50,
	BETA_SCHEDULE_CONST,
	BETA_SCHEDULE_JSD
};

/**
 * This is the deprecated API for creating beta schedules. See getNamedBetaSchedule() for the new library of schedules.
 */
xt::xtensor<double, 1> getBetaSchedule(BetaScheduleType betaSchedule, double betaStart, double betaEnd, int numDiffusionTimesteps);

enum BetaScheduleTypeV2 {
	BETA_SCHEDULE_LINEAR_V2,
	BETA_SCHEDULE_SQUAREDCOS_CAP_V2
};

xt::xtensor<double, 1> getNamedBetaSchedule(BetaScheduleTypeV2 scheduleType, int numDiffusionTimesteps);

xt::xtensor<double, 1> betasForAlphaBar(int numDiffusionTimesteps, std::function<double(double)> alphaBarFunc, double maxBeta = 0.999);

}