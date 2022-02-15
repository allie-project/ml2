/**
 * MIT License
 * 
 * Copyright (c) 2022 pyke.io
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

#include <gtest/gtest.h>
#include <xtensor/xnpy.hpp>
#include <xtensor/xtensor.hpp>

#include "libglide/core/operators.hh"
#include "libglide/models/glide/betaSchedule.hh"

TEST(GLIDEBetaSchedule, BetaScheduleQuad) {
	auto libglide = glide::getBetaSchedule(glide::BetaScheduleType::BETA_SCHEDULE_QUAD, 0.1f, 0.3f, 1000);
	auto numpy = xt::load_npy<double>("./test/data/glide_beta_schedule_quad.npy");
	
	auto libglideShape = libglide.shape();
	auto numpyShape = numpy.shape();

	ASSERT_EQ(libglideShape.size(), 1) << "Beta schedule should be a scalar";
	ASSERT_EQ(libglideShape.size(), numpyShape.size()) << "Beta schedule shape should match reference shape";
	ASSERT_EQ(libglideShape[0], numpyShape[0]) << "Dimension count should be identical";
	ASSERT_TRUE(xt::allclose(libglide, numpy)) << "Data should be identical or within rounding error";
}

TEST(GLIDEBetaSchedule, BetaScheduleLinear) {
	auto libglide = glide::getBetaSchedule(glide::BetaScheduleType::BETA_SCHEDULE_LINEAR, 0.1f, 0.3f, 1000);
	auto numpy = xt::load_npy<double>("./test/data/glide_beta_schedule_linear.npy");
	
	auto libglideShape = libglide.shape();
	auto numpyShape = numpy.shape();

	ASSERT_EQ(libglideShape.size(), 1) << "Beta schedule should be a scalar";
	ASSERT_EQ(libglideShape.size(), numpyShape.size()) << "Beta schedule shape should match reference shape";
	ASSERT_EQ(libglideShape[0], numpyShape[0]) << "Dimension count should be identical";
	ASSERT_TRUE(xt::allclose(libglide, numpy)) << "Data should be identical or within rounding error";
}

TEST(GLIDEBetaSchedule, BetaScheduleWarmup10) {
	auto libglide = glide::getBetaSchedule(glide::BetaScheduleType::BETA_SCHEDULE_WARMUP_10, 0.1f, 0.3f, 1000);
	auto numpy = xt::load_npy<double>("./test/data/glide_beta_schedule_warmup10.npy");
	
	auto libglideShape = libglide.shape();
	auto numpyShape = numpy.shape();

	ASSERT_EQ(libglideShape.size(), 1) << "Beta schedule should be a scalar";
	ASSERT_EQ(libglideShape.size(), numpyShape.size()) << "Beta schedule shape should match reference shape";
	ASSERT_EQ(libglideShape[0], numpyShape[0]) << "Dimension count should be identical";
	ASSERT_TRUE(xt::allclose(libglide, numpy)) << "Data should be identical or within rounding error";
}

TEST(GLIDEBetaSchedule, BetaScheduleWarmup50) {
	auto libglide = glide::getBetaSchedule(glide::BetaScheduleType::BETA_SCHEDULE_WARMUP_50, 0.1f, 0.3f, 1000);
	auto numpy = xt::load_npy<double>("./test/data/glide_beta_schedule_warmup50.npy");
	
	auto libglideShape = libglide.shape();
	auto numpyShape = numpy.shape();

	ASSERT_EQ(libglideShape.size(), 1) << "Beta schedule should be a scalar";
	ASSERT_EQ(libglideShape.size(), numpyShape.size()) << "Beta schedule shape should match reference shape";
	ASSERT_EQ(libglideShape[0], numpyShape[0]) << "Dimension count should be identical";
	ASSERT_TRUE(xt::allclose(libglide, numpy)) << "Data should be identical or within rounding error";
}

TEST(GLIDEBetaSchedule, BetaScheduleConst) {
	auto libglide = glide::getBetaSchedule(glide::BetaScheduleType::BETA_SCHEDULE_CONST, 0.1f, 0.3f, 1000);
	auto numpy = xt::load_npy<double>("./test/data/glide_beta_schedule_const.npy");
	
	auto libglideShape = libglide.shape();
	auto numpyShape = numpy.shape();

	ASSERT_EQ(libglideShape.size(), 1) << "Beta schedule should be a scalar";
	ASSERT_EQ(libglideShape.size(), numpyShape.size()) << "Beta schedule shape should match reference shape";
	ASSERT_EQ(libglideShape[0], numpyShape[0]) << "Dimension count should be identical";
	ASSERT_TRUE(xt::allclose(libglide, numpy)) << "Data should be identical or within rounding error";
}

TEST(GLIDEBetaSchedule, BetaScheduleJSD) {
	auto libglide = glide::getBetaSchedule(glide::BetaScheduleType::BETA_SCHEDULE_JSD, 0.1f, 0.3f, 1000);
	auto numpy = xt::load_npy<double>("./test/data/glide_beta_schedule_jsd.npy");
	
	auto libglideShape = libglide.shape();
	auto numpyShape = numpy.shape();

	ASSERT_EQ(libglideShape.size(), 1) << "Beta schedule should be a scalar";
	ASSERT_EQ(libglideShape.size(), numpyShape.size()) << "Beta schedule shape should match reference shape";
	ASSERT_EQ(libglideShape[0], numpyShape[0]) << "Dimension count should be identical";
	ASSERT_TRUE(xt::allclose(libglide, numpy)) << "Data should be identical or within rounding error";
}



TEST(GLIDEBetaSchedule, BetaScheduleLinearV2) {
	auto libglide = glide::getNamedBetaSchedule(glide::BetaScheduleTypeV2::BETA_SCHEDULE_LINEAR_V2, 1000);
	auto numpy = xt::load_npy<double>("./test/data/glide_beta_schedule_linear_v2.npy");
	
	auto libglideShape = libglide.shape();
	auto numpyShape = numpy.shape();

	ASSERT_EQ(libglideShape.size(), 1) << "Beta schedule should be a scalar";
	ASSERT_EQ(libglideShape.size(), numpyShape.size()) << "Beta schedule shape should match reference shape";
	ASSERT_EQ(libglideShape[0], numpyShape[0]) << "Dimension count should be identical";
	ASSERT_TRUE(xt::allclose(libglide, numpy)) << "Data should be identical or within rounding error";
}

TEST(GLIDEBetaSchedule, BetaScheduleSquaredCosCapV2) {
	auto libglide = glide::getNamedBetaSchedule(glide::BetaScheduleTypeV2::BETA_SCHEDULE_SQUAREDCOS_CAP_V2, 1000);
	auto numpy = xt::load_npy<double>("./test/data/glide_beta_schedule_squaredcos_cap_v2.npy");
	
	auto libglideShape = libglide.shape();
	auto numpyShape = numpy.shape();

	ASSERT_EQ(libglideShape.size(), 1) << "Beta schedule should be a scalar";
	ASSERT_EQ(libglideShape.size(), numpyShape.size()) << "Beta schedule shape should match reference shape";
	ASSERT_EQ(libglideShape[0], numpyShape[0]) << "Dimension count should be identical";
	ASSERT_TRUE(xt::allclose(libglide, numpy)) << "Data should be identical or within rounding error";
}