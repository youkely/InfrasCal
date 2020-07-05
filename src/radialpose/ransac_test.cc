#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <limits>
#include <RansacLib/ransac.h>

#include <time.h> 
#include <cmath>
#include <iostream>
#include <numeric>
#include "radialpose.h"
#include "misc/ransac_estimator.h"
#include "misc/unit_test_misc.h"

bool test_simple_ransac_no_outliers() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params2 = { -0.12, 0.034 };

	larsson_iccv19::Solver<2, 0, true> estimator;

	generate_scene_and_image(100, 2, 20, 70, false, &pose_gt, &x, &X, 1.0);
	add_rational_distortion(params2, 2, 0, &pose_gt, &x);
	add_focal(2000.0, &pose_gt, &x);
	add_noise(0.5, &x);

	RansacEstimator<larsson_iccv19::Solver<2, 0, true>> solver(x, X, estimator);

	ransac_lib::LORansacOptions options;
	options.squared_inlier_threshold_ = 4;

	ransac_lib::LocallyOptimizedMSAC<Camera,
		std::vector<Camera>,
		RansacEstimator<larsson_iccv19::Solver<2, 0, true>>> lomsac;
	ransac_lib::RansacStatistics ransac_stats;

	Camera best_model;
	int num_ransac_inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	std::cout << "   ... LOMSAC found " << num_ransac_inliers
		<< " inliers in " << ransac_stats.num_iterations
		<< " iterations with an inlier ratio of "
		<< ransac_stats.inlier_ratio << std::endl;

	return (ransac_stats.inlier_ratio > 0.99);
}


bool test_simple_ransac_some_outliers() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params2 = { -0.12, 0.034 };

	larsson_iccv19::Solver<2, 0, true> estimator;

	generate_scene_and_image(100, 2, 20, 70, false, &pose_gt, &x, &X, 1.0);
	add_rational_distortion(params2, 2, 0, &pose_gt, &x);
	add_focal(2000.0, &pose_gt, &x);
	add_noise(1.0, &x);

	for (int i = 0; i < 20; ++i) {
		Vector2d n; n.setRandom(); n *= 0.2 * pose_gt.focal;
		x.col(i) += n;
	}

	RansacEstimator<larsson_iccv19::Solver<2, 0, true>> solver(x, X, estimator);

	ransac_lib::LORansacOptions options;
	options.squared_inlier_threshold_ = 4;
	
	ransac_lib::LocallyOptimizedMSAC<Camera,
		std::vector<Camera>,
		RansacEstimator<larsson_iccv19::Solver<2, 0, true>>> lomsac;
	ransac_lib::RansacStatistics ransac_stats;

	Camera best_model;
	int num_ransac_inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	std::cout << "   ... LOMSAC found " << num_ransac_inliers
		<< " inliers in " << ransac_stats.num_iterations
		<< " iterations with an inlier ratio of "
		<< ransac_stats.inlier_ratio << std::endl;
	
	return (ransac_stats.inlier_ratio > .79);
}


int main() {
	std::cout << "Running tests...\n\n";
	srand((unsigned int)time(0));
	//srand(2.0);

	int passed = 0;
	int num_tests = 0;

	TEST(test_simple_ransac_no_outliers);
	TEST(test_simple_ransac_some_outliers);

}