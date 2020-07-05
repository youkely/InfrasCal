#define _USE_MATH_DEFINES

#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <limits>


#include <time.h> 
#include <cmath>
#include <iostream>
#include <numeric>
#include "radialpose.h"
#include "misc/unit_test_misc.h"
#include "misc/refinement.h"

template<typename Solver, typename AddDistortionFunc>
void benchmark_solver(std::string name, PoseEstimator<Solver>* estimator, AddDistortionFunc addDistortion, std::vector<double> params, int iters) {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<Camera> poses;
	std::vector<double> residuals;
	residuals.reserve(iters);
	std::vector<int> num_of_poses;
	num_of_poses.reserve(iters);

	srand(12345);

	for (int iter = 0; iter < iters; ++iter) {

		generate_scene_and_image(estimator->minimal_sample_size(), 2, 100, 90, false, &pose_gt, &x, &X, 100.0);
		addDistortion(params, &pose_gt, &x);
		add_focal(2000.0, &pose_gt, &x);

		int n_sols = estimator->estimate(x, X, &poses);
		double err = minimum_pose_distance(pose_gt, poses, true, false);
		
		residuals.push_back(std::log10(err));
		num_of_poses.push_back(poses.size());
	}

	std::sort(residuals.begin(), residuals.end());

	double average = accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();

	double median = residuals[residuals.size() / 2];
	double q25 = residuals[residuals.size() / 4];
	double q75 = residuals[(3 * residuals.size()) / 4];
	double q95 = residuals[(95 * residuals.size()) / 100];
	double average_num_pose = accumulate(num_of_poses.begin(), num_of_poses.end(), 0.0) / num_of_poses.size();

	std::cout << "BENCH " << name << ": average=" << average << ", median=" << median << " (q25=" << q25 << ", q75=" << q75 << ", q95=" << q95 << "), avg_num_poses=" << average_num_pose << "\n";
}


/////////////////////////// TESTS START HERE /////////////////////////



bool test_rational_distortion() {
		
	Matrix<double, 2, Dynamic> x0(2, 100);
	Matrix<double, 2, Dynamic> x1(2, 100);

	int nps[] = { 1, 2, 3, 0, 0, 0, 1, 3 };
	int nds[] = { 0, 0, 0, 1, 2, 0, 1, 3 };

	double lambdas[] = { -0.1, -0.01, -0.001 };
	double mus[] = { -0.1, -0.01, -0.001 };

	x0.setRandom();
	x0 *= 0.5;

	std::vector<double> params;
	
	int fail = 0;
	for (int i = 0; i < 8; ++i) {

		int np = nps[i];
		int nd = nds[i];
		for (int j = 0; j < np; ++j) {
			params.push_back(mus[j]);
		}
		for (int j = 0; j < nd; ++j) {
			params.push_back(lambdas[j]);
		}

		inverse_rational_model(params, np, nd, x0, &x1);
		forward_rational_model(params, np, nd, x1, &x1);
		double res = (x0 - x1).norm();
		if (res > TOL_POSE) {
			fail++;
		}
		
		forward_rational_model(params, np, nd, x0, &x1);
		inverse_rational_model(params, np, nd, x1, &x1);
		res = (x0 - x1).norm();
		if (res > TOL_POSE) {
			fail++;
		}
	}

	return fail == 0;

}

bool test_1param_div_model() {

	Matrix<double, 2, Dynamic> x0(2, 100);
	Matrix<double, 2, Dynamic> x1(2, 100);

	double lambdas[] = { 0.0, -0.01, -0.1, -0.2 };

	x0.setRandom();
	x0 *= 0.5;


	int fail = 0;
	for (int i = 0; i < 4; ++i) {
		inverse_1param_division_model(lambdas[i], x0, &x1);
		forward_1param_division_model(lambdas[i], x1, &x1);
		
		double res = (x0 - x1).norm();		
		if (res > TOL_POSE) {
			fail++;
		}

		forward_1param_division_model(lambdas[i], x0, &x1);
		inverse_1param_division_model(lambdas[i], x1, &x1);
		res = (x0 - x1).norm();
		if (res > TOL_POSE) {
			fail++;
		}
	}
	
	return fail == 0;
}


bool test_planar_oskarsson() {

	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	generate_scene_and_image(4, 2, 20, 70, true, &pose_gt, &x, &X);
	add_distortion_1pdiv(-0.2, &pose_gt, &x);
	add_focal(2000.0, &pose_gt, &x);

	std::vector<Camera> poses;

	// Transform such that make X(3,:) = 0
	Vector3d t0 = X.rowwise().mean();
	X.colwise() -= t0;

	JacobiSVD<Matrix<double, 3, Dynamic>> svd(X, ComputeThinU);
	Matrix<double, 3, 3> R0 = svd.matrixU();
	if (R0.determinant() < 0)
		R0 *= -1.0;
	X = R0.transpose() * X;
	pose_gt.t = pose_gt.t + pose_gt.R * t0;
	pose_gt.R = pose_gt.R * R0;

	oskarsson_arxiv18::PlanarSolver estimator;
	int n_sols = estimator.estimate(x, X, &poses);
	double err = minimum_pose_distance(pose_gt, poses);


	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);


	return (err < TOL_POSE);
}


bool test_kukelova_1d_radial() {

	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;
	
	generate_scene_and_image(5, 2, 20, 70, false, &pose_gt, &x, &X);
	add_focal(2000.0, &pose_gt, &x);

	kukelova_iccv13::Radial1DSolver estimator;
	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);

	// Fix sign and t_Z = 0 for all poses since we are invariant to these.
	for (int i = 0; i < poses.size(); ++i) {
		if (poses[i].R(0) * pose_gt.R(0) < 0) {
			poses[i].R *= -1.0;
			poses[i].t *= -1.0;
		}
		if (poses[i].R(2) * pose_gt.R(2) < 0) {
			poses[i].R.row(2) *= -1.0;
			poses[i].t.row(2) *= -1.0;
		}

		poses[i].t(2) = 0.0;
	}
	pose_gt.t(2) = 0.0;

	double err = minimum_pose_distance(pose_gt, poses, false, false);
	

	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);



	return (err < TOL_POSE);
	

}


bool test_kukelova_solver_1p_div() {

	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	generate_scene_and_image(5, 2, 20, 70, false, &pose_gt, &x, &X);
	add_distortion_1pdiv(-0.2, &pose_gt, &x);
	add_focal(5.0, &pose_gt, &x);
	
	std::vector<Camera> poses;

	kukelova_iccv13::Solver estimator(1);

	int n_sols = estimator.estimate(x, X, &poses);
	
	double err = minimum_pose_distance(pose_gt, poses);

	
	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);
	


	return (err < TOL_POSE);


}



bool test_kukelova_solver_3p_div() {

	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	generate_scene_and_image(5, 2, 20, 70, false, &pose_gt, &x, &X);
	
	std::vector<double> params = { -0.12, -0.034, -0.0056 };
	add_rational_undistortion(params, 0, 3, &pose_gt, &x);
	add_focal(5.0, &pose_gt, &x);

	std::vector<Camera> poses;

	kukelova_iccv13::Solver estimator(3);

	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses, true, true);


	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);



	return (err < TOL_POSE);


}


bool test_larsson17_solver_1p_div() {

	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	generate_scene_and_image(4, 2, 20, 70, false, &pose_gt, &x, &X);
	add_distortion_1pdiv(-0.2, &pose_gt, &x);
	add_focal(5.0, &pose_gt, &x);
	larsson_iccv17::NonPlanarSolver estimator;	

	std::vector<Camera> poses;

	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses);

	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err < TOL_POSE);


}


bool test_bujnak10_solver_1p_div() {

	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	generate_scene_and_image(4, 2, 20, 70, false, &pose_gt, &x, &X);
	add_distortion_1pdiv(-0.2, &pose_gt, &x);
	add_focal(5.0, &pose_gt, &x);
	bujnak_accv10::NonPlanarSolver estimator;

	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses);

	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err < TOL_POSE);


}


bool test_larsson19_distortion_1_0() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12 };
	larsson_iccv19::Solver<1, 0, true> estimator;

	generate_scene_and_image(estimator.minimal_sample_size(), 2, 20, 70, false, &pose_gt, &x, &X);
	add_rational_distortion(params, 1, 0, &pose_gt, &x);
	add_focal(2.0, &pose_gt, &x);

	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses, true, true);

	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err < TOL_POSE);
}


bool test_larsson19_distortion_1_0_planar() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12 };
	larsson_iccv19::Solver<1, 0, true> estimator;

	generate_scene_and_image(estimator.minimal_sample_size(), 2, 20, 70, true, &pose_gt, &x, &X);
	add_rational_distortion(params, 1, 0, &pose_gt, &x);
	add_focal(2.0, &pose_gt, &x);


	// Transform such that make X(3,:) = 0
	Vector3d t0 = X.rowwise().mean();
	X.colwise() -= t0;

	JacobiSVD<Matrix<double, 3, Dynamic>> svd(X, ComputeThinU);
	Matrix<double, 3, 3> R0 = svd.matrixU();
	if (R0.determinant() < 0)
		R0 *= -1.0;
	X = R0.transpose() * X;
	pose_gt.t = pose_gt.t + pose_gt.R * t0;
	pose_gt.R = pose_gt.R * R0;
	X.row(2).array() = 0.0;

	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses, true, true);

	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err < TOL_POSE);
}

bool test_larsson19_distortion_2_0() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12, 0.034 };
	larsson_iccv19::Solver<2, 0, true> estimator;

	generate_scene_and_image(estimator.minimal_sample_size(), 2, 20, 70, false, &pose_gt, &x, &X);
	add_rational_distortion(params, 2, 0, &pose_gt, &x);
	add_focal(2.0, &pose_gt, &x);

	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses, true, true);

	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err < TOL_POSE);
}

bool test_larsson19_distortion_3_0() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12, 0.034, -0.0056 };
	larsson_iccv19::Solver<3,0,true> estimator;

	generate_scene_and_image(estimator.minimal_sample_size(), 2, 20, 70, false, &pose_gt, &x, &X);
	add_rational_distortion(params, 3, 0, &pose_gt, &x);
	add_focal(2.0, &pose_gt, &x);

	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses, true, true);

	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err < TOL_POSE);
}


bool test_larsson19_distortion_3_3() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12, 0.034, -0.0056, 0.13, -0.012, 0.003 };
	larsson_iccv19::Solver<3, 3, true> estimator;

	generate_scene_and_image(estimator.minimal_sample_size(), 2, 20, 70, false, &pose_gt, &x, &X);
	add_rational_distortion(params, 3, 3, &pose_gt, &x);
	add_focal(2.0, &pose_gt, &x);

	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses, true, false);

	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err < TOL_POSE);
}

bool test_larsson19_precond() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12, 0.034, -0.0056 };
	larsson_iccv19::Solver<3, 0, true> estimator;

	//estimator.normalize_image_coord = true;
	//estimator.normalize_world_coord = true;
	
	generate_scene_and_image(estimator.minimal_sample_size(), 2, 20, 70, false, &pose_gt, &x, &X, 100.0);
	add_rational_distortion(params, 3, 0, &pose_gt, &x);
	add_focal(2000.0, &pose_gt, &x);

	estimator.use_precond = false;
	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);
	double err1 = minimum_pose_distance(pose_gt, poses, true, true);

//	std::cout << "precond=false, error=" << err1 << "\n";

	estimator.use_precond = true;
	n_sols = estimator.estimate(x, X, &poses);
	double err2 = minimum_pose_distance(pose_gt, poses, true, true);

//	std::cout << "precond=true, error=" << err2 << "\n";

	if (err2 > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err2 < TOL_POSE);
}


bool test_larsson19_refinement() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params6 = { -0.12, -0.034, 0.0056, 0.13, -0.023, 0.0035 };

	larsson_iccv19::Solver<3, 3, true> estimator;

	estimator.root_refinement = true;

	generate_scene_and_image(estimator.minimal_sample_size(), 2, 20, 70, false, &pose_gt, &x, &X, 1.0);
	add_rational_distortion(params6, 3, 3, &pose_gt, &x);
	add_focal(2000.0, &pose_gt, &x);

	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses, true, false);

	//	std::cout << "precond=true, error=" << err2 << "\n";


	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err < TOL_POSE);

}

bool test_larsson19_undistortion_1_0() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12 };
	larsson_iccv19::Solver<1, 0, false> estimator;

	generate_scene_and_image(estimator.minimal_sample_size(), 2, 20, 70, false, &pose_gt, &x, &X);
	add_rational_undistortion(params, 1, 0, &pose_gt, &x);
	add_focal(2.0, &pose_gt, &x);

	std::vector<Camera> poses;
	int n_sols = estimator.estimate(x, X, &poses);

	double err = minimum_pose_distance(pose_gt, poses, true, true);

	if (err > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	return (err < TOL_POSE);
}



bool test_radial_upgrade() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 2, Dynamic> x1;
	Matrix<double, 2, Dynamic> x2;

	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params3 = { -0.12, -0.034, 0.0056 };

	kukelova_iccv13::Solver estimator_U03(3);
	larsson_iccv19::Solver<3, 0, true> estimator_D30;

	generate_scene_and_image(estimator_D30.minimal_sample_size(), 2, 20, 70, false, &pose_gt, &x, &X, 1.0);
	x1 = x; x2 = x;

	add_rational_distortion(params3, 3, 0, &pose_gt, &x1);
	add_focal(2000.0, &pose_gt, &x1);

	add_rational_undistortion(params3, 0, 3, &pose_gt, &x2);
	add_focal(2000.0, &pose_gt, &x2);

	// Transform to current coordinate system
	std::vector<Camera> poses;

	X = pose_gt.R * X;
	X.row(0).array() += pose_gt.t(0);
	X.row(1).array() += pose_gt.t(1);

	pose_gt.R = Matrix3d::Identity();
	pose_gt.t(0) = 0.0;
	pose_gt.t(1) = 0.0;

	estimator_D30.use_radial_solver = false;
	estimator_D30.normalize_world_coord = true;
	estimator_D30.center_world_coord = false;
	estimator_D30.check_chirality = false;
	estimator_D30.check_reprojection_error = false;

	estimator_U03.use_radial_solver = false;
	estimator_U03.normalize_world_coord = true;
	estimator_U03.center_world_coord = false;
	estimator_U03.check_chirality = false;
	estimator_U03.check_reprojection_error = false;


	int n_sols1 = estimator_D30.estimate(x1, X, &poses);
	double err1 = minimum_pose_distance(pose_gt, poses, true, false);
	if (err1 > TOL_POSE)
		debug_print_poses(pose_gt, poses);

	int n_sols2 = estimator_U03.estimate(x2, X, &poses);
	double err2 = minimum_pose_distance(pose_gt, poses, true, false);
	if (err2 > TOL_POSE)
		debug_print_poses(pose_gt, poses);


	return (err1 < TOL_POSE && err2 < TOL_POSE);
}

int run_solver_benchmark() {

	int trials = 1000;
	//int trials = 10; // To speed things up during debug.
	std::vector<double> params1 = { -0.12 };
	std::vector<double> params2 = { -0.12, -0.034 };
	std::vector<double> params3 = { -0.12, -0.034, 0.0056 };
	std::vector<double> params6 = { -0.12, -0.034, 0.0056, 0.13, -0.023, 0.0035 };

	larsson_iccv19::Solver<1, 0, true> estimator_D10;
	larsson_iccv19::Solver<2, 0, true> estimator_D20;
	larsson_iccv19::Solver<3, 0, true> estimator_D30;
	larsson_iccv19::Solver<3, 3, true> estimator_D33;

	larsson_iccv19::Solver<1, 0, false> estimator_U10;
	larsson_iccv19::Solver<2, 0, false> estimator_U20;
	larsson_iccv19::Solver<3, 0, false> estimator_U30;
	larsson_iccv19::Solver<3, 3, false> estimator_U33;


	larsson_iccv17::NonPlanarSolver estimator_L17_U01;
	bujnak_accv10::NonPlanarSolver estimator_B10_U01;
	kukelova_iccv13::Solver estimator_K13_U01(1);
	kukelova_iccv13::Solver estimator_K13_U02(2);
	kukelova_iccv13::Solver estimator_K13_U03(3);

	auto addUndistortion01 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 0, 1, p, x); };
	auto addUndistortion02 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 0, 2, p, x); };
	auto addUndistortion03 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 0, 3, p, x); };
	
	auto addUndistortion10 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 1, 0, p, x); };
	auto addUndistortion20 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 2, 0, p, x); };
	auto addUndistortion30 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 3, 0, p, x); };
	auto addUndistortion33 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 3, 3, p, x); };

	auto addDistortion10 = [](std::vector<double> params, Camera *p, Matrix<double, 2, Dynamic> *x) {	add_rational_distortion(params, 1, 0, p, x); };
	auto addDistortion20 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_distortion(params, 2, 0, p, x); };
	auto addDistortion30 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_distortion(params, 3, 0, p, x); };
	auto addDistortion33 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_distortion(params, 3, 3, p, x); };

	benchmark_solver("KUKELOVA_13_U01", &estimator_K13_U01, addUndistortion01, params1, trials);
	benchmark_solver("KUKELOVA_13_U02", &estimator_K13_U02, addUndistortion02, params2, trials);
	benchmark_solver("KUKELOVA_13_U03", &estimator_K13_U03, addUndistortion03, params3, trials);
	benchmark_solver("LARSSON_17_U01", &estimator_L17_U01, addUndistortion01, params1, trials);
	benchmark_solver("BUJNAK_10_U01", &estimator_B10_U01, addUndistortion01, params1, trials);
	benchmark_solver("LARSSON_19_D10", &estimator_D10, addDistortion10, params1, trials);
	benchmark_solver("LARSSON_19_D20", &estimator_D20, addDistortion20, params2, trials);
	benchmark_solver("LARSSON_19_D30", &estimator_D30, addDistortion30, params3, trials);
	benchmark_solver("LARSSON_19_D33", &estimator_D33, addDistortion33, params6, trials);
	benchmark_solver("LARSSON_19_U10", &estimator_U10, addUndistortion10, params1, trials);

	return 0;
}

int run_solver_benchmark2() {

	int trials = 1000;
	//int trials = 10; // To speed things up during debug.
	std::vector<double> params1 = { -0.12 };
	std::vector<double> params2 = { -0.12, -0.034 };
	std::vector<double> params3 = { -0.12, -0.034, 0.0056 };
	std::vector<double> params6 = { -0.12, -0.034, 0.0056, -0.08, -0.024, 0.0026 };

	larsson_iccv19::Solver<1, 0, true> estimator_D10;
	larsson_iccv19::Solver<2, 0, true> estimator_D20;
	larsson_iccv19::Solver<3, 0, true> estimator_D30;
	larsson_iccv19::Solver<3, 3, true> estimator_D33;

	
	auto addUndistortion01 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 0, 1, p, x); };
	auto addUndistortion02 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 0, 2, p, x); };
	auto addUndistortion03 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 0, 3, p, x); };
	auto addDistortion10 = [](std::vector<double> params, Camera *p, Matrix<double, 2, Dynamic> *x) {	add_rational_distortion(params, 1, 0, p, x); };
	auto addDistortion20 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_distortion(params, 2, 0, p, x); };
	auto addDistortion30 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_distortion(params, 3, 0, p, x); };
	auto addDistortion33 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_distortion(params, 3, 3, p, x); };

	bool use_rescaling = true;
	bool use_precond = true;
	bool use_qz_solver = false;

	//FLIPPYBOOL(use_rescaling) {
	//	FLIPPYBOOL(use_qz_solver) {
	FLIPPYBOOL(normalize_image) {
		FLIPPYBOOL(normalize_world) {
			FLIPPYBOOL(center_world) {

			//FLIPPYBOOL(use_precond) {

				estimator_D10.use_rescaling = use_rescaling;
				estimator_D20.use_rescaling = use_rescaling;
				estimator_D30.use_rescaling = use_rescaling;
				estimator_D33.use_rescaling = use_rescaling;

				estimator_D10.use_qz_solver = use_qz_solver;
				estimator_D20.use_qz_solver = use_qz_solver;
				estimator_D30.use_qz_solver = use_qz_solver;
				estimator_D33.use_qz_solver = use_qz_solver;
				estimator_D10.normalize_image_coord = normalize_image;
				estimator_D20.normalize_image_coord = normalize_image;
				estimator_D30.normalize_image_coord = normalize_image;
				estimator_D33.normalize_image_coord = normalize_image;

				estimator_D10.normalize_world_coord = normalize_world;
				estimator_D20.normalize_world_coord = normalize_world;
				estimator_D30.normalize_world_coord = normalize_world;
				estimator_D33.normalize_world_coord = normalize_world;;

				estimator_D10.center_world_coord = center_world;
				estimator_D20.center_world_coord = center_world;
				estimator_D30.center_world_coord = center_world;
				estimator_D33.center_world_coord = center_world;

				estimator_D10.use_precond = use_precond;
				estimator_D20.use_precond = use_precond;
				estimator_D30.use_precond = use_precond;
				estimator_D33.use_precond = use_precond;

				std::cout << "Benchmark: use_rescaling=" << use_rescaling << ", use_qz_solver=" << use_qz_solver << ", norm_image=" << normalize_image << ", norm_world=" << normalize_world << ", center_world=" << center_world << ", use_precond=" << use_precond << "\n";
				benchmark_solver("LARSSON_19_D10", &estimator_D10, addDistortion10, params1, trials);
				benchmark_solver("LARSSON_19_D20", &estimator_D20, addDistortion20, params2, trials);
				benchmark_solver("LARSSON_19_D30", &estimator_D30, addDistortion30, params3, trials);
				benchmark_solver("LARSSON_19_D33", &estimator_D33, addDistortion33, params6, trials);

				std::cout << "------------------------\n";
			}
		}
	}

	return 0;
}


int run_solver_benchmark3() {

	int trials = 1000;
	//int trials = 10; // To speed things up during debug.
	std::vector<double> params1 = { -0.12 };
	std::vector<double> params2 = { -0.12, -0.034 };
	std::vector<double> params3 = { -0.12, -0.034, 0.0056 };
	std::vector<double> params6 = { -0.12, -0.034, 0.0056, 0.13, -0.023, 0.0035 };

	larsson_iccv19::Solver<1, 0, true> estimator_D10;
	larsson_iccv19::Solver<2, 0, true> estimator_D20;
	larsson_iccv19::Solver<3, 0, true> estimator_D30;
	larsson_iccv19::Solver<3, 3, true> estimator_D33;
	larsson_iccv19::Solver<1, 0, false> estimator_U10;


	auto addUndistortion01 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 0, 1, p, x); };
	auto addUndistortion02 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 0, 2, p, x); };
	auto addUndistortion03 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 0, 3, p, x); };
	auto addUndistortion10 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_undistortion(params, 1, 0, p, x); };

	auto addDistortion10 = [](std::vector<double> params, Camera *p, Matrix<double, 2, Dynamic> *x) {	add_rational_distortion(params, 1, 0, p, x); };
	auto addDistortion20 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_distortion(params, 2, 0, p, x); };
	auto addDistortion30 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_distortion(params, 3, 0, p, x); };
	auto addDistortion33 = [](std::vector<double> params, Camera* p, Matrix<double, 2, Dynamic>* x) {	add_rational_distortion(params, 3, 3, p, x); };


	FLIPPYBOOL(root_refinement) {

		estimator_D10.root_refinement = root_refinement;
		estimator_D20.root_refinement = root_refinement;
		estimator_D30.root_refinement = root_refinement;
		estimator_D33.root_refinement = root_refinement;
		estimator_U10.root_refinement = root_refinement;

		std::cout << "Benchmark: root_refinement=" << root_refinement << "\n";
		benchmark_solver("LARSSON_19_D10", &estimator_D10, addDistortion10, params1, trials);
		benchmark_solver("LARSSON_19_D20", &estimator_D20, addDistortion20, params2, trials);
		benchmark_solver("LARSSON_19_D30", &estimator_D30, addDistortion30, params3, trials);
		benchmark_solver("LARSSON_19_D33", &estimator_D33, addDistortion33, params6, trials);
		benchmark_solver("LARSSON_19_U10", &estimator_U10, addUndistortion10, params1, trials);

		std::cout << "------------------------\n";
	
	}

	return 0;
}


bool test_refinement_dist() {

	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12, 0.034, -0.0056, 0.01, 0.02, 0.003 };

	generate_scene_and_image(10, 2, 20, 70, false, &pose_gt, &x, &X);
	add_rational_distortion(params, 3, 3, &pose_gt, &x);
	add_focal(2.0, &pose_gt, &x);
	add_noise(0.001, &x);

	Camera p = pose_gt;

	Vector3d tmp;
	tmp.setRandom();
	tmp *= 0.01;
	Quaterniond qq;
	qq.w() = 1.0;
	qq.x() = tmp(0);
	qq.y() = tmp(1);
	qq.z() = tmp(2);
	qq.normalize();
	
	p.R = qq.toRotationMatrix() * p.R;

	p.t(0) -= 0.02;
	p.t(1) -= 0.01;
	p.t(2) += 0.01;
	p.focal += 0.01;
	p.dist_params[0] += 0.001;
	p.dist_params[1] += 0.0001;
	p.dist_params[2] += 0.00001;
	p.dist_params[3] += 0.001;
	p.dist_params[4] += 0.0001;
	p.dist_params[5] += 0.00001;

	double err_before = (pose_gt.R - p.R).norm() + (pose_gt.t - p.t).norm() + abs(pose_gt.focal - p.focal);
	refinement_dist(x, X, p, 3, 3);
	double err_after = (pose_gt.R - p.R).norm() + (pose_gt.t - p.t).norm() + abs(pose_gt.focal - p.focal);
	std::cout << "test_refinement_dist: error = " << err_before << " --> " << err_after << "\n";

	return err_after < err_before;
}


bool test_refinement_undist() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12, 0.034, -0.0056, 0.1, 0.01, 0.001 };

	generate_scene_and_image(10, 2, 20, 70, false, &pose_gt, &x, &X);
	add_rational_undistortion(params, 3, 3, &pose_gt, &x);
	add_focal(2.0, &pose_gt, &x);
	add_noise(0.001, &x);

	Camera p = pose_gt;

	Vector3d tmp;
	tmp.setRandom();
	tmp *= 0.01;
	Quaterniond qq;
	qq.w() = 1.0;
	qq.x() = tmp(0);
	qq.y() = tmp(1);
	qq.z() = tmp(2);
	qq.normalize();

	p.R = qq.toRotationMatrix() * p.R;
	p.t(0) -= 0.02;
	p.t(1) -= 0.01;
	p.t(2) += 0.01;
	p.focal += 0.01;
	p.dist_params[0] += 0.001;
	p.dist_params[1] += 0.0001;
	p.dist_params[2] += 0.00001;
	p.dist_params[3] += 0.001;
	p.dist_params[4] += 0.0001;
	p.dist_params[5] += 0.00001;
	double err_before = (pose_gt.R - p.R).norm() + (pose_gt.t - p.t).norm() + abs(pose_gt.focal - p.focal);

	refinement_undist(x, X, p, 3, 3);

	double err_after = (pose_gt.R - p.R).norm() + (pose_gt.t - p.t).norm() + abs(pose_gt.focal - p.focal);
	std::cout << "test_refinement_undist: error = " << err_before << " --> " << err_after << "\n";

	return err_after < err_before;
}


bool test_refinement_undist_division() {
	Matrix<double, 2, Dynamic> x;
	Matrix<double, 3, Dynamic> X;
	Camera pose_gt;

	std::vector<double> params = { -0.12 };

	generate_scene_and_image(10, 2, 20, 70, false, &pose_gt, &x, &X);
	add_rational_undistortion(params, 0, 1, &pose_gt, &x);
	add_focal(2.0, &pose_gt, &x);
	add_noise(0.0001, &x);

	Camera p = pose_gt;

	Vector3d tmp;
	tmp.setRandom();
	tmp *= 0.01;
	Quaterniond qq;
	qq.w() = 1.0;
	qq.x() = tmp(0);
	qq.y() = tmp(1);
	qq.z() = tmp(2);
	qq.normalize();

	p.R = qq.toRotationMatrix() * p.R;
	p.t(0) -= 0.02;
	p.t(1) -= 0.01;
	p.t(2) += 0.01;
	p.focal += 0.01;
	p.dist_params[0] += 0.001;
	double err_before = (pose_gt.R - p.R).norm() + (pose_gt.t - p.t).norm() + abs(pose_gt.focal - p.focal);

	refinement_undist(x, X, p, 0, 1);

	double err_after = (pose_gt.R - p.R).norm() + (pose_gt.t - p.t).norm() + abs(pose_gt.focal - p.focal);
	std::cout << "test_refinement_undist: error = " << err_before << " --> " << err_after << "\n";

	return err_after < err_before;
}

int main() {
	std::cout << "Running tests...\n\n";
	srand((unsigned int)time(0));	
	srand(2.0);

	int passed = 0;
	int num_tests = 0;
	
	TEST(test_kukelova_1d_radial);
	TEST(test_kukelova_solver_1p_div);	
	TEST(test_kukelova_solver_3p_div);
	TEST(test_larsson17_solver_1p_div);
	TEST(test_bujnak10_solver_1p_div);
	TEST(test_planar_oskarsson);
	TEST(test_larsson19_distortion_1_0);
	TEST(test_larsson19_distortion_2_0);
	TEST(test_larsson19_distortion_3_0);
	TEST(test_larsson19_distortion_3_3);
	TEST(test_larsson19_undistortion_1_0);
	TEST(test_larsson19_distortion_1_0_planar);
	TEST(test_larsson19_precond);
	TEST(test_1param_div_model);
	TEST(test_rational_distortion);	
	TEST(test_radial_upgrade);
	TEST(test_refinement_dist);
	TEST(test_refinement_undist);
	TEST(test_refinement_undist_division);

	std::cout << "\nDone! Passed " << passed << "/" << num_tests << " tests.\n";

	
	run_solver_benchmark();

}
