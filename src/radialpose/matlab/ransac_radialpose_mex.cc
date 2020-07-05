#include "mex.h"
#include <Eigen/Dense>
#include "../radialpose.h"
#include "../misc/ransac_estimator.h"
#include <RansacLib/ransac.h>

using namespace radialpose;

void print_usage() {
	mexPrintf("[R,t,f,params] = ransac_radialpose_mex(x,X,solver,tol,[min_iter],[max_iter]);\n");
	mexPrintf(" Solvers:\n");
	mexPrintf("  1 - D(1,0) - 5p -- Larsson et al.  ICCV 2019\n");
	mexPrintf("  2 - D(2,0) - 5p -- Larsson et al.  ICCV 2019\n");
	mexPrintf("  3 - D(3,0) - 5p -- Larsson et al.  ICCV 2019  (Minimal)\n");
	mexPrintf("  4 - D(3,3) - 8p -- Larsson et al.  ICCV 2019\n");
	mexPrintf("  5 - U(1,0) - 5p -- Larsson et al.  ICCV 2019\n");
	mexPrintf("  6 - U(0,1) - 4p -- Larsson et al.  ICCV 2017  (Minimal, Non-planar)\n");
	mexPrintf("  7 - U(0,1) - 4p -- Bujnak et al.   ACCV 2010  (Minimal, Non-planar)\n");
	mexPrintf("  8 - U(0,1) - 5p -- Kukelova et al. ICCV 2013\n");
	mexPrintf("  9 - U(0,2) - 5p -- Kukelova et al. ICCV 2013\n");
	mexPrintf(" 10 - U(0,3) - 5p -- Kukelova et al. ICCV 2013  (Minimal)\n");
	mexPrintf(" 11 - U(0,1) - 4p -- Oskarsson       arxiv 2018 (Minimal, Planar)\n");
	mexPrintf(" 12 - N/A    - 5p -- Kukelova et al. ICCV 2013  (Minimal, 1D Radial)\n\n");
}

void save_pose(int nlhs, mxArray *plhs[], Camera pose) {
	int n_sols = 1;
	int n_params = 0;
	if (n_sols > 0)
		n_params = pose.dist_params.size();

	if (nlhs >= 1) {
		plhs[0] = mxCreateDoubleMatrix(3, 3, mxREAL);
		double *p = mxGetPr(plhs[0]);		
		for (int j = 0; j < 9; ++j)
			p[j] = pose.R(j);
	}
	if (nlhs >= 2) {
		plhs[1] = mxCreateDoubleMatrix(3, 1, mxREAL);
		double *p = mxGetPr(plhs[1]);		
		for (int j = 0; j < 3; ++j)
			p[j] = pose.t(j);
	}
	if (nlhs >= 3) {
		plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
		double *p = mxGetPr(plhs[2]);		
		*p = pose.focal;

	}
	if (nlhs >= 4) {
		plhs[3] = mxCreateDoubleMatrix(n_params, 1, mxREAL);
		double *p = mxGetPr(plhs[3]);		
		for (int j = 0; j < n_params; ++j)
			p[j] = pose.dist_params[j];
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	if (nrhs < 4 || nrhs > 7) {
		print_usage();
		mexErrMsgTxt("Incorrect number of input arguments.");
	}
	if (nlhs > 4) {
		print_usage();
		mexErrMsgTxt("Wrong number of output arguments.");
	}

	if (mxGetM(prhs[0]) != 2) {
		print_usage();
		mexErrMsgTxt("First input must be 2 x N matrix.");
	}
	if (mxGetM(prhs[1]) != 3) {
		print_usage();
		mexErrMsgTxt("Second input must be 3 x N matrix.");
	}
	if (mxGetN(prhs[0]) != mxGetN(prhs[1])) {
		print_usage();
		mexErrMsgTxt("Not the same number of 2D points and 3D points.");
	}

	double tol = 5.0;
	if (nrhs >= 4) {
		tol = mxGetScalar(prhs[3]);
	}

	ransac_lib::LORansacOptions options;
	options.squared_inlier_threshold_ = tol * tol;

	options.final_least_squares_ = true;

	if (nrhs >= 5)
		options.min_num_iterations_ = static_cast<int>(mxGetScalar(prhs[4]));
	if (nrhs >= 6)
		options.max_num_iterations_ = static_cast<int>(mxGetScalar(prhs[5]));

	double damp_factor = 0.0;
	if (nrhs >= 7) {
		damp_factor = mxGetScalar(prhs[6]);
	}

	Eigen::Matrix<double, 2, Eigen::Dynamic> x = Eigen::Map<Eigen::Matrix<double, 2, Eigen::Dynamic>>(mxGetPr(prhs[0]), 2, mxGetN(prhs[0]));
	Eigen::Matrix<double, 3, Eigen::Dynamic> X = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(mxGetPr(prhs[1]), 3, mxGetN(prhs[1]));
	int solver_idx = static_cast<int>(mxGetScalar(prhs[2]));

	ransac_lib::RansacStatistics ransac_stats;
	int inliers = 0;
	Camera best_model;
	best_model.R = Eigen::Matrix3d::Identity();
	best_model.t = Eigen::Vector3d::Zero();


	if (solver_idx == 1) { // D(1,0)	
		radialpose::larsson_iccv19::Solver<1, 0, true> estimator;
		estimator.damp_factor = damp_factor;
		radialpose::RansacEstimator<larsson_iccv19::Solver<1, 0, true>> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<larsson_iccv19::Solver<1, 0, true>>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	} else if (solver_idx == 2) { // D(2,0)
		radialpose::larsson_iccv19::Solver<2, 0, true> estimator;
		estimator.damp_factor = damp_factor;
		radialpose::RansacEstimator<larsson_iccv19::Solver<2, 0, true>> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<larsson_iccv19::Solver<2, 0, true>>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	} else if (solver_idx == 3) { // D(3,0)
		radialpose::larsson_iccv19::Solver<3, 0, true> estimator;		
		estimator.damp_factor = damp_factor;
		radialpose::RansacEstimator<larsson_iccv19::Solver<3, 0, true>> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<larsson_iccv19::Solver<3, 0, true>>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	} else if (solver_idx == 4) { // D(3,3)
		radialpose::larsson_iccv19::Solver<3, 3, true> estimator;
		estimator.damp_factor = damp_factor;
		radialpose::RansacEstimator<larsson_iccv19::Solver<3, 3, true>> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<larsson_iccv19::Solver<3, 3, true>>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	} else if (solver_idx == 5) { // U(1,0)
		radialpose::larsson_iccv19::Solver<1, 0, false> estimator;
		estimator.damp_factor = damp_factor;
		radialpose::RansacEstimator<larsson_iccv19::Solver<1, 0, false>> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<larsson_iccv19::Solver<1, 0, false>>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	} else if (solver_idx == 6) { // U(0,1)
		radialpose::larsson_iccv17::NonPlanarSolver estimator;
		radialpose::RansacEstimator<larsson_iccv17::NonPlanarSolver> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<larsson_iccv17::NonPlanarSolver>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	} else if (solver_idx == 7) {  // U(0,1)
		radialpose::bujnak_accv10::NonPlanarSolver estimator;
		radialpose::RansacEstimator<bujnak_accv10::NonPlanarSolver> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<bujnak_accv10::NonPlanarSolver>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	} else if (solver_idx == 8) {  // U(0,1)
		radialpose::kukelova_iccv13::Solver estimator(1);	
		radialpose::RansacEstimator<kukelova_iccv13::Solver> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<kukelova_iccv13::Solver>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	} else if (solver_idx == 9) {  // U(0,2)
		radialpose::kukelova_iccv13::Solver estimator(2);
		radialpose::RansacEstimator<kukelova_iccv13::Solver> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<kukelova_iccv13::Solver>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);


	} else if (solver_idx == 10) { // U(0,3)
		radialpose::kukelova_iccv13::Solver estimator(3);
		radialpose::RansacEstimator<kukelova_iccv13::Solver> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<kukelova_iccv13::Solver>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);


	} else if (solver_idx == 11) {  // U(0,1)
		radialpose::oskarsson_arxiv18::PlanarSolver estimator;
		radialpose::RansacEstimator<oskarsson_arxiv18::PlanarSolver> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<oskarsson_arxiv18::PlanarSolver>> lomsac;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);


	} else if (solver_idx == 12) {  // N/A
		radialpose::kukelova_iccv13::Radial1DSolver estimator;
		radialpose::RansacEstimator<kukelova_iccv13::Radial1DSolver> solver(x, X, estimator);
		ransac_lib::LocallyOptimizedMSAC<Camera,
			std::vector<Camera>,
			RansacEstimator<kukelova_iccv13::Radial1DSolver>> lomsac;
		solver.use_local_opt = false;
		inliers = lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

	} else {
		print_usage();
		mexErrMsgTxt("Solver NYI.\n");
	}

	save_pose(nlhs, plhs, best_model);

}
