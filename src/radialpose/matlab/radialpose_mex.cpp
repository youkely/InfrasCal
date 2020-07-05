#include "mex.h"
#include <Eigen/Dense>
#include "../radialpose.h"

void print_usage() {
	mexPrintf("[R,t,f,params] = radialpose_mex(x,X,solver,[upgrade_only]);\n");
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
	mexPrintf("If upgrade_only=true we don't solve for the 1D radial camera and only try to upgrade.\n This is only for the two-step solvers.\n");
}

void save_poses(int nlhs, mxArray *plhs[], std::vector<radialpose::Camera> &poses) {
	int n_sols = poses.size();
	int n_params = 0;
	if (n_sols > 0)
		n_params = poses[0].dist_params.size();

	if (nlhs >= 1) {
		plhs[0] = mxCreateDoubleMatrix(9, n_sols, mxREAL);
		double *p = mxGetPr(plhs[0]);
		for (int i = 0; i < n_sols; ++i)
			for (int j = 0; j < 9; ++j)
				p[9 * i + j] = poses[i].R(j);
	}
	if (nlhs >= 2) {
		plhs[1] = mxCreateDoubleMatrix(3, n_sols, mxREAL);
		double *p = mxGetPr(plhs[1]);
		for (int i = 0; i < n_sols; ++i)
			for (int j = 0; j < 3; ++j)
				p[3 * i + j] = poses[i].t(j);
	}
	if (nlhs >= 3) {
		plhs[2] = mxCreateDoubleMatrix(1, n_sols, mxREAL);
		double *p = mxGetPr(plhs[2]);
		for (int i = 0; i < n_sols; ++i)
			p[i] = poses[i].focal;

	}
	if (nlhs >= 4) {
		plhs[3] = mxCreateDoubleMatrix(n_params, n_sols, mxREAL);
		double *p = mxGetPr(plhs[3]);
		for (int i = 0; i < n_sols; ++i)
			for (int j = 0; j < n_params; ++j)
				p[n_params * i + j] = poses[i].dist_params[j];
	}
}

void mexFunction(int nlhs,mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	if (nrhs < 3 || nrhs > 5) {
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

	bool use_radial_solver = true;
	if (nrhs >= 4 && mxGetScalar(prhs[3]) != 0.0) {
		use_radial_solver = false;
	}

	double damp_factor = 0.0;
	if (nrhs >= 5) {
		damp_factor = mxGetScalar(prhs[4]);
	}


	// If we don't call the radial solver we don't normalize world coord system
	bool center_world_coordinates = use_radial_solver;	

	Eigen::Matrix<double, 2, Eigen::Dynamic> x = Eigen::Map<Eigen::Matrix<double, 2, Eigen::Dynamic>>(mxGetPr(prhs[0]), 2, mxGetN(prhs[0]));
	Eigen::Matrix<double, 3, Eigen::Dynamic> X = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(mxGetPr(prhs[1]), 3, mxGetN(prhs[1]));

	if(x.cols() > 8) {
		// This is to avoid crash when solver is called with more than 8 points
		x = x.block(0,0,2,8).eval();
		X = X.block(0,0,3,8).eval();
	}

	int solver_idx = static_cast<int>(mxGetScalar(prhs[2]));

	std::vector<radialpose::Camera> poses;

	if (solver_idx == 1) { // D(1,0)
		radialpose::larsson_iccv19::Solver<1, 0, true> estimator;
		estimator.use_radial_solver = use_radial_solver;
		estimator.center_world_coord = center_world_coordinates;
		estimator.damp_factor = damp_factor;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 2) { // D(2,0)
		radialpose::larsson_iccv19::Solver<2, 0, true> estimator;
		estimator.use_radial_solver = use_radial_solver;
		estimator.center_world_coord = center_world_coordinates;
		estimator.damp_factor = damp_factor;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 3) { // D(3,0)
		radialpose::larsson_iccv19::Solver<3, 0, true> estimator;
		estimator.use_radial_solver = use_radial_solver;
		estimator.center_world_coord = center_world_coordinates;
		estimator.damp_factor = damp_factor;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 4) { // D(3,3)
		radialpose::larsson_iccv19::Solver<3, 3, true> estimator;
		estimator.use_radial_solver = use_radial_solver;
		estimator.center_world_coord = center_world_coordinates;
		estimator.damp_factor = damp_factor;		
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 5) { // U(1,0)
		radialpose::larsson_iccv19::Solver<1, 0, false> estimator;
		estimator.use_radial_solver = use_radial_solver;
		estimator.center_world_coord = center_world_coordinates;
		estimator.damp_factor = damp_factor;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 6) { // U(0,1)
		radialpose::larsson_iccv17::NonPlanarSolver estimator;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 7) {  // U(0,1)
		radialpose::bujnak_accv10::NonPlanarSolver estimator;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 8) {  // U(0,1)
		radialpose::kukelova_iccv13::Solver estimator(1);
		estimator.use_radial_solver = use_radial_solver;
		estimator.center_world_coord = center_world_coordinates;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 9) {  // U(0,2)
		radialpose::kukelova_iccv13::Solver estimator(2);
		estimator.use_radial_solver = use_radial_solver;
		estimator.center_world_coord = center_world_coordinates;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 10) { // U(0,3)
		radialpose::kukelova_iccv13::Solver estimator(3);
		estimator.use_radial_solver = use_radial_solver;
		estimator.center_world_coord = center_world_coordinates;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 11) {  // U(0,1)
		radialpose::oskarsson_arxiv18::PlanarSolver estimator;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else if (solver_idx == 12) {  // N/A
		radialpose::kukelova_iccv13::Radial1DSolver estimator;
		estimator.estimate(x, X, &poses);
		save_poses(nlhs, plhs, poses);

	} else {
		print_usage();
		mexErrMsgTxt("Solver NYI.\n");
	}
}
