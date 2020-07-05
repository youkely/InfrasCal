#include "distortion.h"

using namespace Eigen;

//#include <iostream>

static const double TOL_ITERATIVE = 1e-10;

void radialpose::inverse_1param_division_model(double lambda, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x0, Eigen::Matrix<double, 2, Eigen::Dynamic>* x1) {

	if (lambda == 0.0) {
		*x1 = x0;
		return;
	}


	Array<double, 1, Dynamic> ru2 = x0.colwise().squaredNorm();
	Array<double, 1, Dynamic> ru = ru2.sqrt();

	double dist_sign = 1.0;
	if (lambda < 0)
		dist_sign = -1.0;

	Array<double, 1, Dynamic> rd;
	rd.resizeLike(ru);
	rd = (0.5 / lambda) / ru - dist_sign * ((0.25 / (lambda * lambda)) / ru2 - 1 / lambda).sqrt();
//  rd = 1 / 2 / kappa. / ru - sign(kappa) * sqrt(1 / 4 / kappa ^ 2. / ru2 - 1 / kappa);

	rd /= ru;
	x1->resizeLike(x0);
	x1->row(0) = x0.row(0).array() * rd;
	x1->row(1) = x0.row(1).array() * rd;

}

void radialpose::forward_1param_division_model(double lambda, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x0, Eigen::Matrix<double, 2, Eigen::Dynamic>* x1) {

	Array<double, 1, Dynamic> r2 = x0.colwise().squaredNorm();
	r2 *= lambda;
	r2 += 1.0;

	x1->resizeLike(x0);
	x1->row(0) = x0.row(0).array() / r2;
	x1->row(1) = x0.row(1).array() / r2;
}




void radialpose::forward_rational_model(const std::vector<double> &params, int np, int nd,
	const Eigen::Matrix<double, 2, Eigen::Dynamic> &x0, Eigen::Matrix<double, 2, Eigen::Dynamic>* x1)
{

	Array<double, 1, Dynamic> r2 = x0.colwise().squaredNorm();

	Array<double, 1, Dynamic> s_num(1, r2.cols());
	Array<double, 1, Dynamic> s_den(1, r2.cols());
	s_num.setOnes();
	s_den.setOnes();

	Array<double, 1, Dynamic> rk = r2;
	for (int i = 0; i < np; ++i) {
		s_num += params[i] * rk;
		rk *= r2;
	}
	// TODO: Optimize this code to avoid redundant computations of rk
	rk = r2;
	for (int i = 0; i < nd; ++i) {
		s_den += params[np+i] * rk;
		rk *= r2;
	}

	s_num /= s_den;

	x1->resizeLike(x0);
	x1->row(0) = x0.row(0).array() * s_num;
	x1->row(1) = x0.row(1).array() * s_num;
}

void radialpose::inverse_rational_model(const std::vector<double> &params, int n_p, int n_d,
	const Eigen::Matrix<double, 2, Eigen::Dynamic> &x0, Eigen::Matrix<double, 2, Eigen::Dynamic>* x1)
{

	x1->resizeLike(x0);


	// support at most degree 12!
	double rr[12];
	rr[0] = 1.0;

	// f = pp / qq 
	double r, r0, pp, pp_d, qq, qq_d, g, g_p;
		
	int max_deg = std::max(n_p, n_d);

	for (int i = 0; i < x0.cols(); i++) {
		r0 = x0.col(i).norm();
		r = r0;

		int iter;
		for (iter = 0; iter < 100; iter++) {

			// compute powers
			rr[1] = r * r;
			for (int k = 2; k <= max_deg; k++)
				rr[k] = rr[k - 1] * rr[1];

			// compute f = p / q and derivatives
			pp = 1; pp_d = 0;
			for (int k = 0; k < n_p; k++) {
				pp += params[k] * rr[k + 1];
				pp_d += 2 * (k + 1) * params[k] * rr[k] * r;
			}
			qq = 1; qq_d = 0;
			for (int k = 0; k < n_d; k++) {
				qq += params[n_p + k] * rr[k + 1];
				qq_d += 2 * (k + 1) * params[n_p + k] * rr[k] * r;
			}

			// compute residuals
			g = pp * r - r0 * qq;            
			g_p = pp + pp_d * r - r0 * qq_d;

			//g = pp / qq * r - r0;
			//g_p = (pp_d * r * qq + pp * qq - pp * r * qq_d) / (qq * qq);

			if (std::abs(g) < TOL_ITERATIVE)
				break;

			// newton step
			r = r - g / g_p;
		}
		x1->col(i) = x0.col(i) / r0 * r;		
	}

}

