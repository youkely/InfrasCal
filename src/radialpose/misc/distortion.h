#pragma once
#include <Eigen/Dense>
#include <vector>

namespace radialpose {

	/* Computes x1 such that x1 = x0/(1+lambda*sum(x0.^2)) */
	void forward_1param_division_model(double lambda, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x0, Eigen::Matrix<double, 2, Eigen::Dynamic>* x1);
	/* Computes x1 such that x1/(1+lambda*sum(x1.^2)) = x0 */
	void inverse_1param_division_model(double lambda, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x0, Eigen::Matrix<double, 2, Eigen::Dynamic>* x1);


	/*
	 Rational distortion functions
	  f(r) = (1+ mu_1 * r2 + ... + mu_np * r^(2*np)) / (1+ lambda_1 * r2 + ... + lambda_nd * r^(2*nd))
	 distortion parmeters are dist_params = [mu; lambda]
	*/


	/* Computes x1 such that x1 = f(|x0|) * x0  */
	void forward_rational_model(const std::vector<double> &params, int np, int nd, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x0, Eigen::Matrix<double, 2, Eigen::Dynamic>* x1);
	/* Computes x1 such that x0 = f(|x1|) * x1.
       Since there is no analytical solution, this is done by iterative method */
	void inverse_rational_model(const std::vector<double> &params, int np, int nd, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x0, Eigen::Matrix<double, 2, Eigen::Dynamic>* x1);


};