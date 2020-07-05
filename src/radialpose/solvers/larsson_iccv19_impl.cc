#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include "larsson_iccv19.h"
#include "../misc/univariate.h"
#include "../misc/distortion.h"

using namespace Eigen;
using namespace radialpose;
using std::complex;


inline void precompute_distortion(Matrix<double, 2, Dynamic> x, Matrix<double, 3, Dynamic> X, Array<double, 1, Dynamic> *r2, Array<double, 1, Dynamic> *z, Array<double, 1, Dynamic> *alpha) {
	*r2 = X.topRows(2).colwise().squaredNorm();
	*r2 /= r2->abs().mean();
	*z = X.row(2);
	*alpha = x.colwise().squaredNorm().array();
	*alpha /= x.cwiseProduct(X.topRows(2)).colwise().sum().array();
}
inline void precompute_undistortion(Matrix<double, 2, Dynamic> x, Matrix<double, 3, Dynamic> X, Array<double, 1, Dynamic> *r2, Array<double, 1, Dynamic> *z, Array<double, 1, Dynamic> *alpha) {
	*r2 = x.colwise().squaredNorm();
	*r2 /= r2->abs().mean();
	*z = X.row(2);
	*alpha = x.colwise().squaredNorm().array();
	*alpha /= x.cwiseProduct(X.topRows(2)).colwise().sum().array();
}

template<>
int radialpose::larsson_iccv19::Solver<1, 0, false>::solver_impl(const Points2D& x, const Points3D& X, std::vector<double>* t3) const
{
	Array<double, 1, Dynamic> r2;
	Array<double, 1, Dynamic> z;
	Array<double, 1, Dynamic> alpha;
	precompute_undistortion(x, X, &r2, &z, &alpha);

	double cc_1_1 = alpha(1) * r2(1) - alpha(0) * r2(0);
	double cc_1_2 = alpha(1) - alpha(0);
	double cc_1_3 = alpha(1) * r2(1) * z(1) - alpha(0) * r2(0) * z(0);
	double cc_1_4 = alpha(1) * z(1) - alpha(0) * z(0);
	double cc_2_1 = alpha(2) * r2(2) - alpha(0) * r2(0);
	double cc_2_2 = alpha(2) - alpha(0);
	double cc_2_3 = alpha(2) * r2(2) * z(2) - alpha(0) * r2(0) * z(0);
	double cc_2_4 = alpha(2) * z(2) - alpha(0) * z(0);
	Eigen::Matrix<double, 2, 2> AA;
	AA << -cc_1_1, -cc_1_2, -cc_2_1, -cc_2_2;
	Eigen::Matrix<double, 2, 2> A0;
	A0 << cc_1_3, cc_1_4, cc_2_3, cc_2_4;

	if (use_rescaling) {
		// Some preconditioning
		double s0 = A0.col(0).norm();
		A0.col(0) /= s0;
		AA.col(0) /= s0;
		double s1 = A0.col(1).norm();
		A0.col(1) /= s1;
		AA.col(1) /= s1;
	}

	if (use_qz_solver) {
		// Solve as generalized eigenvalue problem
		
		Eigen::GeneralizedEigenSolver<Eigen::Matrix<double, 2, 2>> es(A0, AA, false);
		es.setMaxIterations(1000);

		if (es.info() != Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 2, 1> alphas = es.alphas();
		Eigen::Matrix<double, 2, 1> betas = es.betas();
		for (int k = 0; k < 2; k++) {
			if (std::fabs(alphas(k).imag()) < 1e-8 && std::fabs(betas(k)) > 1e-10)
				t3->push_back(alphas(k).real() / betas(k));
		}
	} else {
		// This is a 2x2 eigenvalue problem
		AA = A0.partialPivLu().solve(AA);

		double b = -AA(0,0)-AA(1,1);
		double c = AA(0, 0)*AA(1, 1) - AA(0, 1)*AA(1, 0);

		std::complex<double> roots[2];
		solve_quadratic(1.0, b, c, roots);
		for (int k = 0; k < 2; k++) {
			if (std::fabs(roots[k].imag()) < 1e-8 && std::fabs(roots[k].real()) > 1e-8)
				t3->push_back(1.0 / roots[k].real());
		}

	}
	return t3->size();
}

/*
int radialpose::larsson_iccv19::Solver<2, 0, false>::solver_impl(Matrix<double, 2, Dynamic> x, Matrix<double, 3, Dynamic> X, std::vector<double> *t3)
{
	Array<double, 1, Dynamic> r2;
	Array<double, 1, Dynamic> z;
	Array<double, 1, Dynamic> alpha;
	precompute_undistortion(x, X, &r2, &z, &alpha);

	double cc_1_1 = alpha(1) * r2(1) - alpha(0) * r2(0);
	double cc_1_2 = alpha(1) * std::pow(r2(1), 2) - alpha(0) * std::pow(r2(0), 2);
	double cc_1_3 = alpha(1) - alpha(0);
	double cc_1_4 = alpha(1) * r2(1) * z(1) - alpha(0) * r2(0) * z(0);
	double cc_1_5 = alpha(1) * std::pow(r2(1), 2)*z(1) - alpha(0) * std::pow(r2(0), 2)*z(0);
	double cc_1_6 = alpha(1) * z(1) - alpha(0) * z(0);
	double cc_2_1 = alpha(2) * r2(2) - alpha(0) * r2(0);
	double cc_2_2 = alpha(2) * std::pow(r2(2), 2) - alpha(0) * std::pow(r2(0), 2);
	double cc_2_3 = alpha(2) - alpha(0);
	double cc_2_4 = alpha(2) * r2(2) * z(2) - alpha(0) * r2(0) * z(0);
	double cc_2_5 = alpha(2) * std::pow(r2(2), 2)*z(2) - alpha(0) * std::pow(r2(0), 2)*z(0);
	double cc_2_6 = alpha(2) * z(2) - alpha(0) * z(0);
	double cc_3_1 = alpha(3) * r2(3) - alpha(0) * r2(0);
	double cc_3_2 = alpha(3) * std::pow(r2(3), 2) - alpha(0) * std::pow(r2(0), 2);
	double cc_3_3 = alpha(3) - alpha(0);
	double cc_3_4 = alpha(3) * r2(3) * z(3) - alpha(0) * r2(0) * z(0);
	double cc_3_5 = alpha(3) * std::pow(r2(3), 2)*z(3) - alpha(0) * std::pow(r2(0), 2)*z(0);
	double cc_3_6 = alpha(3) * z(3) - alpha(0) * z(0);
	Eigen::Matrix<double, 3, 3> AA;
	AA << -cc_1_1, -cc_1_2, -cc_1_3, -cc_2_1, -cc_2_2, -cc_2_3, -cc_3_1, -cc_3_2, -cc_3_3;
	Eigen::Matrix<double, 3, 3> A0;
	A0 << cc_1_4, cc_1_5, cc_1_6, cc_2_4, cc_2_5, cc_2_6, cc_3_4, cc_3_5, cc_3_6;


	if (use_rescaling) {
		// Some preconditioning
		double s0 = A0.col(0).norm();
		A0.col(0) /= s0;
		AA.col(0) /= s0;
		double s1 = A0.col(1).norm();
		A0.col(1) /= s1;
		AA.col(1) /= s1;
		double s2 = A0.col(2).norm();
		A0.col(2) /= s2;
		AA.col(2) /= s2;
	}

	if (use_qz_solver) {
		// Solve as generalized eigenvalue problem
		
		Eigen::GeneralizedEigenSolver<Eigen::Matrix<double, 3, 3>> es(A0, AA, false);
		es.setMaxIterations(1000);

		if (!es.info() == Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 3, 1> alphas = es.alphas();
		Eigen::Matrix<double, 3, 1> betas = es.betas();
		for (int k = 0; k < 3; k++) {
			if (std::fabs(alphas(k).imag()) < 1e-8 && std::fabs(betas(k)) > 1e-10)
				t3->push_back(alphas(k).real() / betas(k));
		}
	} else {
		// This is a 3x3 eigenvalue problem
		AA = A0.partialPivLu().solve(AA);

		Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> es;
		es.compute(AA, false);
		Eigen::Matrix<std::complex<double>, 3, 1> ev = es.eigenvalues();
		for (int k = 0; k < 3; k++) {
			if (std::fabs(ev(k).imag()) < 1e-8 && std::fabs(ev(k).real()) > 1e-8)
				t3->push_back(1.0 / ev(k).real());
		}

	}
	return t3->size();
}


int radialpose::larsson_iccv19::Solver<3, 0, false>::solver_impl(Matrix<double, 2, Dynamic> x, Matrix<double, 3, Dynamic> X, std::vector<double> *t3)
{
	Array<double, 1, Dynamic> r2;
	Array<double, 1, Dynamic> z;
	Array<double, 1, Dynamic> alpha;
	precompute_undistortion(x, X, &r2, &z, &alpha);

	double cc_1_1 = alpha(1) * r2(1) - alpha(0) * r2(0);
	double cc_1_2 = alpha(1) * std::pow(r2(1), 2) - alpha(0) * std::pow(r2(0), 2);
	double cc_1_3 = alpha(1) * std::pow(r2(1), 3) - alpha(0) * std::pow(r2(0), 3);
	double cc_1_4 = alpha(1) - alpha(0);
	double cc_1_5 = alpha(1) * r2(1) * z(1) - alpha(0) * r2(0) * z(0);
	double cc_1_6 = alpha(1) * std::pow(r2(1), 2)*z(1) - alpha(0) * std::pow(r2(0), 2)*z(0);
	double cc_1_7 = alpha(1) * std::pow(r2(1), 3)*z(1) - alpha(0) * std::pow(r2(0), 3)*z(0);
	double cc_1_8 = alpha(1) * z(1) - alpha(0) * z(0);
	double cc_2_1 = alpha(2) * r2(2) - alpha(0) * r2(0);
	double cc_2_2 = alpha(2) * std::pow(r2(2), 2) - alpha(0) * std::pow(r2(0), 2);
	double cc_2_3 = alpha(2) * std::pow(r2(2), 3) - alpha(0) * std::pow(r2(0), 3);
	double cc_2_4 = alpha(2) - alpha(0);
	double cc_2_5 = alpha(2) * r2(2) * z(2) - alpha(0) * r2(0) * z(0);
	double cc_2_6 = alpha(2) * std::pow(r2(2), 2)*z(2) - alpha(0) * std::pow(r2(0), 2)*z(0);
	double cc_2_7 = alpha(2) * std::pow(r2(2), 3)*z(2) - alpha(0) * std::pow(r2(0), 3)*z(0);
	double cc_2_8 = alpha(2) * z(2) - alpha(0) * z(0);
	double cc_3_1 = alpha(3) * r2(3) - alpha(0) * r2(0);
	double cc_3_2 = alpha(3) * std::pow(r2(3), 2) - alpha(0) * std::pow(r2(0), 2);
	double cc_3_3 = alpha(3) * std::pow(r2(3), 3) - alpha(0) * std::pow(r2(0), 3);
	double cc_3_4 = alpha(3) - alpha(0);
	double cc_3_5 = alpha(3) * r2(3) * z(3) - alpha(0) * r2(0) * z(0);
	double cc_3_6 = alpha(3) * std::pow(r2(3), 2)*z(3) - alpha(0) * std::pow(r2(0), 2)*z(0);
	double cc_3_7 = alpha(3) * std::pow(r2(3), 3)*z(3) - alpha(0) * std::pow(r2(0), 3)*z(0);
	double cc_3_8 = alpha(3) * z(3) - alpha(0) * z(0);
	double cc_4_1 = alpha(4) * r2(4) - alpha(0) * r2(0);
	double cc_4_2 = alpha(4) * std::pow(r2(4), 2) - alpha(0) * std::pow(r2(0), 2);
	double cc_4_3 = alpha(4) * std::pow(r2(4), 3) - alpha(0) * std::pow(r2(0), 3);
	double cc_4_4 = alpha(4) - alpha(0);
	double cc_4_5 = alpha(4) * r2(4) * z(4) - alpha(0) * r2(0) * z(0);
	double cc_4_6 = alpha(4) * std::pow(r2(4), 2)*z(4) - alpha(0) * std::pow(r2(0), 2)*z(0);
	double cc_4_7 = alpha(4) * std::pow(r2(4), 3)*z(4) - alpha(0) * std::pow(r2(0), 3)*z(0);
	double cc_4_8 = alpha(4) * z(4) - alpha(0) * z(0);
	Eigen::Matrix<double, 4, 4> AA;
	AA << -cc_1_1, -cc_1_2, -cc_1_3, -cc_1_4, -cc_2_1, -cc_2_2, -cc_2_3, -cc_2_4, -cc_3_1, -cc_3_2, -cc_3_3, -cc_3_4, -cc_4_1, -cc_4_2, -cc_4_3, -cc_4_4;
	Eigen::Matrix<double, 4, 4> A0;
	A0 << cc_1_5, cc_1_6, cc_1_7, cc_1_8, cc_2_5, cc_2_6, cc_2_7, cc_2_8, cc_3_5, cc_3_6, cc_3_7, cc_3_8, cc_4_5, cc_4_6, cc_4_7, cc_4_8;


	if (use_rescaling) {
		// Some preconditioning
		double s0 = A0.col(0).norm();
		A0.col(0) /= s0;
		AA.col(0) /= s0;
		double s1 = A0.col(1).norm();
		A0.col(1) /= s1;
		AA.col(1) /= s1;
		double s2 = A0.col(2).norm();
		A0.col(2) /= s2;
		AA.col(2) /= s2;
		double s3 = A0.col(3).norm();
		A0.col(3) /= s3;
		AA.col(3) /= s3;
	}

	if (use_qz_solver) {
		// Solve as generalized eigenvalue problem

		Eigen::GeneralizedEigenSolver<Eigen::Matrix<double, 4, 4>> es(A0, AA, false);
		es.setMaxIterations(1000);

		if (!es.info() == Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 4, 1> alphas = es.alphas();
		Eigen::Matrix<double, 4, 1> betas = es.betas();
		for (int k = 0; k < 4; k++) {
			if (std::fabs(alphas(k).imag()) < 1e-8 && std::fabs(betas(k)) > 1e-10)
				t3->push_back(alphas(k).real() / betas(k));
		}
	} else {
		// This is a 4x4 eigenvalue problem
		AA = A0.partialPivLu().solve(AA);

		Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> es;
		es.compute(AA, false);
		Eigen::Matrix<std::complex<double>, 4, 1> ev = es.eigenvalues();
		for (int k = 0; k < 4; k++) {
			if (std::fabs(ev(k).imag()) < 1e-8 && std::fabs(ev(k).real()) > 1e-8)
				t3->push_back(1.0 / ev(k).real());
		}

	}
	return t3->size();
}


int radialpose::larsson_iccv19::Solver<3, 3, false>::solver_impl(Matrix<double, 2, Dynamic> x, Matrix<double, 3, Dynamic> X, std::vector<double>* t3)
{
	Array<double, 1, Dynamic> r2;
	Array<double, 1, Dynamic> z;
	Array<double, 1, Dynamic> alpha;

	precompute_distortion(x, X, &r2, &z, &alpha);

	double cc_1_1 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[2] * r2[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[3])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[3] * r2[3])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[4]) - (std::pow(r2[0], 3) - std::pow(r2[4], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[2] * r2[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[4])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[4] * r2[4])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_1_2 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2) - alpha[2] * std::pow(r2[2], 2))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 2) - alpha[3] * std::pow(r2[3], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[4]) - (std::pow(r2[0], 3) - std::pow(r2[4], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2) - alpha[2] * std::pow(r2[2], 2))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[4]) - (alpha[0] * std::pow(r2[0], 2) - alpha[4] * std::pow(r2[4], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_1_3 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3) - alpha[2] * std::pow(r2[2], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 3) - alpha[3] * std::pow(r2[3], 3))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[4]) - (std::pow(r2[0], 3) - std::pow(r2[4], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3) - alpha[2] * std::pow(r2[2], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[4]) - (alpha[0] * std::pow(r2[0], 3) - alpha[4] * std::pow(r2[4], 3))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_1_4 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[4]) - (alpha[0] - alpha[4])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[2]) - (alpha[0] - alpha[2])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[3]) - (alpha[0] - alpha[3])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[2]) - (alpha[0] - alpha[2])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[4]) - (std::pow(r2[0], 3) - std::pow(r2[4], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_1_5 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[2] * r2[2] * z[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[3])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[3] * r2[3] * z[3])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[4]) - (std::pow(r2[0], 3) - std::pow(r2[4], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[2] * r2[2] * z[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[4])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[4] * r2[4] * z[4])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_1_6 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[2] * std::pow(r2[2], 2)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[3] * std::pow(r2[3], 2)*z[3])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[4]) - (std::pow(r2[0], 3) - std::pow(r2[4], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[2] * std::pow(r2[2], 2)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[4]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[4] * std::pow(r2[4], 2)*z[4])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_1_7 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[2] * std::pow(r2[2], 3)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[3] * std::pow(r2[3], 3)*z[3])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[4]) - (std::pow(r2[0], 3) - std::pow(r2[4], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[2] * std::pow(r2[2], 3)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[4]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[4] * std::pow(r2[4], 3)*z[4])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_1_8 = (((r2[0] - r2[2])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[2] * z[2]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1])) - ((r2[0] - r2[3])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[3] * z[3]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[4]) - (std::pow(r2[0], 3) - std::pow(r2[4], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((r2[0] - r2[2])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[2] * z[2]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[4]) - (std::pow(r2[0], 2) - std::pow(r2[4], 2))*(r2[0] - r2[1])) - ((r2[0] - r2[4])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[4] * z[4]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_2_1 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[2] * r2[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[3])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[3] * r2[3])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[5]) - (std::pow(r2[0], 3) - std::pow(r2[5], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[2] * r2[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[5])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[5] * r2[5])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_2_2 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2) - alpha[2] * std::pow(r2[2], 2))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 2) - alpha[3] * std::pow(r2[3], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[5]) - (std::pow(r2[0], 3) - std::pow(r2[5], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2) - alpha[2] * std::pow(r2[2], 2))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[5]) - (alpha[0] * std::pow(r2[0], 2) - alpha[5] * std::pow(r2[5], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_2_3 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3) - alpha[2] * std::pow(r2[2], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 3) - alpha[3] * std::pow(r2[3], 3))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[5]) - (std::pow(r2[0], 3) - std::pow(r2[5], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3) - alpha[2] * std::pow(r2[2], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[5]) - (alpha[0] * std::pow(r2[0], 3) - alpha[5] * std::pow(r2[5], 3))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_2_4 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[5]) - (alpha[0] - alpha[5])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[2]) - (alpha[0] - alpha[2])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[3]) - (alpha[0] - alpha[3])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[2]) - (alpha[0] - alpha[2])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[5]) - (std::pow(r2[0], 3) - std::pow(r2[5], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1])));
	double cc_2_5 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[2] * r2[2] * z[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[3])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[3] * r2[3] * z[3])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[5]) - (std::pow(r2[0], 3) - std::pow(r2[5], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[2] * r2[2] * z[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[5])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[5] * r2[5] * z[5])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_2_6 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[2] * std::pow(r2[2], 2)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[3] * std::pow(r2[3], 2)*z[3])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[5]) - (std::pow(r2[0], 3) - std::pow(r2[5], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[2] * std::pow(r2[2], 2)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[5]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[5] * std::pow(r2[5], 2)*z[5])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_2_7 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[2] * std::pow(r2[2], 3)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[3] * std::pow(r2[3], 3)*z[3])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[5]) - (std::pow(r2[0], 3) - std::pow(r2[5], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[2] * std::pow(r2[2], 3)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[5]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[5] * std::pow(r2[5], 3)*z[5])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_2_8 = (((r2[0] - r2[2])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[2] * z[2]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1])) - ((r2[0] - r2[3])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[3] * z[3]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[5]) - (std::pow(r2[0], 3) - std::pow(r2[5], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1]))) - (((r2[0] - r2[2])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[2] * z[2]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[5]) - (std::pow(r2[0], 2) - std::pow(r2[5], 2))*(r2[0] - r2[1])) - ((r2[0] - r2[5])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[5] * z[5]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_3_1 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[2] * r2[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[3])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[3] * r2[3])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[6]) - (std::pow(r2[0], 3) - std::pow(r2[6], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[2] * r2[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[6])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[6] * r2[6])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_3_2 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2) - alpha[2] * std::pow(r2[2], 2))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 2) - alpha[3] * std::pow(r2[3], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[6]) - (std::pow(r2[0], 3) - std::pow(r2[6], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2) - alpha[2] * std::pow(r2[2], 2))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[6]) - (alpha[0] * std::pow(r2[0], 2) - alpha[6] * std::pow(r2[6], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_3_3 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3) - alpha[2] * std::pow(r2[2], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 3) - alpha[3] * std::pow(r2[3], 3))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[6]) - (std::pow(r2[0], 3) - std::pow(r2[6], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3) - alpha[2] * std::pow(r2[2], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[6]) - (alpha[0] * std::pow(r2[0], 3) - alpha[6] * std::pow(r2[6], 3))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_3_4 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[6]) - (alpha[0] - alpha[6])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[2]) - (alpha[0] - alpha[2])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[3]) - (alpha[0] - alpha[3])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[2]) - (alpha[0] - alpha[2])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[6]) - (std::pow(r2[0], 3) - std::pow(r2[6], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1])));
	double cc_3_5 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[2] * r2[2] * z[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[3])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[3] * r2[3] * z[3])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[6]) - (std::pow(r2[0], 3) - std::pow(r2[6], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[2] * r2[2] * z[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[6])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[6] * r2[6] * z[6])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_3_6 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[2] * std::pow(r2[2], 2)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[3] * std::pow(r2[3], 2)*z[3])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[6]) - (std::pow(r2[0], 3) - std::pow(r2[6], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[2] * std::pow(r2[2], 2)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[6]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[6] * std::pow(r2[6], 2)*z[6])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_3_7 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[2] * std::pow(r2[2], 3)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[3] * std::pow(r2[3], 3)*z[3])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[6]) - (std::pow(r2[0], 3) - std::pow(r2[6], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[2] * std::pow(r2[2], 3)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[6]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[6] * std::pow(r2[6], 3)*z[6])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_3_8 = (((r2[0] - r2[2])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[2] * z[2]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1])) - ((r2[0] - r2[3])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[3] * z[3]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[6]) - (std::pow(r2[0], 3) - std::pow(r2[6], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1]))) - (((r2[0] - r2[2])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[2] * z[2]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[6]) - (std::pow(r2[0], 2) - std::pow(r2[6], 2))*(r2[0] - r2[1])) - ((r2[0] - r2[6])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[6] * z[6]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_4_1 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[2] * r2[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[3])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[3] * r2[3])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[7]) - (std::pow(r2[0], 3) - std::pow(r2[7], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[2] * r2[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[7])*(alpha[0] * r2[0] - alpha[1] * r2[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] - alpha[7] * r2[7])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_4_2 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2) - alpha[2] * std::pow(r2[2], 2))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 2) - alpha[3] * std::pow(r2[3], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[7]) - (std::pow(r2[0], 3) - std::pow(r2[7], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2) - alpha[2] * std::pow(r2[2], 2))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2) - alpha[1] * std::pow(r2[1], 2))*(r2[0] - r2[7]) - (alpha[0] * std::pow(r2[0], 2) - alpha[7] * std::pow(r2[7], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_4_3 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3) - alpha[2] * std::pow(r2[2], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 3) - alpha[3] * std::pow(r2[3], 3))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[7]) - (std::pow(r2[0], 3) - std::pow(r2[7], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3) - alpha[2] * std::pow(r2[2], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3) - alpha[1] * std::pow(r2[1], 3))*(r2[0] - r2[7]) - (alpha[0] * std::pow(r2[0], 3) - alpha[7] * std::pow(r2[7], 3))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_4_4 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[7]) - (alpha[0] - alpha[7])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[2]) - (alpha[0] - alpha[2])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[3]) - (alpha[0] - alpha[3])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] - alpha[1])*(r2[0] - r2[2]) - (alpha[0] - alpha[2])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[7]) - (std::pow(r2[0], 3) - std::pow(r2[7], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1])));
	double cc_4_5 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[2] * r2[2] * z[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[3])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[3] * r2[3] * z[3])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[7]) - (std::pow(r2[0], 3) - std::pow(r2[7], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))*((r2[0] - r2[2])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[2] * r2[2] * z[2])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((r2[0] - r2[7])*(alpha[0] * r2[0] * z[0] - alpha[1] * r2[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * r2[0] * z[0] - alpha[7] * r2[7] * z[7])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_4_6 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[2] * std::pow(r2[2], 2)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[3] * std::pow(r2[3], 2)*z[3])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[7]) - (std::pow(r2[0], 3) - std::pow(r2[7], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[2] * std::pow(r2[2], 2)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[1] * std::pow(r2[1], 2)*z[1])*(r2[0] - r2[7]) - (alpha[0] * std::pow(r2[0], 2)*z[0] - alpha[7] * std::pow(r2[7], 2)*z[7])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_4_7 = (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[2] * std::pow(r2[2], 3)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[3]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[3] * std::pow(r2[3], 3)*z[3])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[7]) - (std::pow(r2[0], 3) - std::pow(r2[7], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))) - (((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[2]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[2] * std::pow(r2[2], 3)*z[2])*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[1] * std::pow(r2[1], 3)*z[1])*(r2[0] - r2[7]) - (alpha[0] * std::pow(r2[0], 3)*z[0] - alpha[7] * std::pow(r2[7], 3)*z[7])*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));
	double cc_4_8 = (((r2[0] - r2[2])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[2] * z[2]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1])) - ((r2[0] - r2[3])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[3] * z[3]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[7]) - (std::pow(r2[0], 3) - std::pow(r2[7], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1]))) - (((r2[0] - r2[2])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[2] * z[2]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[7]) - (std::pow(r2[0], 2) - std::pow(r2[7], 2))*(r2[0] - r2[1])) - ((r2[0] - r2[7])*(alpha[0] * z[0] - alpha[1] * z[1]) - (r2[0] - r2[1])*(alpha[0] * z[0] - alpha[7] * z[7]))*((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1])))*(((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[2]) - (std::pow(r2[0], 2) - std::pow(r2[2], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[3]) - (std::pow(r2[0], 3) - std::pow(r2[3], 3))*(r2[0] - r2[1])) - ((std::pow(r2[0], 2) - std::pow(r2[1], 2))*(r2[0] - r2[3]) - (std::pow(r2[0], 2) - std::pow(r2[3], 2))*(r2[0] - r2[1]))*((std::pow(r2[0], 3) - std::pow(r2[1], 3))*(r2[0] - r2[2]) - (std::pow(r2[0], 3) - std::pow(r2[2], 3))*(r2[0] - r2[1])));

	Eigen::Matrix<double, 4, 4> AA;
	AA << -cc_1_1, -cc_1_2, -cc_1_3, -cc_1_4, -cc_2_1, -cc_2_2, -cc_2_3, -cc_2_4, -cc_3_1, -cc_3_2, -cc_3_3, -cc_3_4, -cc_4_1, -cc_4_2, -cc_4_3, -cc_4_4;
	Eigen::Matrix<double, 4, 4> A0;
	A0 << cc_1_5, cc_1_6, cc_1_7, cc_1_8, cc_2_5, cc_2_6, cc_2_7, cc_2_8, cc_3_5, cc_3_6, cc_3_7, cc_3_8, cc_4_5, cc_4_6, cc_4_7, cc_4_8;


	if (use_rescaling) {
		// Some preconditioning
		double s0 = A0.col(0).norm();
		A0.col(0) /= s0;
		AA.col(0) /= s0;
		double s1 = A0.col(1).norm();
		A0.col(1) /= s1;
		AA.col(1) /= s1;
		double s2 = A0.col(2).norm();
		A0.col(2) /= s2;
		AA.col(2) /= s2;
		double s3 = A0.col(3).norm();
		A0.col(3) /= s3;
		AA.col(3) /= s3;
	}

	if (use_qz_solver) {
		// Solve as generalized eigenvalue problem

		Eigen::GeneralizedEigenSolver<Eigen::Matrix<double, 4, 4>> es(A0, AA, false);
		es.setMaxIterations(1000);

		if (!es.info() == Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 4, 1> alphas = es.alphas();
		Eigen::Matrix<double, 4, 1> betas = es.betas();
		for (int k = 0; k < 4; k++) {
			if (std::fabs(alphas(k).imag()) < 1e-8 && std::fabs(betas(k)) > 1e-10)
				t3->push_back(alphas(k).real() / betas(k));
		}
	} else {
		// This is a 4x4 eigenvalue problem
		AA = A0.partialPivLu().solve(AA);

		Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> es;
		es.compute(AA, false);
		Eigen::Matrix<std::complex<double>, 4, 1> ev = es.eigenvalues();
		for (int k = 0; k < 4; k++) {
			if (std::fabs(ev(k).imag()) < 1e-8 && std::fabs(ev(k).real()) > 1e-8)
				t3->push_back(1.0 / ev(k).real());
		}

	}


	return t3->size();
}
*/

template<>
int radialpose::larsson_iccv19::Solver<1, 0, true>::solver_impl(const Points2D& x, const Points3D& X, std::vector<double>* t3) const
{
	// Assumes that X = R*X0+diag([1,1,0])*t

	Array<double, 1, Dynamic> r2;
	Array<double, 1, Dynamic> z;
	Array<double, 1, Dynamic> alpha;

	precompute_distortion(x, X, &r2, &z, &alpha);

	double cc_1_1 = r2(1) - r2(0);
	double cc_1_2 = 2 * r2(1) * z(0) - 2 * r2(0) * z(1);
	double cc_1_3 = r2(1) * std::pow(z(0), 2) - r2(0) * std::pow(z(1), 2);
	double cc_1_4 = alpha(1) * r2(0) - alpha(0) * r2(1);
	double cc_1_5 = 3 * alpha(1) * r2(0) * z(1) - 3 * alpha(0) * r2(1) * z(0);
	double cc_1_6 = 3 * alpha(1) * r2(0) * std::pow(z(1), 2) - 3 * alpha(0) * r2(1) * std::pow(z(0), 2);
	double cc_1_7 = alpha(1) * r2(0) * std::pow(z(1), 3) - alpha(0) * r2(1) * std::pow(z(0), 3);
	double cc_2_1 = r2(2) - r2(0);
	double cc_2_2 = 2 * r2(2) * z(0) - 2 * r2(0) * z(2);
	double cc_2_3 = r2(2) * std::pow(z(0), 2) - r2(0) * std::pow(z(2), 2);
	double cc_2_4 = alpha(2) * r2(0) - alpha(0) * r2(2);
	double cc_2_5 = 3 * alpha(2) * r2(0) * z(2) - 3 * alpha(0) * r2(2) * z(0);
	double cc_2_6 = 3 * alpha(2) * r2(0) * std::pow(z(2), 2) - 3 * alpha(0) * r2(2) * std::pow(z(0), 2);
	double cc_2_7 = alpha(2) * r2(0) * std::pow(z(2), 3) - alpha(0) * r2(2) * std::pow(z(0), 3);
	Eigen::Matrix<double, 2, 5> AA;
	AA << -cc_1_2, -cc_1_6, -cc_1_1, -cc_1_5, -cc_1_4, -cc_2_2, -cc_2_6, -cc_2_1, -cc_2_5, -cc_2_4;
	Eigen::Matrix<double, 2, 2> A0;
	A0 << cc_1_3, cc_1_7, cc_2_3, cc_2_7;
	Eigen::Matrix<double, 5, 5> B;
	B.setZero();
	B(2, 0) = 1;
	B(3, 1) = 1;
	B(4, 3) = 1;

	if (use_rescaling) {
		// Some preconditioning
		double s0 = A0.col(0).norm();
		A0.col(0) /= s0;
		AA.col(0) /= s0; AA.col(2) /= s0;
		double s1 = A0.col(1).norm();
		A0.col(1) /= s1;
		AA.col(1) /= s1; AA.col(3) /= s1; AA.col(4) /= s1;
	}

	if (use_qz_solver) {
		// Solve as generalized eigenvalue problem
		B.block<2, 5>(0, 0) = AA;

		Eigen::Matrix<double, 5, 5> A;
		A.setIdentity();
		A.block<2, 2>(0, 0) = A0;

		Eigen::GeneralizedEigenSolver<Eigen::Matrix<double, 5, 5>> es(A, B, false);
		es.setMaxIterations(1000);

		if (es.info() != Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 5, 1> alphas = es.alphas();
		Eigen::Matrix<double, 5, 1> betas = es.betas();
		for (int k = 0; k < 5; k++) {
			if (std::fabs(alphas(k).imag()) < 1e-8 && std::fabs(betas(k)) > 1e-10)
				t3->push_back(alphas(k).real() / betas(k));
		}
	} else {
		// Convert to eigenvalue problem and solve using normal eigensolver
		AA = A0.partialPivLu().solve(AA);
		B.block<2, 5>(0, 0) = AA;

		Eigen::EigenSolver<Eigen::Matrix<double, 5, 5>> es;
		es.compute(B, false);
		Eigen::Matrix<std::complex<double>, 5, 1> ev = es.eigenvalues();
		for (int k = 0; k < 5; k++) {
			if (std::fabs(ev(k).imag()) < 1e-8 && std::fabs(ev(k).real()) > 1e-8)
				t3->push_back(1.0 / ev(k).real());
		}

	}

	return t3->size();
}

template<>
int radialpose::larsson_iccv19::Solver<2, 0, true>::solver_impl(const Points2D& x, const Points3D& X, std::vector<double>* t3) const
{
	Array<double, 1, Dynamic> r2;
	Array<double, 1, Dynamic> z;
	Array<double, 1, Dynamic> alpha;

	precompute_distortion(x, X, &r2, &z, &alpha);

	double cc_1_1 = std::pow(r2(1), 2) - std::pow(r2(0), 2);
	double cc_1_2 = 4 * std::pow(r2(1), 2) * z(0) - 4 * std::pow(r2(0), 2) * z(1);
	double cc_1_3 = 6 * std::pow(r2(1), 2) * std::pow(z(0), 2) - 6 * std::pow(r2(0), 2) * std::pow(z(1), 2);
	double cc_1_4 = 4 * std::pow(r2(1), 2) * std::pow(z(0), 3) - 4 * std::pow(r2(0), 2) * std::pow(z(1), 3);
	double cc_1_5 = std::pow(r2(1), 2) * std::pow(z(0), 4) - std::pow(r2(0), 2) * std::pow(z(1), 4);
	double cc_1_6 = alpha(1) * std::pow(r2(0), 2) - alpha(0) * std::pow(r2(1), 2);
	double cc_1_7 = 5 * alpha(1) * std::pow(r2(0), 2) * z(1) - 5 * alpha(0) * std::pow(r2(1), 2) * z(0);
	double cc_1_8 = 10 * alpha(1) * std::pow(r2(0), 2) * std::pow(z(1), 2) - 10 * alpha(0) * std::pow(r2(1), 2) * std::pow(z(0), 2);
	double cc_1_9 = r2(0) * std::pow(r2(1), 2) - std::pow(r2(0), 2) * r2(1);
	double cc_1_10 = 10 * alpha(1) * std::pow(r2(0), 2) * std::pow(z(1), 3) - 10 * alpha(0) * std::pow(r2(1), 2) * std::pow(z(0), 3);
	double cc_1_11 = 2 * r2(0) * std::pow(r2(1), 2) * z(0) - 2 * std::pow(r2(0), 2) * r2(1) * z(1);
	double cc_1_12 = 5 * alpha(1) * std::pow(r2(0), 2) * std::pow(z(1), 4) - 5 * alpha(0) * std::pow(r2(1), 2) * std::pow(z(0), 4);
	double cc_1_13 = r2(0) * std::pow(r2(1), 2) * std::pow(z(0), 2) - std::pow(r2(0), 2) * r2(1) * std::pow(z(1), 2);
	double cc_1_14 = alpha(1) * std::pow(r2(0), 2) * std::pow(z(1), 5) - alpha(0) * std::pow(r2(1), 2) * std::pow(z(0), 5);
	double cc_2_1 = std::pow(r2(2), 2) - std::pow(r2(0), 2);
	double cc_2_2 = 4 * std::pow(r2(2), 2) * z(0) - 4 * std::pow(r2(0), 2) * z(2);
	double cc_2_3 = 6 * std::pow(r2(2), 2) * std::pow(z(0), 2) - 6 * std::pow(r2(0), 2) * std::pow(z(2), 2);
	double cc_2_4 = 4 * std::pow(r2(2), 2) * std::pow(z(0), 3) - 4 * std::pow(r2(0), 2) * std::pow(z(2), 3);
	double cc_2_5 = std::pow(r2(2), 2) * std::pow(z(0), 4) - std::pow(r2(0), 2) * std::pow(z(2), 4);
	double cc_2_6 = alpha(2) * std::pow(r2(0), 2) - alpha(0) * std::pow(r2(2), 2);
	double cc_2_7 = 5 * alpha(2) * std::pow(r2(0), 2) * z(2) - 5 * alpha(0) * std::pow(r2(2), 2) * z(0);
	double cc_2_8 = 10 * alpha(2) * std::pow(r2(0), 2) * std::pow(z(2), 2) - 10 * alpha(0) * std::pow(r2(2), 2) * std::pow(z(0), 2);
	double cc_2_9 = r2(0) * std::pow(r2(2), 2) - std::pow(r2(0), 2) * r2(2);
	double cc_2_10 = 10 * alpha(2) * std::pow(r2(0), 2) * std::pow(z(2), 3) - 10 * alpha(0) * std::pow(r2(2), 2) * std::pow(z(0), 3);
	double cc_2_11 = 2 * r2(0) * std::pow(r2(2), 2) * z(0) - 2 * std::pow(r2(0), 2) * r2(2) * z(2);
	double cc_2_12 = 5 * alpha(2) * std::pow(r2(0), 2) * std::pow(z(2), 4) - 5 * alpha(0) * std::pow(r2(2), 2) * std::pow(z(0), 4);
	double cc_2_13 = r2(0) * std::pow(r2(2), 2) * std::pow(z(0), 2) - std::pow(r2(0), 2) * r2(2) * std::pow(z(2), 2);
	double cc_2_14 = alpha(2) * std::pow(r2(0), 2) * std::pow(z(2), 5) - alpha(0) * std::pow(r2(2), 2) * std::pow(z(0), 5);
	double cc_3_1 = std::pow(r2(3), 2) - std::pow(r2(0), 2);
	double cc_3_2 = 4 * std::pow(r2(3), 2) * z(0) - 4 * std::pow(r2(0), 2) * z(3);
	double cc_3_3 = 6 * std::pow(r2(3), 2) * std::pow(z(0), 2) - 6 * std::pow(r2(0), 2) * std::pow(z(3), 2);
	double cc_3_4 = 4 * std::pow(r2(3), 2) * std::pow(z(0), 3) - 4 * std::pow(r2(0), 2) * std::pow(z(3), 3);
	double cc_3_5 = std::pow(r2(3), 2) * std::pow(z(0), 4) - std::pow(r2(0), 2) * std::pow(z(3), 4);
	double cc_3_6 = alpha(3) * std::pow(r2(0), 2) - alpha(0) * std::pow(r2(3), 2);
	double cc_3_7 = 5 * alpha(3) * std::pow(r2(0), 2) * z(3) - 5 * alpha(0) * std::pow(r2(3), 2) * z(0);
	double cc_3_8 = 10 * alpha(3) * std::pow(r2(0), 2) * std::pow(z(3), 2) - 10 * alpha(0) * std::pow(r2(3), 2) * std::pow(z(0), 2);
	double cc_3_9 = r2(0) * std::pow(r2(3), 2) - std::pow(r2(0), 2) * r2(3);
	double cc_3_10 = 10 * alpha(3) * std::pow(r2(0), 2) * std::pow(z(3), 3) - 10 * alpha(0) * std::pow(r2(3), 2) * std::pow(z(0), 3);
	double cc_3_11 = 2 * r2(0) * std::pow(r2(3), 2) * z(0) - 2 * std::pow(r2(0), 2) * r2(3) * z(3);
	double cc_3_12 = 5 * alpha(3) * std::pow(r2(0), 2) * std::pow(z(3), 4) - 5 * alpha(0) * std::pow(r2(3), 2) * std::pow(z(0), 4);
	double cc_3_13 = r2(0) * std::pow(r2(3), 2) * std::pow(z(0), 2) - std::pow(r2(0), 2) * r2(3) * std::pow(z(3), 2);
	double cc_3_14 = alpha(3) * std::pow(r2(0), 2) * std::pow(z(3), 5) - alpha(0) * std::pow(r2(3), 2) * std::pow(z(0), 5);

	Eigen::Matrix<double, 3, 11> AA;
	AA << -cc_1_4, -cc_1_11, -cc_1_12, -cc_1_3, -cc_1_9, -cc_1_10, -cc_1_2, -cc_1_8, -cc_1_1, -cc_1_7, -cc_1_6, -cc_2_4, -cc_2_11, -cc_2_12, -cc_2_3, -cc_2_9, -cc_2_10, -cc_2_2, -cc_2_8, -cc_2_1, -cc_2_7, -cc_2_6, -cc_3_4, -cc_3_11, -cc_3_12, -cc_3_3, -cc_3_9, -cc_3_10, -cc_3_2, -cc_3_8, -cc_3_1, -cc_3_7, -cc_3_6;
	Eigen::Matrix<double, 3, 3> A0;
	A0 << cc_1_5, cc_1_13, cc_1_14, cc_2_5, cc_2_13, cc_2_14, cc_3_5, cc_3_13, cc_3_14;
	Eigen::Matrix<double, 11, 11> B;
	B.setZero();
	B(3, 0) = 1; B(4, 1) = 1; B(5, 2) = 1;
	B(6, 3) = 1; B(7, 5) = 1; B(8, 6) = 1;
	B(9, 7) = 1; B(10, 9) = 1;

	if (use_rescaling) {
		// Some preconditioning
		double s0 = A0.col(0).norm();
		A0.col(0) /= s0;
		AA.col(0) /= s0; AA.col(3) /= s0; AA.col(6) /= s0; AA.col(8) /= s0;
		double s1 = A0.col(1).norm();
		A0.col(1) /= s1;
		AA.col(1) /= s1; AA.col(4) /= s1;
		double s2 = A0.col(2).norm();
		A0.col(2) /= s2;
		AA.col(2) /= s2; AA.col(5) /= s2; AA.col(7) /= s2; AA.col(9) /= s2; AA.col(10) /= s2;
	}

	if (use_qz_solver) {
		// Solve as generalized eigenvalue problem
		B.block<3, 11>(0, 0) = AA;

		Eigen::Matrix<double, 11, 11> A;
		A.setIdentity();
		A.block<3, 3>(0, 0) = A0;

		Eigen::GeneralizedEigenSolver<Eigen::Matrix<double, 11, 11>> es(A, B, false);
		es.setMaxIterations(1000);

		if (es.info() != Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 11, 1> alphas = es.alphas();
		Eigen::Matrix<double, 11, 1> betas = es.betas();
		for (int k = 0; k < 19; k++) {
			if (std::fabs(alphas(k).imag()) < 1e-8 && std::fabs(betas(k)) > 1e-10)
				t3->push_back(alphas(k).real() / betas(k));
		}

	} else {
		AA = A0.partialPivLu().solve(AA);
		B.block<3, 11>(0, 0) = AA;

		Eigen::EigenSolver<Eigen::Matrix<double, 11, 11>> es;
		es.compute(B, false);
		Eigen::Matrix<std::complex<double>, 11, 1> ev = es.eigenvalues();
		for (int k = 0; k < 11; k++) {
			if (std::fabs(ev(k).imag()) < 1e-8 && std::fabs(ev(k).real()) > 1e-8)
				t3->push_back(1.0 / ev(k).real());
		}
	}


	return t3->size();
}


template<>
int radialpose::larsson_iccv19::Solver<3, 0, true>::solver_impl(const Points2D& x, const Points3D& X, std::vector<double>* t3) const
{
	Array<double, 1, Dynamic> r2;
	Array<double, 1, Dynamic> z;
	Array<double, 1, Dynamic> alpha;

	precompute_distortion(x, X, &r2, &z, &alpha);

	double cc_1_1 = std::pow(r2(1), 3) - std::pow(r2(0), 3);
	double cc_1_2 = 6 * std::pow(r2(1), 3) * z(0) - 6 * std::pow(r2(0), 3) * z(1);
	double cc_1_3 = 15 * std::pow(r2(1), 3) * std::pow(z(0), 2) - 15 * std::pow(r2(0), 3) * std::pow(z(1), 2);
	double cc_1_4 = 20 * std::pow(r2(1), 3) * std::pow(z(0), 3) - 20 * std::pow(r2(0), 3) * std::pow(z(1), 3);
	double cc_1_5 = 15 * std::pow(r2(1), 3) * std::pow(z(0), 4) - 15 * std::pow(r2(0), 3) * std::pow(z(1), 4);
	double cc_1_6 = 6 * std::pow(r2(1), 3) * std::pow(z(0), 5) - 6 * std::pow(r2(0), 3) * std::pow(z(1), 5);
	double cc_1_7 = std::pow(r2(1), 3) * std::pow(z(0), 6) - std::pow(r2(0), 3) * std::pow(z(1), 6);
	double cc_1_8 = alpha(1) * std::pow(r2(0), 3) - alpha(0) * std::pow(r2(1), 3);
	double cc_1_9 = 7 * alpha(1) * std::pow(r2(0), 3) * z(1) - 7 * alpha(0) * std::pow(r2(1), 3) * z(0);
	double cc_1_10 = 21 * alpha(1) * std::pow(r2(0), 3) * std::pow(z(1), 2) - 21 * alpha(0) * std::pow(r2(1), 3) * std::pow(z(0), 2);
	double cc_1_11 = r2(0) * std::pow(r2(1), 3) - std::pow(r2(0), 3) * r2(1);
	double cc_1_12 = 35 * alpha(1) * std::pow(r2(0), 3) * std::pow(z(1), 3) - 35 * alpha(0) * std::pow(r2(1), 3) * std::pow(z(0), 3);
	double cc_1_13 = 4 * r2(0) * std::pow(r2(1), 3) * z(0) - 4 * std::pow(r2(0), 3) * r2(1) * z(1);
	double cc_1_14 = 35 * alpha(1) * std::pow(r2(0), 3) * std::pow(z(1), 4) - 35 * alpha(0) * std::pow(r2(1), 3) * std::pow(z(0), 4);
	double cc_1_15 = 6 * r2(0) * std::pow(r2(1), 3) * std::pow(z(0), 2) - 6 * std::pow(r2(0), 3) * r2(1) * std::pow(z(1), 2);
	double cc_1_16 = std::pow(r2(0), 2) * std::pow(r2(1), 3) - std::pow(r2(0), 3) * std::pow(r2(1), 2);
	double cc_1_17 = 21 * alpha(1) * std::pow(r2(0), 3) * std::pow(z(1), 5) - 21 * alpha(0) * std::pow(r2(1), 3) * std::pow(z(0), 5);
	double cc_1_18 = 4 * r2(0) * std::pow(r2(1), 3) * std::pow(z(0), 3) - 4 * std::pow(r2(0), 3) * r2(1) * std::pow(z(1), 3);
	double cc_1_19 = 2 * std::pow(r2(0), 2) * std::pow(r2(1), 3) * z(0) - 2 * std::pow(r2(0), 3) * std::pow(r2(1), 2) * z(1);
	double cc_1_20 = 7 * alpha(1) * std::pow(r2(0), 3) * std::pow(z(1), 6) - 7 * alpha(0) * std::pow(r2(1), 3) * std::pow(z(0), 6);
	double cc_1_21 = r2(0) * std::pow(r2(1), 3) * std::pow(z(0), 4) - std::pow(r2(0), 3) * r2(1) * std::pow(z(1), 4);
	double cc_1_22 = std::pow(r2(0), 2) * std::pow(r2(1), 3) * std::pow(z(0), 2) - std::pow(r2(0), 3) * std::pow(r2(1), 2) * std::pow(z(1), 2);
	double cc_1_23 = alpha(1) * std::pow(r2(0), 3) * std::pow(z(1), 7) - alpha(0) * std::pow(r2(1), 3) * std::pow(z(0), 7);
	double cc_2_1 = std::pow(r2(2), 3) - std::pow(r2(0), 3);
	double cc_2_2 = 6 * std::pow(r2(2), 3) * z(0) - 6 * std::pow(r2(0), 3) * z(2);
	double cc_2_3 = 15 * std::pow(r2(2), 3) * std::pow(z(0), 2) - 15 * std::pow(r2(0), 3) * std::pow(z(2), 2);
	double cc_2_4 = 20 * std::pow(r2(2), 3) * std::pow(z(0), 3) - 20 * std::pow(r2(0), 3) * std::pow(z(2), 3);
	double cc_2_5 = 15 * std::pow(r2(2), 3) * std::pow(z(0), 4) - 15 * std::pow(r2(0), 3) * std::pow(z(2), 4);
	double cc_2_6 = 6 * std::pow(r2(2), 3) * std::pow(z(0), 5) - 6 * std::pow(r2(0), 3) * std::pow(z(2), 5);
	double cc_2_7 = std::pow(r2(2), 3) * std::pow(z(0), 6) - std::pow(r2(0), 3) * std::pow(z(2), 6);
	double cc_2_8 = alpha(2) * std::pow(r2(0), 3) - alpha(0) * std::pow(r2(2), 3);
	double cc_2_9 = 7 * alpha(2) * std::pow(r2(0), 3) * z(2) - 7 * alpha(0) * std::pow(r2(2), 3) * z(0);
	double cc_2_10 = 21 * alpha(2) * std::pow(r2(0), 3) * std::pow(z(2), 2) - 21 * alpha(0) * std::pow(r2(2), 3) * std::pow(z(0), 2);
	double cc_2_11 = r2(0) * std::pow(r2(2), 3) - std::pow(r2(0), 3) * r2(2);
	double cc_2_12 = 35 * alpha(2) * std::pow(r2(0), 3) * std::pow(z(2), 3) - 35 * alpha(0) * std::pow(r2(2), 3) * std::pow(z(0), 3);
	double cc_2_13 = 4 * r2(0) * std::pow(r2(2), 3) * z(0) - 4 * std::pow(r2(0), 3) * r2(2) * z(2);
	double cc_2_14 = 35 * alpha(2) * std::pow(r2(0), 3) * std::pow(z(2), 4) - 35 * alpha(0) * std::pow(r2(2), 3) * std::pow(z(0), 4);
	double cc_2_15 = 6 * r2(0) * std::pow(r2(2), 3) * std::pow(z(0), 2) - 6 * std::pow(r2(0), 3) * r2(2) * std::pow(z(2), 2);
	double cc_2_16 = std::pow(r2(0), 2) * std::pow(r2(2), 3) - std::pow(r2(0), 3) * std::pow(r2(2), 2);
	double cc_2_17 = 21 * alpha(2) * std::pow(r2(0), 3) * std::pow(z(2), 5) - 21 * alpha(0) * std::pow(r2(2), 3) * std::pow(z(0), 5);
	double cc_2_18 = 4 * r2(0) * std::pow(r2(2), 3) * std::pow(z(0), 3) - 4 * std::pow(r2(0), 3) * r2(2) * std::pow(z(2), 3);
	double cc_2_19 = 2 * std::pow(r2(0), 2) * std::pow(r2(2), 3) * z(0) - 2 * std::pow(r2(0), 3) * std::pow(r2(2), 2) * z(2);
	double cc_2_20 = 7 * alpha(2) * std::pow(r2(0), 3) * std::pow(z(2), 6) - 7 * alpha(0) * std::pow(r2(2), 3) * std::pow(z(0), 6);
	double cc_2_21 = r2(0) * std::pow(r2(2), 3) * std::pow(z(0), 4) - std::pow(r2(0), 3) * r2(2) * std::pow(z(2), 4);
	double cc_2_22 = std::pow(r2(0), 2) * std::pow(r2(2), 3) * std::pow(z(0), 2) - std::pow(r2(0), 3) * std::pow(r2(2), 2) * std::pow(z(2), 2);
	double cc_2_23 = alpha(2) * std::pow(r2(0), 3) * std::pow(z(2), 7) - alpha(0) * std::pow(r2(2), 3) * std::pow(z(0), 7);
	double cc_3_1 = std::pow(r2(3), 3) - std::pow(r2(0), 3);
	double cc_3_2 = 6 * std::pow(r2(3), 3) * z(0) - 6 * std::pow(r2(0), 3) * z(3);
	double cc_3_3 = 15 * std::pow(r2(3), 3) * std::pow(z(0), 2) - 15 * std::pow(r2(0), 3) * std::pow(z(3), 2);
	double cc_3_4 = 20 * std::pow(r2(3), 3) * std::pow(z(0), 3) - 20 * std::pow(r2(0), 3) * std::pow(z(3), 3);
	double cc_3_5 = 15 * std::pow(r2(3), 3) * std::pow(z(0), 4) - 15 * std::pow(r2(0), 3) * std::pow(z(3), 4);
	double cc_3_6 = 6 * std::pow(r2(3), 3) * std::pow(z(0), 5) - 6 * std::pow(r2(0), 3) * std::pow(z(3), 5);
	double cc_3_7 = std::pow(r2(3), 3) * std::pow(z(0), 6) - std::pow(r2(0), 3) * std::pow(z(3), 6);
	double cc_3_8 = alpha(3) * std::pow(r2(0), 3) - alpha(0) * std::pow(r2(3), 3);
	double cc_3_9 = 7 * alpha(3) * std::pow(r2(0), 3) * z(3) - 7 * alpha(0) * std::pow(r2(3), 3) * z(0);
	double cc_3_10 = 21 * alpha(3) * std::pow(r2(0), 3) * std::pow(z(3), 2) - 21 * alpha(0) * std::pow(r2(3), 3) * std::pow(z(0), 2);
	double cc_3_11 = r2(0) * std::pow(r2(3), 3) - std::pow(r2(0), 3) * r2(3);
	double cc_3_12 = 35 * alpha(3) * std::pow(r2(0), 3) * std::pow(z(3), 3) - 35 * alpha(0) * std::pow(r2(3), 3) * std::pow(z(0), 3);
	double cc_3_13 = 4 * r2(0) * std::pow(r2(3), 3) * z(0) - 4 * std::pow(r2(0), 3) * r2(3) * z(3);
	double cc_3_14 = 35 * alpha(3) * std::pow(r2(0), 3) * std::pow(z(3), 4) - 35 * alpha(0) * std::pow(r2(3), 3) * std::pow(z(0), 4);
	double cc_3_15 = 6 * r2(0) * std::pow(r2(3), 3) * std::pow(z(0), 2) - 6 * std::pow(r2(0), 3) * r2(3) * std::pow(z(3), 2);
	double cc_3_16 = std::pow(r2(0), 2) * std::pow(r2(3), 3) - std::pow(r2(0), 3) * std::pow(r2(3), 2);
	double cc_3_17 = 21 * alpha(3) * std::pow(r2(0), 3) * std::pow(z(3), 5) - 21 * alpha(0) * std::pow(r2(3), 3) * std::pow(z(0), 5);
	double cc_3_18 = 4 * r2(0) * std::pow(r2(3), 3) * std::pow(z(0), 3) - 4 * std::pow(r2(0), 3) * r2(3) * std::pow(z(3), 3);
	double cc_3_19 = 2 * std::pow(r2(0), 2) * std::pow(r2(3), 3) * z(0) - 2 * std::pow(r2(0), 3) * std::pow(r2(3), 2) * z(3);
	double cc_3_20 = 7 * alpha(3) * std::pow(r2(0), 3) * std::pow(z(3), 6) - 7 * alpha(0) * std::pow(r2(3), 3) * std::pow(z(0), 6);
	double cc_3_21 = r2(0) * std::pow(r2(3), 3) * std::pow(z(0), 4) - std::pow(r2(0), 3) * r2(3) * std::pow(z(3), 4);
	double cc_3_22 = std::pow(r2(0), 2) * std::pow(r2(3), 3) * std::pow(z(0), 2) - std::pow(r2(0), 3) * std::pow(r2(3), 2) * std::pow(z(3), 2);
	double cc_3_23 = alpha(3) * std::pow(r2(0), 3) * std::pow(z(3), 7) - alpha(0) * std::pow(r2(3), 3) * std::pow(z(0), 7);
	double cc_4_1 = std::pow(r2(4), 3) - std::pow(r2(0), 3);
	double cc_4_2 = 6 * std::pow(r2(4), 3) * z(0) - 6 * std::pow(r2(0), 3) * z(4);
	double cc_4_3 = 15 * std::pow(r2(4), 3) * std::pow(z(0), 2) - 15 * std::pow(r2(0), 3) * std::pow(z(4), 2);
	double cc_4_4 = 20 * std::pow(r2(4), 3) * std::pow(z(0), 3) - 20 * std::pow(r2(0), 3) * std::pow(z(4), 3);
	double cc_4_5 = 15 * std::pow(r2(4), 3) * std::pow(z(0), 4) - 15 * std::pow(r2(0), 3) * std::pow(z(4), 4);
	double cc_4_6 = 6 * std::pow(r2(4), 3) * std::pow(z(0), 5) - 6 * std::pow(r2(0), 3) * std::pow(z(4), 5);
	double cc_4_7 = std::pow(r2(4), 3) * std::pow(z(0), 6) - std::pow(r2(0), 3) * std::pow(z(4), 6);
	double cc_4_8 = alpha(4) * std::pow(r2(0), 3) - alpha(0) * std::pow(r2(4), 3);
	double cc_4_9 = 7 * alpha(4) * std::pow(r2(0), 3) * z(4) - 7 * alpha(0) * std::pow(r2(4), 3) * z(0);
	double cc_4_10 = 21 * alpha(4) * std::pow(r2(0), 3) * std::pow(z(4), 2) - 21 * alpha(0) * std::pow(r2(4), 3) * std::pow(z(0), 2);
	double cc_4_11 = r2(0) * std::pow(r2(4), 3) - std::pow(r2(0), 3) * r2(4);
	double cc_4_12 = 35 * alpha(4) * std::pow(r2(0), 3) * std::pow(z(4), 3) - 35 * alpha(0) * std::pow(r2(4), 3) * std::pow(z(0), 3);
	double cc_4_13 = 4 * r2(0) * std::pow(r2(4), 3) * z(0) - 4 * std::pow(r2(0), 3) * r2(4) * z(4);
	double cc_4_14 = 35 * alpha(4) * std::pow(r2(0), 3) * std::pow(z(4), 4) - 35 * alpha(0) * std::pow(r2(4), 3) * std::pow(z(0), 4);
	double cc_4_15 = 6 * r2(0) * std::pow(r2(4), 3) * std::pow(z(0), 2) - 6 * std::pow(r2(0), 3) * r2(4) * std::pow(z(4), 2);
	double cc_4_16 = std::pow(r2(0), 2) * std::pow(r2(4), 3) - std::pow(r2(0), 3) * std::pow(r2(4), 2);
	double cc_4_17 = 21 * alpha(4) * std::pow(r2(0), 3) * std::pow(z(4), 5) - 21 * alpha(0) * std::pow(r2(4), 3) * std::pow(z(0), 5);
	double cc_4_18 = 4 * r2(0) * std::pow(r2(4), 3) * std::pow(z(0), 3) - 4 * std::pow(r2(0), 3) * r2(4) * std::pow(z(4), 3);
	double cc_4_19 = 2 * std::pow(r2(0), 2) * std::pow(r2(4), 3) * z(0) - 2 * std::pow(r2(0), 3) * std::pow(r2(4), 2) * z(4);
	double cc_4_20 = 7 * alpha(4) * std::pow(r2(0), 3) * std::pow(z(4), 6) - 7 * alpha(0) * std::pow(r2(4), 3) * std::pow(z(0), 6);
	double cc_4_21 = r2(0) * std::pow(r2(4), 3) * std::pow(z(0), 4) - std::pow(r2(0), 3) * r2(4) * std::pow(z(4), 4);
	double cc_4_22 = std::pow(r2(0), 2) * std::pow(r2(4), 3) * std::pow(z(0), 2) - std::pow(r2(0), 3) * std::pow(r2(4), 2) * std::pow(z(4), 2);
	double cc_4_23 = alpha(4) * std::pow(r2(0), 3) * std::pow(z(4), 7) - alpha(0) * std::pow(r2(4), 3) * std::pow(z(0), 7);

	Eigen::Matrix<double, 4, 19> AA;
	AA << -cc_1_6, -cc_1_18, -cc_1_19, -cc_1_20, -cc_1_5, -cc_1_15, -cc_1_16, -cc_1_17, -cc_1_4, -cc_1_13, -cc_1_14, -cc_1_3, -cc_1_11, -cc_1_12, -cc_1_2, -cc_1_10, -cc_1_1, -cc_1_9, -cc_1_8, -cc_2_6, -cc_2_18, -cc_2_19, -cc_2_20, -cc_2_5, -cc_2_15, -cc_2_16, -cc_2_17, -cc_2_4, -cc_2_13, -cc_2_14, -cc_2_3, -cc_2_11, -cc_2_12, -cc_2_2, -cc_2_10, -cc_2_1, -cc_2_9, -cc_2_8, -cc_3_6, -cc_3_18, -cc_3_19, -cc_3_20, -cc_3_5, -cc_3_15, -cc_3_16, -cc_3_17, -cc_3_4, -cc_3_13, -cc_3_14, -cc_3_3, -cc_3_11, -cc_3_12, -cc_3_2, -cc_3_10, -cc_3_1, -cc_3_9, -cc_3_8, -cc_4_6, -cc_4_18, -cc_4_19, -cc_4_20, -cc_4_5, -cc_4_15, -cc_4_16, -cc_4_17, -cc_4_4, -cc_4_13, -cc_4_14, -cc_4_3, -cc_4_11, -cc_4_12, -cc_4_2, -cc_4_10, -cc_4_1, -cc_4_9, -cc_4_8;
	Eigen::Matrix<double, 4, 4> A0;
	A0 << cc_1_7, cc_1_21, cc_1_22, cc_1_23, cc_2_7, cc_2_21, cc_2_22, cc_2_23, cc_3_7, cc_3_21, cc_3_22, cc_3_23, cc_4_7, cc_4_21, cc_4_22, cc_4_23;
	Eigen::Matrix<double, 19, 19> B;
	B.setZero();
	B(4, 0) = 1; B(5, 1) = 1; B(6, 2) = 1;
	B(7, 3) = 1; B(8, 4) = 1; B(9, 5) = 1;
	B(10, 7) = 1; B(11, 8) = 1;	B(12, 9) = 1;
	B(13, 10) = 1; B(14, 11) = 1; B(15, 13) = 1;
	B(16, 14) = 1; B(17, 15) = 1; B(18, 17) = 1;

	if (use_rescaling) {
		// Some preconditioning
		double s0 = A0.col(0).array().abs().mean();
		A0.col(0) /= s0;
		AA.col(0) /= s0; AA.col(4) /= s0; AA.col(8) /= s0; AA.col(11) /= s0; AA.col(14) /= s0; AA.col(16) /= s0;
		double s1 = A0.col(1).array().abs().mean();
		A0.col(1) /= s1;
		AA.col(1) /= s1; AA.col(5) /= s1; AA.col(9) /= s1; AA.col(12) /= s1;
		double s2 = A0.col(2).array().abs().mean();
		A0.col(2) /= s2;
		AA.col(2) /= s2; AA.col(6) /= s2;
		double s3 = A0.col(3).array().abs().mean();
		A0.col(3) /= s3;
		AA.col(3) /= s3; AA.col(7) /= s3; AA.col(10) /= s3; AA.col(13) /= s3; AA.col(15) /= s3; AA.col(17) /= s3; AA.col(18) /= s3;
	}

	if (use_qz_solver) {
		// Solve as generalized eigenvalue problem
		B.block<4, 19>(0, 0) = AA;

		Eigen::Matrix<double, 19, 19> A;
		A.setIdentity();
		A.block<4, 4>(0, 0) = A0;

		Eigen::GeneralizedEigenSolver<Eigen::Matrix<double, 19, 19>> es(A, B, false);
		es.setMaxIterations(1000);

		if (es.info() != Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 19, 1> alphas = es.alphas();
		Eigen::Matrix<double, 19, 1> betas = es.betas();
		for (int k = 0; k < 19; k++) {
			if (std::fabs(alphas(k).imag()) < 1e-8 && std::fabs(betas(k)) > 1e-10)
				t3->push_back(alphas(k).real() / betas(k));
		}
	} else {
		// Convert to eigenvalue problem and solve using normal eigensolver
		AA = A0.partialPivLu().solve(AA);
		B.block<4, 19>(0, 0) = AA;

		Eigen::EigenSolver<Eigen::Matrix<double, 19, 19>> es(B);

		if (es.info() != Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 19, 1> ev = es.eigenvalues();
		for (int k = 0; k < 19; k++) {
			if (std::fabs(ev(k).imag()) < 1e-8 && std::fabs(ev(k).real()) > 1e-8)
				t3->push_back(1.0 / ev(k).real());
		}
	}


	return t3->size();
}


template<>
int radialpose::larsson_iccv19::Solver<3, 3, true>::solver_impl(const Points2D& x, const Points3D& X, std::vector<double>* t3) const
{
	Array<double, 1, Dynamic> r2;
	Array<double, 1, Dynamic> z;
	Array<double, 1, Dynamic> alpha;

	precompute_distortion(x, X, &r2, &z, &alpha);

	double cc_1_1 = std::pow(r2[1], 3) - std::pow(r2[0], 3);
	double cc_1_2 = 6 * std::pow(r2[1], 3)*z[0] - 6 * std::pow(r2[0], 3)*z[1];
	double cc_1_3 = 15 * std::pow(r2[1], 3)*std::pow(z[0], 2) - 15 * std::pow(r2[0], 3)*std::pow(z[1], 2);
	double cc_1_4 = 20 * std::pow(r2[1], 3)*std::pow(z[0], 3) - 20 * std::pow(r2[0], 3)*std::pow(z[1], 3);
	double cc_1_5 = 15 * std::pow(r2[1], 3)*std::pow(z[0], 4) - 15 * std::pow(r2[0], 3)*std::pow(z[1], 4);
	double cc_1_6 = 6 * std::pow(r2[1], 3)*std::pow(z[0], 5) - 6 * std::pow(r2[0], 3)*std::pow(z[1], 5);
	double cc_1_7 = std::pow(r2[1], 3)*std::pow(z[0], 6) - std::pow(r2[0], 3)*std::pow(z[1], 6);
	double cc_1_8 = alpha[1] * std::pow(r2[0], 3) - alpha[0] * std::pow(r2[1], 3);
	double cc_1_9 = 7 * alpha[1] * std::pow(r2[0], 3)*z[1] - 7 * alpha[0] * std::pow(r2[1], 3)*z[0];
	double cc_1_10 = alpha[1] * std::pow(r2[0], 3)*r2[1] - alpha[0] * r2[0] * std::pow(r2[1], 3);
	double cc_1_11 = 21 * alpha[1] * std::pow(r2[0], 3)*std::pow(z[1], 2) - 21 * alpha[0] * std::pow(r2[1], 3)*std::pow(z[0], 2);
	double cc_1_12 = r2[0] * std::pow(r2[1], 3) - std::pow(r2[0], 3)*r2[1];
	double cc_1_13 = 5 * alpha[1] * std::pow(r2[0], 3)*r2[1] * z[1] - 5 * alpha[0] * r2[0] * std::pow(r2[1], 3)*z[0];
	double cc_1_14 = 35 * alpha[1] * std::pow(r2[0], 3)*std::pow(z[1], 3) - 35 * alpha[0] * std::pow(r2[1], 3)*std::pow(z[0], 3);
	double cc_1_15 = 4 * r2[0] * std::pow(r2[1], 3)*z[0] - 4 * std::pow(r2[0], 3)*r2[1] * z[1];
	double cc_1_16 = 10 * alpha[1] * std::pow(r2[0], 3)*r2[1] * std::pow(z[1], 2) - 10 * alpha[0] * r2[0] * std::pow(r2[1], 3)*std::pow(z[0], 2);
	double cc_1_17 = alpha[1] * std::pow(r2[0], 3)*std::pow(r2[1], 2) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[1], 3);
	double cc_1_18 = 35 * alpha[1] * std::pow(r2[0], 3)*std::pow(z[1], 4) - 35 * alpha[0] * std::pow(r2[1], 3)*std::pow(z[0], 4);
	double cc_1_19 = 6 * r2[0] * std::pow(r2[1], 3)*std::pow(z[0], 2) - 6 * std::pow(r2[0], 3)*r2[1] * std::pow(z[1], 2);
	double cc_1_20 = std::pow(r2[0], 2)*std::pow(r2[1], 3) - std::pow(r2[0], 3)*std::pow(r2[1], 2);
	double cc_1_21 = 10 * alpha[1] * std::pow(r2[0], 3)*r2[1] * std::pow(z[1], 3) - 10 * alpha[0] * r2[0] * std::pow(r2[1], 3)*std::pow(z[0], 3);
	double cc_1_22 = 3 * alpha[1] * std::pow(r2[0], 3)*std::pow(r2[1], 2)*z[1] - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[1], 3)*z[0];
	double cc_1_23 = 21 * alpha[1] * std::pow(r2[0], 3)*std::pow(z[1], 5) - 21 * alpha[0] * std::pow(r2[1], 3)*std::pow(z[0], 5);
	double cc_1_24 = 4 * r2[0] * std::pow(r2[1], 3)*std::pow(z[0], 3) - 4 * std::pow(r2[0], 3)*r2[1] * std::pow(z[1], 3);
	double cc_1_25 = 2 * std::pow(r2[0], 2)*std::pow(r2[1], 3)*z[0] - 2 * std::pow(r2[0], 3)*std::pow(r2[1], 2)*z[1];
	double cc_1_26 = 5 * alpha[1] * std::pow(r2[0], 3)*r2[1] * std::pow(z[1], 4) - 5 * alpha[0] * r2[0] * std::pow(r2[1], 3)*std::pow(z[0], 4);
	double cc_1_27 = 3 * alpha[1] * std::pow(r2[0], 3)*std::pow(r2[1], 2)*std::pow(z[1], 2) - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[1], 3)*std::pow(z[0], 2);
	double cc_1_28 = alpha[1] * std::pow(r2[0], 3)*std::pow(r2[1], 3) - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[1], 3);
	double cc_1_29 = 7 * alpha[1] * std::pow(r2[0], 3)*std::pow(z[1], 6) - 7 * alpha[0] * std::pow(r2[1], 3)*std::pow(z[0], 6);
	double cc_1_30 = r2[0] * std::pow(r2[1], 3)*std::pow(z[0], 4) - std::pow(r2[0], 3)*r2[1] * std::pow(z[1], 4);
	double cc_1_31 = std::pow(r2[0], 2)*std::pow(r2[1], 3)*std::pow(z[0], 2) - std::pow(r2[0], 3)*std::pow(r2[1], 2)*std::pow(z[1], 2);
	double cc_1_32 = alpha[1] * std::pow(r2[0], 3)*r2[1] * std::pow(z[1], 5) - alpha[0] * r2[0] * std::pow(r2[1], 3)*std::pow(z[0], 5);
	double cc_1_33 = alpha[1] * std::pow(r2[0], 3)*std::pow(r2[1], 2)*std::pow(z[1], 3) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[1], 3)*std::pow(z[0], 3);
	double cc_1_34 = alpha[1] * std::pow(r2[0], 3)*std::pow(r2[1], 3)*z[1] - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[1], 3)*z[0];
	double cc_1_35 = alpha[1] * std::pow(r2[0], 3)*std::pow(z[1], 7) - alpha[0] * std::pow(r2[1], 3)*std::pow(z[0], 7);
	double cc_2_1 = std::pow(r2[2], 3) - std::pow(r2[0], 3);
	double cc_2_2 = 6 * std::pow(r2[2], 3)*z[0] - 6 * std::pow(r2[0], 3)*z[2];
	double cc_2_3 = 15 * std::pow(r2[2], 3)*std::pow(z[0], 2) - 15 * std::pow(r2[0], 3)*std::pow(z[2], 2);
	double cc_2_4 = 20 * std::pow(r2[2], 3)*std::pow(z[0], 3) - 20 * std::pow(r2[0], 3)*std::pow(z[2], 3);
	double cc_2_5 = 15 * std::pow(r2[2], 3)*std::pow(z[0], 4) - 15 * std::pow(r2[0], 3)*std::pow(z[2], 4);
	double cc_2_6 = 6 * std::pow(r2[2], 3)*std::pow(z[0], 5) - 6 * std::pow(r2[0], 3)*std::pow(z[2], 5);
	double cc_2_7 = std::pow(r2[2], 3)*std::pow(z[0], 6) - std::pow(r2[0], 3)*std::pow(z[2], 6);
	double cc_2_8 = alpha[2] * std::pow(r2[0], 3) - alpha[0] * std::pow(r2[2], 3);
	double cc_2_9 = 7 * alpha[2] * std::pow(r2[0], 3)*z[2] - 7 * alpha[0] * std::pow(r2[2], 3)*z[0];
	double cc_2_10 = alpha[2] * std::pow(r2[0], 3)*r2[2] - alpha[0] * r2[0] * std::pow(r2[2], 3);
	double cc_2_11 = 21 * alpha[2] * std::pow(r2[0], 3)*std::pow(z[2], 2) - 21 * alpha[0] * std::pow(r2[2], 3)*std::pow(z[0], 2);
	double cc_2_12 = r2[0] * std::pow(r2[2], 3) - std::pow(r2[0], 3)*r2[2];
	double cc_2_13 = 5 * alpha[2] * std::pow(r2[0], 3)*r2[2] * z[2] - 5 * alpha[0] * r2[0] * std::pow(r2[2], 3)*z[0];
	double cc_2_14 = 35 * alpha[2] * std::pow(r2[0], 3)*std::pow(z[2], 3) - 35 * alpha[0] * std::pow(r2[2], 3)*std::pow(z[0], 3);
	double cc_2_15 = 4 * r2[0] * std::pow(r2[2], 3)*z[0] - 4 * std::pow(r2[0], 3)*r2[2] * z[2];
	double cc_2_16 = 10 * alpha[2] * std::pow(r2[0], 3)*r2[2] * std::pow(z[2], 2) - 10 * alpha[0] * r2[0] * std::pow(r2[2], 3)*std::pow(z[0], 2);
	double cc_2_17 = alpha[2] * std::pow(r2[0], 3)*std::pow(r2[2], 2) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[2], 3);
	double cc_2_18 = 35 * alpha[2] * std::pow(r2[0], 3)*std::pow(z[2], 4) - 35 * alpha[0] * std::pow(r2[2], 3)*std::pow(z[0], 4);
	double cc_2_19 = 6 * r2[0] * std::pow(r2[2], 3)*std::pow(z[0], 2) - 6 * std::pow(r2[0], 3)*r2[2] * std::pow(z[2], 2);
	double cc_2_20 = std::pow(r2[0], 2)*std::pow(r2[2], 3) - std::pow(r2[0], 3)*std::pow(r2[2], 2);
	double cc_2_21 = 10 * alpha[2] * std::pow(r2[0], 3)*r2[2] * std::pow(z[2], 3) - 10 * alpha[0] * r2[0] * std::pow(r2[2], 3)*std::pow(z[0], 3);
	double cc_2_22 = 3 * alpha[2] * std::pow(r2[0], 3)*std::pow(r2[2], 2)*z[2] - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[2], 3)*z[0];
	double cc_2_23 = 21 * alpha[2] * std::pow(r2[0], 3)*std::pow(z[2], 5) - 21 * alpha[0] * std::pow(r2[2], 3)*std::pow(z[0], 5);
	double cc_2_24 = 4 * r2[0] * std::pow(r2[2], 3)*std::pow(z[0], 3) - 4 * std::pow(r2[0], 3)*r2[2] * std::pow(z[2], 3);
	double cc_2_25 = 2 * std::pow(r2[0], 2)*std::pow(r2[2], 3)*z[0] - 2 * std::pow(r2[0], 3)*std::pow(r2[2], 2)*z[2];
	double cc_2_26 = 5 * alpha[2] * std::pow(r2[0], 3)*r2[2] * std::pow(z[2], 4) - 5 * alpha[0] * r2[0] * std::pow(r2[2], 3)*std::pow(z[0], 4);
	double cc_2_27 = 3 * alpha[2] * std::pow(r2[0], 3)*std::pow(r2[2], 2)*std::pow(z[2], 2) - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[2], 3)*std::pow(z[0], 2);
	double cc_2_28 = alpha[2] * std::pow(r2[0], 3)*std::pow(r2[2], 3) - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[2], 3);
	double cc_2_29 = 7 * alpha[2] * std::pow(r2[0], 3)*std::pow(z[2], 6) - 7 * alpha[0] * std::pow(r2[2], 3)*std::pow(z[0], 6);
	double cc_2_30 = r2[0] * std::pow(r2[2], 3)*std::pow(z[0], 4) - std::pow(r2[0], 3)*r2[2] * std::pow(z[2], 4);
	double cc_2_31 = std::pow(r2[0], 2)*std::pow(r2[2], 3)*std::pow(z[0], 2) - std::pow(r2[0], 3)*std::pow(r2[2], 2)*std::pow(z[2], 2);
	double cc_2_32 = alpha[2] * std::pow(r2[0], 3)*r2[2] * std::pow(z[2], 5) - alpha[0] * r2[0] * std::pow(r2[2], 3)*std::pow(z[0], 5);
	double cc_2_33 = alpha[2] * std::pow(r2[0], 3)*std::pow(r2[2], 2)*std::pow(z[2], 3) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[2], 3)*std::pow(z[0], 3);
	double cc_2_34 = alpha[2] * std::pow(r2[0], 3)*std::pow(r2[2], 3)*z[2] - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[2], 3)*z[0];
	double cc_2_35 = alpha[2] * std::pow(r2[0], 3)*std::pow(z[2], 7) - alpha[0] * std::pow(r2[2], 3)*std::pow(z[0], 7);
	double cc_3_1 = std::pow(r2[3], 3) - std::pow(r2[0], 3);
	double cc_3_2 = 6 * std::pow(r2[3], 3)*z[0] - 6 * std::pow(r2[0], 3)*z[3];
	double cc_3_3 = 15 * std::pow(r2[3], 3)*std::pow(z[0], 2) - 15 * std::pow(r2[0], 3)*std::pow(z[3], 2);
	double cc_3_4 = 20 * std::pow(r2[3], 3)*std::pow(z[0], 3) - 20 * std::pow(r2[0], 3)*std::pow(z[3], 3);
	double cc_3_5 = 15 * std::pow(r2[3], 3)*std::pow(z[0], 4) - 15 * std::pow(r2[0], 3)*std::pow(z[3], 4);
	double cc_3_6 = 6 * std::pow(r2[3], 3)*std::pow(z[0], 5) - 6 * std::pow(r2[0], 3)*std::pow(z[3], 5);
	double cc_3_7 = std::pow(r2[3], 3)*std::pow(z[0], 6) - std::pow(r2[0], 3)*std::pow(z[3], 6);
	double cc_3_8 = alpha[3] * std::pow(r2[0], 3) - alpha[0] * std::pow(r2[3], 3);
	double cc_3_9 = 7 * alpha[3] * std::pow(r2[0], 3)*z[3] - 7 * alpha[0] * std::pow(r2[3], 3)*z[0];
	double cc_3_10 = alpha[3] * std::pow(r2[0], 3)*r2[3] - alpha[0] * r2[0] * std::pow(r2[3], 3);
	double cc_3_11 = 21 * alpha[3] * std::pow(r2[0], 3)*std::pow(z[3], 2) - 21 * alpha[0] * std::pow(r2[3], 3)*std::pow(z[0], 2);
	double cc_3_12 = r2[0] * std::pow(r2[3], 3) - std::pow(r2[0], 3)*r2[3];
	double cc_3_13 = 5 * alpha[3] * std::pow(r2[0], 3)*r2[3] * z[3] - 5 * alpha[0] * r2[0] * std::pow(r2[3], 3)*z[0];
	double cc_3_14 = 35 * alpha[3] * std::pow(r2[0], 3)*std::pow(z[3], 3) - 35 * alpha[0] * std::pow(r2[3], 3)*std::pow(z[0], 3);
	double cc_3_15 = 4 * r2[0] * std::pow(r2[3], 3)*z[0] - 4 * std::pow(r2[0], 3)*r2[3] * z[3];
	double cc_3_16 = 10 * alpha[3] * std::pow(r2[0], 3)*r2[3] * std::pow(z[3], 2) - 10 * alpha[0] * r2[0] * std::pow(r2[3], 3)*std::pow(z[0], 2);
	double cc_3_17 = alpha[3] * std::pow(r2[0], 3)*std::pow(r2[3], 2) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[3], 3);
	double cc_3_18 = 35 * alpha[3] * std::pow(r2[0], 3)*std::pow(z[3], 4) - 35 * alpha[0] * std::pow(r2[3], 3)*std::pow(z[0], 4);
	double cc_3_19 = 6 * r2[0] * std::pow(r2[3], 3)*std::pow(z[0], 2) - 6 * std::pow(r2[0], 3)*r2[3] * std::pow(z[3], 2);
	double cc_3_20 = std::pow(r2[0], 2)*std::pow(r2[3], 3) - std::pow(r2[0], 3)*std::pow(r2[3], 2);
	double cc_3_21 = 10 * alpha[3] * std::pow(r2[0], 3)*r2[3] * std::pow(z[3], 3) - 10 * alpha[0] * r2[0] * std::pow(r2[3], 3)*std::pow(z[0], 3);
	double cc_3_22 = 3 * alpha[3] * std::pow(r2[0], 3)*std::pow(r2[3], 2)*z[3] - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[3], 3)*z[0];
	double cc_3_23 = 21 * alpha[3] * std::pow(r2[0], 3)*std::pow(z[3], 5) - 21 * alpha[0] * std::pow(r2[3], 3)*std::pow(z[0], 5);
	double cc_3_24 = 4 * r2[0] * std::pow(r2[3], 3)*std::pow(z[0], 3) - 4 * std::pow(r2[0], 3)*r2[3] * std::pow(z[3], 3);
	double cc_3_25 = 2 * std::pow(r2[0], 2)*std::pow(r2[3], 3)*z[0] - 2 * std::pow(r2[0], 3)*std::pow(r2[3], 2)*z[3];
	double cc_3_26 = 5 * alpha[3] * std::pow(r2[0], 3)*r2[3] * std::pow(z[3], 4) - 5 * alpha[0] * r2[0] * std::pow(r2[3], 3)*std::pow(z[0], 4);
	double cc_3_27 = 3 * alpha[3] * std::pow(r2[0], 3)*std::pow(r2[3], 2)*std::pow(z[3], 2) - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[3], 3)*std::pow(z[0], 2);
	double cc_3_28 = alpha[3] * std::pow(r2[0], 3)*std::pow(r2[3], 3) - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[3], 3);
	double cc_3_29 = 7 * alpha[3] * std::pow(r2[0], 3)*std::pow(z[3], 6) - 7 * alpha[0] * std::pow(r2[3], 3)*std::pow(z[0], 6);
	double cc_3_30 = r2[0] * std::pow(r2[3], 3)*std::pow(z[0], 4) - std::pow(r2[0], 3)*r2[3] * std::pow(z[3], 4);
	double cc_3_31 = std::pow(r2[0], 2)*std::pow(r2[3], 3)*std::pow(z[0], 2) - std::pow(r2[0], 3)*std::pow(r2[3], 2)*std::pow(z[3], 2);
	double cc_3_32 = alpha[3] * std::pow(r2[0], 3)*r2[3] * std::pow(z[3], 5) - alpha[0] * r2[0] * std::pow(r2[3], 3)*std::pow(z[0], 5);
	double cc_3_33 = alpha[3] * std::pow(r2[0], 3)*std::pow(r2[3], 2)*std::pow(z[3], 3) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[3], 3)*std::pow(z[0], 3);
	double cc_3_34 = alpha[3] * std::pow(r2[0], 3)*std::pow(r2[3], 3)*z[3] - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[3], 3)*z[0];
	double cc_3_35 = alpha[3] * std::pow(r2[0], 3)*std::pow(z[3], 7) - alpha[0] * std::pow(r2[3], 3)*std::pow(z[0], 7);
	double cc_4_1 = std::pow(r2[4], 3) - std::pow(r2[0], 3);
	double cc_4_2 = 6 * std::pow(r2[4], 3)*z[0] - 6 * std::pow(r2[0], 3)*z[4];
	double cc_4_3 = 15 * std::pow(r2[4], 3)*std::pow(z[0], 2) - 15 * std::pow(r2[0], 3)*std::pow(z[4], 2);
	double cc_4_4 = 20 * std::pow(r2[4], 3)*std::pow(z[0], 3) - 20 * std::pow(r2[0], 3)*std::pow(z[4], 3);
	double cc_4_5 = 15 * std::pow(r2[4], 3)*std::pow(z[0], 4) - 15 * std::pow(r2[0], 3)*std::pow(z[4], 4);
	double cc_4_6 = 6 * std::pow(r2[4], 3)*std::pow(z[0], 5) - 6 * std::pow(r2[0], 3)*std::pow(z[4], 5);
	double cc_4_7 = std::pow(r2[4], 3)*std::pow(z[0], 6) - std::pow(r2[0], 3)*std::pow(z[4], 6);
	double cc_4_8 = alpha[4] * std::pow(r2[0], 3) - alpha[0] * std::pow(r2[4], 3);
	double cc_4_9 = 7 * alpha[4] * std::pow(r2[0], 3)*z[4] - 7 * alpha[0] * std::pow(r2[4], 3)*z[0];
	double cc_4_10 = alpha[4] * std::pow(r2[0], 3)*r2[4] - alpha[0] * r2[0] * std::pow(r2[4], 3);
	double cc_4_11 = 21 * alpha[4] * std::pow(r2[0], 3)*std::pow(z[4], 2) - 21 * alpha[0] * std::pow(r2[4], 3)*std::pow(z[0], 2);
	double cc_4_12 = r2[0] * std::pow(r2[4], 3) - std::pow(r2[0], 3)*r2[4];
	double cc_4_13 = 5 * alpha[4] * std::pow(r2[0], 3)*r2[4] * z[4] - 5 * alpha[0] * r2[0] * std::pow(r2[4], 3)*z[0];
	double cc_4_14 = 35 * alpha[4] * std::pow(r2[0], 3)*std::pow(z[4], 3) - 35 * alpha[0] * std::pow(r2[4], 3)*std::pow(z[0], 3);
	double cc_4_15 = 4 * r2[0] * std::pow(r2[4], 3)*z[0] - 4 * std::pow(r2[0], 3)*r2[4] * z[4];
	double cc_4_16 = 10 * alpha[4] * std::pow(r2[0], 3)*r2[4] * std::pow(z[4], 2) - 10 * alpha[0] * r2[0] * std::pow(r2[4], 3)*std::pow(z[0], 2);
	double cc_4_17 = alpha[4] * std::pow(r2[0], 3)*std::pow(r2[4], 2) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[4], 3);
	double cc_4_18 = 35 * alpha[4] * std::pow(r2[0], 3)*std::pow(z[4], 4) - 35 * alpha[0] * std::pow(r2[4], 3)*std::pow(z[0], 4);
	double cc_4_19 = 6 * r2[0] * std::pow(r2[4], 3)*std::pow(z[0], 2) - 6 * std::pow(r2[0], 3)*r2[4] * std::pow(z[4], 2);
	double cc_4_20 = std::pow(r2[0], 2)*std::pow(r2[4], 3) - std::pow(r2[0], 3)*std::pow(r2[4], 2);
	double cc_4_21 = 10 * alpha[4] * std::pow(r2[0], 3)*r2[4] * std::pow(z[4], 3) - 10 * alpha[0] * r2[0] * std::pow(r2[4], 3)*std::pow(z[0], 3);
	double cc_4_22 = 3 * alpha[4] * std::pow(r2[0], 3)*std::pow(r2[4], 2)*z[4] - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[4], 3)*z[0];
	double cc_4_23 = 21 * alpha[4] * std::pow(r2[0], 3)*std::pow(z[4], 5) - 21 * alpha[0] * std::pow(r2[4], 3)*std::pow(z[0], 5);
	double cc_4_24 = 4 * r2[0] * std::pow(r2[4], 3)*std::pow(z[0], 3) - 4 * std::pow(r2[0], 3)*r2[4] * std::pow(z[4], 3);
	double cc_4_25 = 2 * std::pow(r2[0], 2)*std::pow(r2[4], 3)*z[0] - 2 * std::pow(r2[0], 3)*std::pow(r2[4], 2)*z[4];
	double cc_4_26 = 5 * alpha[4] * std::pow(r2[0], 3)*r2[4] * std::pow(z[4], 4) - 5 * alpha[0] * r2[0] * std::pow(r2[4], 3)*std::pow(z[0], 4);
	double cc_4_27 = 3 * alpha[4] * std::pow(r2[0], 3)*std::pow(r2[4], 2)*std::pow(z[4], 2) - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[4], 3)*std::pow(z[0], 2);
	double cc_4_28 = alpha[4] * std::pow(r2[0], 3)*std::pow(r2[4], 3) - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[4], 3);
	double cc_4_29 = 7 * alpha[4] * std::pow(r2[0], 3)*std::pow(z[4], 6) - 7 * alpha[0] * std::pow(r2[4], 3)*std::pow(z[0], 6);
	double cc_4_30 = r2[0] * std::pow(r2[4], 3)*std::pow(z[0], 4) - std::pow(r2[0], 3)*r2[4] * std::pow(z[4], 4);
	double cc_4_31 = std::pow(r2[0], 2)*std::pow(r2[4], 3)*std::pow(z[0], 2) - std::pow(r2[0], 3)*std::pow(r2[4], 2)*std::pow(z[4], 2);
	double cc_4_32 = alpha[4] * std::pow(r2[0], 3)*r2[4] * std::pow(z[4], 5) - alpha[0] * r2[0] * std::pow(r2[4], 3)*std::pow(z[0], 5);
	double cc_4_33 = alpha[4] * std::pow(r2[0], 3)*std::pow(r2[4], 2)*std::pow(z[4], 3) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[4], 3)*std::pow(z[0], 3);
	double cc_4_34 = alpha[4] * std::pow(r2[0], 3)*std::pow(r2[4], 3)*z[4] - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[4], 3)*z[0];
	double cc_4_35 = alpha[4] * std::pow(r2[0], 3)*std::pow(z[4], 7) - alpha[0] * std::pow(r2[4], 3)*std::pow(z[0], 7);
	double cc_5_1 = std::pow(r2[5], 3) - std::pow(r2[0], 3);
	double cc_5_2 = 6 * std::pow(r2[5], 3)*z[0] - 6 * std::pow(r2[0], 3)*z[5];
	double cc_5_3 = 15 * std::pow(r2[5], 3)*std::pow(z[0], 2) - 15 * std::pow(r2[0], 3)*std::pow(z[5], 2);
	double cc_5_4 = 20 * std::pow(r2[5], 3)*std::pow(z[0], 3) - 20 * std::pow(r2[0], 3)*std::pow(z[5], 3);
	double cc_5_5 = 15 * std::pow(r2[5], 3)*std::pow(z[0], 4) - 15 * std::pow(r2[0], 3)*std::pow(z[5], 4);
	double cc_5_6 = 6 * std::pow(r2[5], 3)*std::pow(z[0], 5) - 6 * std::pow(r2[0], 3)*std::pow(z[5], 5);
	double cc_5_7 = std::pow(r2[5], 3)*std::pow(z[0], 6) - std::pow(r2[0], 3)*std::pow(z[5], 6);
	double cc_5_8 = alpha[5] * std::pow(r2[0], 3) - alpha[0] * std::pow(r2[5], 3);
	double cc_5_9 = 7 * alpha[5] * std::pow(r2[0], 3)*z[5] - 7 * alpha[0] * std::pow(r2[5], 3)*z[0];
	double cc_5_10 = alpha[5] * std::pow(r2[0], 3)*r2[5] - alpha[0] * r2[0] * std::pow(r2[5], 3);
	double cc_5_11 = 21 * alpha[5] * std::pow(r2[0], 3)*std::pow(z[5], 2) - 21 * alpha[0] * std::pow(r2[5], 3)*std::pow(z[0], 2);
	double cc_5_12 = r2[0] * std::pow(r2[5], 3) - std::pow(r2[0], 3)*r2[5];
	double cc_5_13 = 5 * alpha[5] * std::pow(r2[0], 3)*r2[5] * z[5] - 5 * alpha[0] * r2[0] * std::pow(r2[5], 3)*z[0];
	double cc_5_14 = 35 * alpha[5] * std::pow(r2[0], 3)*std::pow(z[5], 3) - 35 * alpha[0] * std::pow(r2[5], 3)*std::pow(z[0], 3);
	double cc_5_15 = 4 * r2[0] * std::pow(r2[5], 3)*z[0] - 4 * std::pow(r2[0], 3)*r2[5] * z[5];
	double cc_5_16 = 10 * alpha[5] * std::pow(r2[0], 3)*r2[5] * std::pow(z[5], 2) - 10 * alpha[0] * r2[0] * std::pow(r2[5], 3)*std::pow(z[0], 2);
	double cc_5_17 = alpha[5] * std::pow(r2[0], 3)*std::pow(r2[5], 2) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[5], 3);
	double cc_5_18 = 35 * alpha[5] * std::pow(r2[0], 3)*std::pow(z[5], 4) - 35 * alpha[0] * std::pow(r2[5], 3)*std::pow(z[0], 4);
	double cc_5_19 = 6 * r2[0] * std::pow(r2[5], 3)*std::pow(z[0], 2) - 6 * std::pow(r2[0], 3)*r2[5] * std::pow(z[5], 2);
	double cc_5_20 = std::pow(r2[0], 2)*std::pow(r2[5], 3) - std::pow(r2[0], 3)*std::pow(r2[5], 2);
	double cc_5_21 = 10 * alpha[5] * std::pow(r2[0], 3)*r2[5] * std::pow(z[5], 3) - 10 * alpha[0] * r2[0] * std::pow(r2[5], 3)*std::pow(z[0], 3);
	double cc_5_22 = 3 * alpha[5] * std::pow(r2[0], 3)*std::pow(r2[5], 2)*z[5] - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[5], 3)*z[0];
	double cc_5_23 = 21 * alpha[5] * std::pow(r2[0], 3)*std::pow(z[5], 5) - 21 * alpha[0] * std::pow(r2[5], 3)*std::pow(z[0], 5);
	double cc_5_24 = 4 * r2[0] * std::pow(r2[5], 3)*std::pow(z[0], 3) - 4 * std::pow(r2[0], 3)*r2[5] * std::pow(z[5], 3);
	double cc_5_25 = 2 * std::pow(r2[0], 2)*std::pow(r2[5], 3)*z[0] - 2 * std::pow(r2[0], 3)*std::pow(r2[5], 2)*z[5];
	double cc_5_26 = 5 * alpha[5] * std::pow(r2[0], 3)*r2[5] * std::pow(z[5], 4) - 5 * alpha[0] * r2[0] * std::pow(r2[5], 3)*std::pow(z[0], 4);
	double cc_5_27 = 3 * alpha[5] * std::pow(r2[0], 3)*std::pow(r2[5], 2)*std::pow(z[5], 2) - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[5], 3)*std::pow(z[0], 2);
	double cc_5_28 = alpha[5] * std::pow(r2[0], 3)*std::pow(r2[5], 3) - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[5], 3);
	double cc_5_29 = 7 * alpha[5] * std::pow(r2[0], 3)*std::pow(z[5], 6) - 7 * alpha[0] * std::pow(r2[5], 3)*std::pow(z[0], 6);
	double cc_5_30 = r2[0] * std::pow(r2[5], 3)*std::pow(z[0], 4) - std::pow(r2[0], 3)*r2[5] * std::pow(z[5], 4);
	double cc_5_31 = std::pow(r2[0], 2)*std::pow(r2[5], 3)*std::pow(z[0], 2) - std::pow(r2[0], 3)*std::pow(r2[5], 2)*std::pow(z[5], 2);
	double cc_5_32 = alpha[5] * std::pow(r2[0], 3)*r2[5] * std::pow(z[5], 5) - alpha[0] * r2[0] * std::pow(r2[5], 3)*std::pow(z[0], 5);
	double cc_5_33 = alpha[5] * std::pow(r2[0], 3)*std::pow(r2[5], 2)*std::pow(z[5], 3) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[5], 3)*std::pow(z[0], 3);
	double cc_5_34 = alpha[5] * std::pow(r2[0], 3)*std::pow(r2[5], 3)*z[5] - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[5], 3)*z[0];
	double cc_5_35 = alpha[5] * std::pow(r2[0], 3)*std::pow(z[5], 7) - alpha[0] * std::pow(r2[5], 3)*std::pow(z[0], 7);
	double cc_6_1 = std::pow(r2[6], 3) - std::pow(r2[0], 3);
	double cc_6_2 = 6 * std::pow(r2[6], 3)*z[0] - 6 * std::pow(r2[0], 3)*z[6];
	double cc_6_3 = 15 * std::pow(r2[6], 3)*std::pow(z[0], 2) - 15 * std::pow(r2[0], 3)*std::pow(z[6], 2);
	double cc_6_4 = 20 * std::pow(r2[6], 3)*std::pow(z[0], 3) - 20 * std::pow(r2[0], 3)*std::pow(z[6], 3);
	double cc_6_5 = 15 * std::pow(r2[6], 3)*std::pow(z[0], 4) - 15 * std::pow(r2[0], 3)*std::pow(z[6], 4);
	double cc_6_6 = 6 * std::pow(r2[6], 3)*std::pow(z[0], 5) - 6 * std::pow(r2[0], 3)*std::pow(z[6], 5);
	double cc_6_7 = std::pow(r2[6], 3)*std::pow(z[0], 6) - std::pow(r2[0], 3)*std::pow(z[6], 6);
	double cc_6_8 = alpha[6] * std::pow(r2[0], 3) - alpha[0] * std::pow(r2[6], 3);
	double cc_6_9 = 7 * alpha[6] * std::pow(r2[0], 3)*z[6] - 7 * alpha[0] * std::pow(r2[6], 3)*z[0];
	double cc_6_10 = alpha[6] * std::pow(r2[0], 3)*r2[6] - alpha[0] * r2[0] * std::pow(r2[6], 3);
	double cc_6_11 = 21 * alpha[6] * std::pow(r2[0], 3)*std::pow(z[6], 2) - 21 * alpha[0] * std::pow(r2[6], 3)*std::pow(z[0], 2);
	double cc_6_12 = r2[0] * std::pow(r2[6], 3) - std::pow(r2[0], 3)*r2[6];
	double cc_6_13 = 5 * alpha[6] * std::pow(r2[0], 3)*r2[6] * z[6] - 5 * alpha[0] * r2[0] * std::pow(r2[6], 3)*z[0];
	double cc_6_14 = 35 * alpha[6] * std::pow(r2[0], 3)*std::pow(z[6], 3) - 35 * alpha[0] * std::pow(r2[6], 3)*std::pow(z[0], 3);
	double cc_6_15 = 4 * r2[0] * std::pow(r2[6], 3)*z[0] - 4 * std::pow(r2[0], 3)*r2[6] * z[6];
	double cc_6_16 = 10 * alpha[6] * std::pow(r2[0], 3)*r2[6] * std::pow(z[6], 2) - 10 * alpha[0] * r2[0] * std::pow(r2[6], 3)*std::pow(z[0], 2);
	double cc_6_17 = alpha[6] * std::pow(r2[0], 3)*std::pow(r2[6], 2) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[6], 3);
	double cc_6_18 = 35 * alpha[6] * std::pow(r2[0], 3)*std::pow(z[6], 4) - 35 * alpha[0] * std::pow(r2[6], 3)*std::pow(z[0], 4);
	double cc_6_19 = 6 * r2[0] * std::pow(r2[6], 3)*std::pow(z[0], 2) - 6 * std::pow(r2[0], 3)*r2[6] * std::pow(z[6], 2);
	double cc_6_20 = std::pow(r2[0], 2)*std::pow(r2[6], 3) - std::pow(r2[0], 3)*std::pow(r2[6], 2);
	double cc_6_21 = 10 * alpha[6] * std::pow(r2[0], 3)*r2[6] * std::pow(z[6], 3) - 10 * alpha[0] * r2[0] * std::pow(r2[6], 3)*std::pow(z[0], 3);
	double cc_6_22 = 3 * alpha[6] * std::pow(r2[0], 3)*std::pow(r2[6], 2)*z[6] - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[6], 3)*z[0];
	double cc_6_23 = 21 * alpha[6] * std::pow(r2[0], 3)*std::pow(z[6], 5) - 21 * alpha[0] * std::pow(r2[6], 3)*std::pow(z[0], 5);
	double cc_6_24 = 4 * r2[0] * std::pow(r2[6], 3)*std::pow(z[0], 3) - 4 * std::pow(r2[0], 3)*r2[6] * std::pow(z[6], 3);
	double cc_6_25 = 2 * std::pow(r2[0], 2)*std::pow(r2[6], 3)*z[0] - 2 * std::pow(r2[0], 3)*std::pow(r2[6], 2)*z[6];
	double cc_6_26 = 5 * alpha[6] * std::pow(r2[0], 3)*r2[6] * std::pow(z[6], 4) - 5 * alpha[0] * r2[0] * std::pow(r2[6], 3)*std::pow(z[0], 4);
	double cc_6_27 = 3 * alpha[6] * std::pow(r2[0], 3)*std::pow(r2[6], 2)*std::pow(z[6], 2) - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[6], 3)*std::pow(z[0], 2);
	double cc_6_28 = alpha[6] * std::pow(r2[0], 3)*std::pow(r2[6], 3) - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[6], 3);
	double cc_6_29 = 7 * alpha[6] * std::pow(r2[0], 3)*std::pow(z[6], 6) - 7 * alpha[0] * std::pow(r2[6], 3)*std::pow(z[0], 6);
	double cc_6_30 = r2[0] * std::pow(r2[6], 3)*std::pow(z[0], 4) - std::pow(r2[0], 3)*r2[6] * std::pow(z[6], 4);
	double cc_6_31 = std::pow(r2[0], 2)*std::pow(r2[6], 3)*std::pow(z[0], 2) - std::pow(r2[0], 3)*std::pow(r2[6], 2)*std::pow(z[6], 2);
	double cc_6_32 = alpha[6] * std::pow(r2[0], 3)*r2[6] * std::pow(z[6], 5) - alpha[0] * r2[0] * std::pow(r2[6], 3)*std::pow(z[0], 5);
	double cc_6_33 = alpha[6] * std::pow(r2[0], 3)*std::pow(r2[6], 2)*std::pow(z[6], 3) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[6], 3)*std::pow(z[0], 3);
	double cc_6_34 = alpha[6] * std::pow(r2[0], 3)*std::pow(r2[6], 3)*z[6] - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[6], 3)*z[0];
	double cc_6_35 = alpha[6] * std::pow(r2[0], 3)*std::pow(z[6], 7) - alpha[0] * std::pow(r2[6], 3)*std::pow(z[0], 7);
	double cc_7_1 = std::pow(r2[7], 3) - std::pow(r2[0], 3);
	double cc_7_2 = 6 * std::pow(r2[7], 3)*z[0] - 6 * std::pow(r2[0], 3)*z[7];
	double cc_7_3 = 15 * std::pow(r2[7], 3)*std::pow(z[0], 2) - 15 * std::pow(r2[0], 3)*std::pow(z[7], 2);
	double cc_7_4 = 20 * std::pow(r2[7], 3)*std::pow(z[0], 3) - 20 * std::pow(r2[0], 3)*std::pow(z[7], 3);
	double cc_7_5 = 15 * std::pow(r2[7], 3)*std::pow(z[0], 4) - 15 * std::pow(r2[0], 3)*std::pow(z[7], 4);
	double cc_7_6 = 6 * std::pow(r2[7], 3)*std::pow(z[0], 5) - 6 * std::pow(r2[0], 3)*std::pow(z[7], 5);
	double cc_7_7 = std::pow(r2[7], 3)*std::pow(z[0], 6) - std::pow(r2[0], 3)*std::pow(z[7], 6);
	double cc_7_8 = alpha[7] * std::pow(r2[0], 3) - alpha[0] * std::pow(r2[7], 3);
	double cc_7_9 = 7 * alpha[7] * std::pow(r2[0], 3)*z[7] - 7 * alpha[0] * std::pow(r2[7], 3)*z[0];
	double cc_7_10 = alpha[7] * std::pow(r2[0], 3)*r2[7] - alpha[0] * r2[0] * std::pow(r2[7], 3);
	double cc_7_11 = 21 * alpha[7] * std::pow(r2[0], 3)*std::pow(z[7], 2) - 21 * alpha[0] * std::pow(r2[7], 3)*std::pow(z[0], 2);
	double cc_7_12 = r2[0] * std::pow(r2[7], 3) - std::pow(r2[0], 3)*r2[7];
	double cc_7_13 = 5 * alpha[7] * std::pow(r2[0], 3)*r2[7] * z[7] - 5 * alpha[0] * r2[0] * std::pow(r2[7], 3)*z[0];
	double cc_7_14 = 35 * alpha[7] * std::pow(r2[0], 3)*std::pow(z[7], 3) - 35 * alpha[0] * std::pow(r2[7], 3)*std::pow(z[0], 3);
	double cc_7_15 = 4 * r2[0] * std::pow(r2[7], 3)*z[0] - 4 * std::pow(r2[0], 3)*r2[7] * z[7];
	double cc_7_16 = 10 * alpha[7] * std::pow(r2[0], 3)*r2[7] * std::pow(z[7], 2) - 10 * alpha[0] * r2[0] * std::pow(r2[7], 3)*std::pow(z[0], 2);
	double cc_7_17 = alpha[7] * std::pow(r2[0], 3)*std::pow(r2[7], 2) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[7], 3);
	double cc_7_18 = 35 * alpha[7] * std::pow(r2[0], 3)*std::pow(z[7], 4) - 35 * alpha[0] * std::pow(r2[7], 3)*std::pow(z[0], 4);
	double cc_7_19 = 6 * r2[0] * std::pow(r2[7], 3)*std::pow(z[0], 2) - 6 * std::pow(r2[0], 3)*r2[7] * std::pow(z[7], 2);
	double cc_7_20 = std::pow(r2[0], 2)*std::pow(r2[7], 3) - std::pow(r2[0], 3)*std::pow(r2[7], 2);
	double cc_7_21 = 10 * alpha[7] * std::pow(r2[0], 3)*r2[7] * std::pow(z[7], 3) - 10 * alpha[0] * r2[0] * std::pow(r2[7], 3)*std::pow(z[0], 3);
	double cc_7_22 = 3 * alpha[7] * std::pow(r2[0], 3)*std::pow(r2[7], 2)*z[7] - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[7], 3)*z[0];
	double cc_7_23 = 21 * alpha[7] * std::pow(r2[0], 3)*std::pow(z[7], 5) - 21 * alpha[0] * std::pow(r2[7], 3)*std::pow(z[0], 5);
	double cc_7_24 = 4 * r2[0] * std::pow(r2[7], 3)*std::pow(z[0], 3) - 4 * std::pow(r2[0], 3)*r2[7] * std::pow(z[7], 3);
	double cc_7_25 = 2 * std::pow(r2[0], 2)*std::pow(r2[7], 3)*z[0] - 2 * std::pow(r2[0], 3)*std::pow(r2[7], 2)*z[7];
	double cc_7_26 = 5 * alpha[7] * std::pow(r2[0], 3)*r2[7] * std::pow(z[7], 4) - 5 * alpha[0] * r2[0] * std::pow(r2[7], 3)*std::pow(z[0], 4);
	double cc_7_27 = 3 * alpha[7] * std::pow(r2[0], 3)*std::pow(r2[7], 2)*std::pow(z[7], 2) - 3 * alpha[0] * std::pow(r2[0], 2)*std::pow(r2[7], 3)*std::pow(z[0], 2);
	double cc_7_28 = alpha[7] * std::pow(r2[0], 3)*std::pow(r2[7], 3) - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[7], 3);
	double cc_7_29 = 7 * alpha[7] * std::pow(r2[0], 3)*std::pow(z[7], 6) - 7 * alpha[0] * std::pow(r2[7], 3)*std::pow(z[0], 6);
	double cc_7_30 = r2[0] * std::pow(r2[7], 3)*std::pow(z[0], 4) - std::pow(r2[0], 3)*r2[7] * std::pow(z[7], 4);
	double cc_7_31 = std::pow(r2[0], 2)*std::pow(r2[7], 3)*std::pow(z[0], 2) - std::pow(r2[0], 3)*std::pow(r2[7], 2)*std::pow(z[7], 2);
	double cc_7_32 = alpha[7] * std::pow(r2[0], 3)*r2[7] * std::pow(z[7], 5) - alpha[0] * r2[0] * std::pow(r2[7], 3)*std::pow(z[0], 5);
	double cc_7_33 = alpha[7] * std::pow(r2[0], 3)*std::pow(r2[7], 2)*std::pow(z[7], 3) - alpha[0] * std::pow(r2[0], 2)*std::pow(r2[7], 3)*std::pow(z[0], 3);
	double cc_7_34 = alpha[7] * std::pow(r2[0], 3)*std::pow(r2[7], 3)*z[7] - alpha[0] * std::pow(r2[0], 3)*std::pow(r2[7], 3)*z[0];
	double cc_7_35 = alpha[7] * std::pow(r2[0], 3)*std::pow(z[7], 7) - alpha[0] * std::pow(r2[7], 3)*std::pow(z[0], 7);

	Eigen::Matrix<double, 7, 28> AA;
	AA << -cc_1_6, -cc_1_24, -cc_1_25, -cc_1_26, -cc_1_27, -cc_1_28, -cc_1_29, -cc_1_5, -cc_1_19, -cc_1_20, -cc_1_21, -cc_1_22, -cc_1_23, -cc_1_4, -cc_1_15, -cc_1_16, -cc_1_17, -cc_1_18, -cc_1_3, -cc_1_12, -cc_1_13, -cc_1_14, -cc_1_2, -cc_1_10, -cc_1_11, -cc_1_1, -cc_1_9, -cc_1_8, -cc_2_6, -cc_2_24, -cc_2_25, -cc_2_26, -cc_2_27, -cc_2_28, -cc_2_29, -cc_2_5, -cc_2_19, -cc_2_20, -cc_2_21, -cc_2_22, -cc_2_23, -cc_2_4, -cc_2_15, -cc_2_16, -cc_2_17, -cc_2_18, -cc_2_3, -cc_2_12, -cc_2_13, -cc_2_14, -cc_2_2, -cc_2_10, -cc_2_11, -cc_2_1, -cc_2_9, -cc_2_8, -cc_3_6, -cc_3_24, -cc_3_25, -cc_3_26, -cc_3_27, -cc_3_28, -cc_3_29, -cc_3_5, -cc_3_19, -cc_3_20, -cc_3_21, -cc_3_22, -cc_3_23, -cc_3_4, -cc_3_15, -cc_3_16, -cc_3_17, -cc_3_18, -cc_3_3, -cc_3_12, -cc_3_13, -cc_3_14, -cc_3_2, -cc_3_10, -cc_3_11, -cc_3_1, -cc_3_9, -cc_3_8, -cc_4_6, -cc_4_24, -cc_4_25, -cc_4_26, -cc_4_27, -cc_4_28, -cc_4_29, -cc_4_5, -cc_4_19, -cc_4_20, -cc_4_21, -cc_4_22, -cc_4_23, -cc_4_4, -cc_4_15, -cc_4_16, -cc_4_17, -cc_4_18, -cc_4_3, -cc_4_12, -cc_4_13, -cc_4_14, -cc_4_2, -cc_4_10, -cc_4_11, -cc_4_1, -cc_4_9, -cc_4_8, -cc_5_6, -cc_5_24, -cc_5_25, -cc_5_26, -cc_5_27, -cc_5_28, -cc_5_29, -cc_5_5, -cc_5_19, -cc_5_20, -cc_5_21, -cc_5_22, -cc_5_23, -cc_5_4, -cc_5_15, -cc_5_16, -cc_5_17, -cc_5_18, -cc_5_3, -cc_5_12, -cc_5_13, -cc_5_14, -cc_5_2, -cc_5_10, -cc_5_11, -cc_5_1, -cc_5_9, -cc_5_8, -cc_6_6, -cc_6_24, -cc_6_25, -cc_6_26, -cc_6_27, -cc_6_28, -cc_6_29, -cc_6_5, -cc_6_19, -cc_6_20, -cc_6_21, -cc_6_22, -cc_6_23, -cc_6_4, -cc_6_15, -cc_6_16, -cc_6_17, -cc_6_18, -cc_6_3, -cc_6_12, -cc_6_13, -cc_6_14, -cc_6_2, -cc_6_10, -cc_6_11, -cc_6_1, -cc_6_9, -cc_6_8, -cc_7_6, -cc_7_24, -cc_7_25, -cc_7_26, -cc_7_27, -cc_7_28, -cc_7_29, -cc_7_5, -cc_7_19, -cc_7_20, -cc_7_21, -cc_7_22, -cc_7_23, -cc_7_4, -cc_7_15, -cc_7_16, -cc_7_17, -cc_7_18, -cc_7_3, -cc_7_12, -cc_7_13, -cc_7_14, -cc_7_2, -cc_7_10, -cc_7_11, -cc_7_1, -cc_7_9, -cc_7_8;
	Eigen::Matrix<double, 7, 7> A0;
	A0 << cc_1_7, cc_1_30, cc_1_31, cc_1_32, cc_1_33, cc_1_34, cc_1_35, cc_2_7, cc_2_30, cc_2_31, cc_2_32, cc_2_33, cc_2_34, cc_2_35, cc_3_7, cc_3_30, cc_3_31, cc_3_32, cc_3_33, cc_3_34, cc_3_35, cc_4_7, cc_4_30, cc_4_31, cc_4_32, cc_4_33, cc_4_34, cc_4_35, cc_5_7, cc_5_30, cc_5_31, cc_5_32, cc_5_33, cc_5_34, cc_5_35, cc_6_7, cc_6_30, cc_6_31, cc_6_32, cc_6_33, cc_6_34, cc_6_35, cc_7_7, cc_7_30, cc_7_31, cc_7_32, cc_7_33, cc_7_34, cc_7_35;
	Eigen::Matrix<double, 28, 28> B;
	B.setZero();
	B(7, 0) = 1;  B(8, 1) = 1;  B(9, 2) = 1; B(10, 3) = 1;
	B(11, 4) = 1; B(12, 6) = 1; B(13, 7) = 1; B(14, 8) = 1;
	B(15, 10) = 1; B(16, 11) = 1; B(17, 12) = 1; B(18, 13) = 1;
	B(19, 14) = 1; B(20, 15) = 1; B(21, 17) = 1; B(22, 18) = 1;
	B(23, 20) = 1; B(24, 21) = 1; B(25, 22) = 1; B(26, 24) = 1;
	B(27, 26) = 1;

	if (use_rescaling) {
		// Some preconditioning
		double s0 = A0.col(0).norm();
		A0.col(0) /= s0;
		AA.col(0) /= s0; AA.col(7) /= s0; AA.col(13) /= s0; AA.col(18) /= s0; AA.col(22) /= s0; AA.col(25) /= s0;
		double s1 = A0.col(1).norm();
		A0.col(1) /= s1;
		AA.col(1) /= s1; AA.col(8) /= s1; AA.col(14) /= s1; AA.col(19) /= s1;
		double s2 = A0.col(2).norm();
		A0.col(2) /= s2;
		AA.col(2) /= s2; AA.col(9) /= s2;
		double s3 = A0.col(3).norm();
		A0.col(3) /= s3;
		AA.col(3) /= s3; AA.col(10) /= s3; AA.col(15) /= s3; AA.col(20) /= s3; AA.col(23) /= s3;
		double s4 = A0.col(4).norm();
		A0.col(4) /= s4;
		AA.col(4) /= s4; AA.col(11) /= s4; AA.col(16) /= s4;
		double s5 = A0.col(5).norm();
		A0.col(5) /= s5;
		AA.col(5) /= s5;
		double s6 = A0.col(6).norm();
		A0.col(6) /= s6;
		AA.col(6) /= s6; AA.col(12) /= s6; AA.col(17) /= s6; AA.col(21) /= s6; AA.col(24) /= s6; AA.col(26) /= s6; AA.col(27) /= s6;
	}

	if (use_qz_solver) {
		// Solve as generalized eigenvalue problem
		B.block<7, 28>(0, 0) = AA;

		Eigen::Matrix<double, 28, 28> A;
		A.setIdentity();
		A.block<7, 7>(0, 0) = A0;

		Eigen::GeneralizedEigenSolver<Eigen::Matrix<double, 28, 28>> es(A, B, false);
		es.setMaxIterations(1000);

		if (es.info() != Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 28, 1> alphas = es.alphas();
		Eigen::Matrix<double, 28, 1> betas = es.betas();
		for (int k = 0; k < 28; k++) {
			if (std::fabs(alphas(k).imag()) < 1e-8 && std::fabs(betas(k)) > 1e-10)
				t3->push_back(alphas(k).real() / betas(k));
		}
	} else {
		// Convert to eigenvalue problem and solve using normal eigensolver
		AA = A0.partialPivLu().solve(AA);
		B.block<7, 28>(0, 0) = AA;

		Eigen::EigenSolver<Eigen::Matrix<double, 28, 28>> es(B);

		if (es.info() != Eigen::ComputationInfo::Success) {
			// Make sure EigenSolver converged
			return 0;
		}

		Eigen::Matrix<std::complex<double>, 28, 1> ev = es.eigenvalues();
		for (int k = 0; k < 28; k++) {
			if (std::fabs(ev(k).imag()) < 1e-8 && std::fabs(ev(k).real()) > 1e-8)
				t3->push_back(1.0 / ev(k).real());
		}
	}


	return t3->size();
}

extern template class radialpose::larsson_iccv19::Solver<1, 0, true>;
extern template class radialpose::larsson_iccv19::Solver<2, 0, true>;
extern template class radialpose::larsson_iccv19::Solver<3, 0, true>;
extern template class radialpose::larsson_iccv19::Solver<3, 3, true>;
extern template class radialpose::larsson_iccv19::Solver<1, 0, false>;
