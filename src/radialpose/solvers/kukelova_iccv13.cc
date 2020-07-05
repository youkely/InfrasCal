#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include "kukelova_iccv13.h"
#include "../misc/univariate.h"

using namespace Eigen;
using std::complex;

static const double SMALL_NUMBER = 1e-8;

void radialpose::kukelova_iccv13::Radial1DSolver::p5p_radial_impl(const Points2D &x, const Points3D &X, std::vector<Camera>* poses) {
	// Setup nullspace
	Matrix<double, 8, 5> cc;
	for (int i = 0; i < 5; i++) {
		cc(0, i) = -x(1, i) * X(0, i);
		cc(1, i) = -x(1, i) * X(1, i);
		cc(2, i) = -x(1, i) * X(2, i);
		cc(3, i) = -x(1, i);
		cc(4, i) = x(0, i) * X(0, i);
		cc(5, i) = x(0, i) * X(1, i);
		cc(6, i) = x(0, i) * X(2, i);
		cc(7, i) = x(0, i);
	}

	Matrix<double, 8, 8> Q = cc.householderQr().householderQ();
	Matrix<double, 8, 3> N = Q.rightCols(3);

	// Compute coefficients for sylvester resultant
	double c11_1 = N(0, 1) * N(4, 1) + N(1, 1) * N(5, 1) + N(2, 1) * N(6, 1);
	double c12_1 = N(0, 1) * N(4, 2) + N(0, 2) * N(4, 1) + N(1, 1) * N(5, 2) + N(1, 2) * N(5, 1) + N(2, 1) * N(6, 2) + N(2, 2) * N(6, 1);
	double c12_2 = N(0, 0) * N(4, 1) + N(0, 1) * N(4, 0) + N(1, 0) * N(5, 1) + N(1, 1) * N(5, 0) + N(2, 0) * N(6, 1) + N(2, 1) * N(6, 0);
	double c13_1 = N(0, 2) * N(4, 2) + N(1, 2) * N(5, 2) + N(2, 2) * N(6, 2);
	double c13_2 = N(0, 0) * N(4, 2) + N(0, 2) * N(4, 0) + N(1, 0) * N(5, 2) + N(1, 2) * N(5, 0) + N(2, 0) * N(6, 2) + N(2, 2) * N(6, 0);
	double c13_3 = N(0, 0) * N(4, 0) + N(1, 0) * N(5, 0) + N(2, 0) * N(6, 0);
	double c21_1 = N(0, 1) * N(0, 1) + N(1, 1) * N(1, 1) + N(2, 1) * N(2, 1) - N(4, 1) * N(4, 1) - N(5, 1) * N(5, 1) - N(6, 1) * N(6, 1);
	double c22_1 = 2 * N(0, 1) * N(0, 2) + 2 * N(1, 1) * N(1, 2) + 2 * N(2, 1) * N(2, 2) - 2 * N(4, 1) * N(4, 2) - 2 * N(5, 1) * N(5, 2) - 2 * N(6, 1) * N(6, 2);
	double c22_2 = 2 * N(0, 0) * N(0, 1) + 2 * N(1, 0) * N(1, 1) + 2 * N(2, 0) * N(2, 1) - 2 * N(4, 0) * N(4, 1) - 2 * N(5, 0) * N(5, 1) - 2 * N(6, 0) * N(6, 1);
	double c23_1 = N(0, 2) * N(0, 2) + N(1, 2) * N(1, 2) + N(2, 2) * N(2, 2) - N(4, 2) * N(4, 2) - N(5, 2) * N(5, 2) - N(6, 2) * N(6, 2);
	double c23_2 = 2 * N(0, 0) * N(0, 2) + 2 * N(1, 0) * N(1, 2) + 2 * N(2, 0) * N(2, 2) - 2 * N(4, 0) * N(4, 2) - 2 * N(5, 0) * N(5, 2) - 2 * N(6, 0) * N(6, 2);
	double c23_3 = N(0, 0) * N(0, 0) + N(1, 0) * N(1, 0) + N(2, 0) * N(2, 0) - N(4, 0) * N(4, 0) - N(5, 0) * N(5, 0) - N(6, 0) * N(6, 0);

	double a0 = c11_1 * c11_1 * c23_3 * c23_3 - c11_1 * c12_2 * c22_2 * c23_3 - 2 * c11_1 * c13_3 * c21_1 * c23_3 + c11_1 * c13_3 * c22_2 * c22_2 + c12_2 * c12_2 * c21_1 * c23_3 - c12_2 * c13_3 * c21_1 * c22_2 + c13_3 * c13_3 * c21_1 * c21_1;
	double a1 = c11_1 * c13_2 * c22_2 * c22_2 + 2 * c13_2 * c13_3 * c21_1 * c21_1 + c12_2 * c12_2 * c21_1 * c23_2 + 2 * c11_1 * c11_1 * c23_2 * c23_3 - c11_1 * c12_1 * c22_2 * c23_3 - c11_1 * c12_2 * c22_1 * c23_3 - c11_1 * c12_2 * c22_2 * c23_2 - 2 * c11_1 * c13_2 * c21_1 * c23_3 - 2 * c11_1 * c13_3 * c21_1 * c23_2 + 2 * c11_1 * c13_3 * c22_1 * c22_2 + 2 * c12_1 * c12_2 * c21_1 * c23_3 - c12_1 * c13_3 * c21_1 * c22_2 - c12_2 * c13_2 * c21_1 * c22_2 - c12_2 * c13_3 * c21_1 * c22_1;
	double a2 = c11_1 * c11_1 * c23_2 * c23_2 + c13_2 * c13_2 * c21_1 * c21_1 + c11_1 * c13_1 * c22_2 * c22_2 + c11_1 * c13_3 * c22_1 * c22_1 + 2 * c13_1 * c13_3 * c21_1 * c21_1 + c12_2 * c12_2 * c21_1 * c23_1 + c12_1 * c12_1 * c21_1 * c23_3 + 2 * c11_1 * c11_1 * c23_1 * c23_3 - c11_1 * c12_1 * c22_1 * c23_3 - c11_1 * c12_1 * c22_2 * c23_2 - c11_1 * c12_2 * c22_1 * c23_2 - c11_1 * c12_2 * c22_2 * c23_1 - 2 * c11_1 * c13_1 * c21_1 * c23_3 - 2 * c11_1 * c13_2 * c21_1 * c23_2 + 2 * c11_1 * c13_2 * c22_1 * c22_2 - 2 * c11_1 * c13_3 * c21_1 * c23_1 + 2 * c12_1 * c12_2 * c21_1 * c23_2 - c12_1 * c13_2 * c21_1 * c22_2 - c12_1 * c13_3 * c21_1 * c22_1 - c12_2 * c13_1 * c21_1 * c22_2 - c12_2 * c13_2 * c21_1 * c22_1;
	double a3 = c11_1 * c13_2 * c22_1 * c22_1 + 2 * c13_1 * c13_2 * c21_1 * c21_1 + c12_1 * c12_1 * c21_1 * c23_2 + 2 * c11_1 * c11_1 * c23_1 * c23_2 - c11_1 * c12_1 * c22_1 * c23_2 - c11_1 * c12_1 * c22_2 * c23_1 - c11_1 * c12_2 * c22_1 * c23_1 - 2 * c11_1 * c13_1 * c21_1 * c23_2 + 2 * c11_1 * c13_1 * c22_1 * c22_2 - 2 * c11_1 * c13_2 * c21_1 * c23_1 + 2 * c12_1 * c12_2 * c21_1 * c23_1 - c12_1 * c13_1 * c21_1 * c22_2 - c12_1 * c13_2 * c21_1 * c22_1 - c12_2 * c13_1 * c21_1 * c22_1;
	double a4 = c11_1 * c11_1 * c23_1 * c23_1 - c11_1 * c12_1 * c22_1 * c23_1 - 2 * c11_1 * c13_1 * c21_1 * c23_1 + c11_1 * c13_1 * c22_1 * c22_1 + c12_1 * c12_1 * c21_1 * c23_1 - c12_1 * c13_1 * c21_1 * c22_1 + c13_1 * c13_1 * c21_1 * c21_1;

	std::complex<double> roots[4];

	// This gives us the value for x
	solve_quartic(a1 / a0, a2 / a0, a3 / a0, a4 / a0, roots);
	

	for (int i = 0; i < 4; i++) {
		if (std::abs(roots[i].imag()) > 1e-6)
			continue;

		// We have two quadratic polynomials in y after substituting x
		double a = roots[i].real();
		double c1a = c11_1;
		double c1b = c12_1 + c12_2 * a;
		double c1c = c13_1 + c13_2 * a + c13_3 * a * a;

		double c2a = c21_1;
		double c2b = c22_1 + c22_2 * a;
		double c2c = c23_1 + c23_2 * a + c23_3 * a * a;

		// we solve the first one
		std::complex<double> bb[2];
		solve_quadratic(c1a, c1b, c1c, bb);

		if (std::abs(bb[0].imag()) > 1e-6)
			continue;

		// and check the residuals of the other
		double res1 = c2a * bb[0].real() * bb[0].real() + c2b * bb[0].real() + c2c;
		double res2;
		
		// For data where X(3,:) = 0 there is only a single solution
		// In this case the second solution will be NaN
		if (std::isnan(bb[1].real())) 
			res2 = std::numeric_limits<double>::max();
		else
			res2 = c2a * bb[1].real() * bb[1].real() + c2b * bb[1].real() + c2c;

		double b = (std::abs(res1) > std::abs(res2)) ? bb[1].real() : bb[0].real();


		Matrix<double, 8, 1> p = N.col(0) * a + N.col(1) * b + N.col(2);

		Camera pose;

		Vector3d r1, r2, r3, t;
		r1 << p(0), p(1), p(2);
		r2 << p(4), p(5), p(6);
		t << p(3), p(7), 0.0;

		double scale = r1.norm();
		r1 /= scale;
		r2 /= scale;
		t /= scale;
		r3 = r1.cross(r2);

		pose.R.row(0) = r1;
		pose.R.row(1) = r2;
		pose.R.row(2) = r3;
		pose.t = t;

		poses->push_back(pose);
	}
}


int radialpose::kukelova_iccv13::Solver::solve(const Points2D& x, const Points3D& X, std::vector<Camera>* poses) const
{
	if(use_radial_solver) {
		Radial1DSolver::p5p_radial_impl(x, X, poses);
	} else {
		poses->push_back(Camera(Matrix3d::Identity(), Vector3d::Zero()));
	}

	Matrix<double, 5, Dynamic> A;
	Matrix<double, 5, 1> b;
	A.resize(5, n_d + 2);

	for (int i = 0; i < poses->size(); ++i) {

		Camera &p = (*poses)[i];

		// solve for delta, p34, k1...k_n_d, delta, p34
		for (int j = 0; j < 5; ++j) {

			double qX = p.R.row(2) * X.col(j);
			double pX;
			
			// inverse focal length and p34
			if (std::abs(x(0, j)) < SMALL_NUMBER) {
				// use y-coordinate
				pX = p.R.row(1) * X.col(j) + p.t(1);
				
				A(j, 0) = -x(1, j) * qX;
				A(j, 1) = -x(1, j);
			} else {
				// use x-coordinate
				pX = p.R.row(0) * X.col(j) + p.t(0);
				
				A(j, 0) = -x(0, j) * qX;
				A(j, 1) = -x(0, j);
			}

			// Constant term
			b(j) = -pX;

			// Distortion parameters
			double r2 = x.col(j).squaredNorm();
			double r2n = r2;
			for (int k = 0; k < n_d; ++k) {
				A(j, 2 + k) = r2n * pX;
				r2n *= r2;
			}
		}
		
		VectorXd sol;
		if (n_d == 3) {
			// minimal problem
			sol = A.partialPivLu().solve(b);
		} else {
			// least squares solution
			sol = A.colPivHouseholderQr().solve(b);
		}

		
		p.focal = 1.0 / sol(0);
		p.t(2) = sol(1) * p.focal;

		if (p.focal < 0) {
			// flipped solution
			p.focal = -p.focal;
			p.R.row(0) = -p.R.row(0);
			p.R.row(1) = -p.R.row(1);
			p.t(0) = -p.t(0);
			p.t(1) = -p.t(1);			
		}

		double f2 = p.focal * p.focal;
		double fk = f2;
		for (int k = 0; k < n_d; k++) {
			p.dist_params.push_back(sol(2 + k) * fk);
			fk *= f2;
		}
	}


	return poses->size();
}


int radialpose::kukelova_iccv13::Radial1DSolver::solve(const Points2D& image_points, const Points3D& world_points, std::vector<Camera>* poses) const
{
	p5p_radial_impl(image_points, world_points, poses);
	return poses->size();			
}

template class radialpose::PoseEstimator<radialpose::kukelova_iccv13::Radial1DSolver>;
template class radialpose::PoseEstimator<radialpose::kukelova_iccv13::Solver>;