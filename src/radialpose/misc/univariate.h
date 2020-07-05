#pragma once
#include <Eigen/Eigen>
#include <complex>

namespace radialpose {

	/* Solves the quadratic equation a*x^2 + b*x + c = 0 */
	void solve_quadratic(double a, double b, double c, std::complex<double> roots[2]);

	/* Solves the quartic equation p[2]*x^2 + p[1]*x + p[0] = 0 */
	void solve_quadratic(double* p, std::complex<double> roots[4]);

	/* Solves the cubic equation a*x^3 + b*x^2 + c*x + d = 0 */
	void solve_cubic(double a, double b, double c, double d, std::complex<double> roots[3]);

	/* Sign of component with largest magnitude */
	double sign2(const std::complex<double> z);

	/* Solves the quartic equation x^4 + b*x^3 + c*x^2 + d*x + e = 0 */
	void solve_quartic(double b, double c, double d, double e, std::complex<double> roots[4]);
	
	/* Solves the quartic equation a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 */
	void solve_quartic(double a, double b, double c, double d, double e, std::complex<double> roots[4]);
	
	/* Solves the quartic equation p[4]*x^4 + p[3]*x^3 + p[2]*x^2 + p[1]*x + p[0] = 0 */
	void solve_quartic(double* p, std::complex<double> roots[4]);

};