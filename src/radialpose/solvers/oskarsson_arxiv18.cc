#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include "oskarsson_arxiv18.h"

using namespace Eigen;
using std::complex;

int radialpose::oskarsson_arxiv18::PlanarSolver::solve(const Points2D& image_points, const Points3D& world_points, std::vector<Camera>* poses) const
{
	// Assumes X(3,:) = 0;
	Matrix<double, 3, 4> X = world_points.block<3, 4>(0, 0);
	Matrix<double, 2, 4> U = image_points.block<2, 4>(0, 0);
	Array<double, 1, 4> r = U.colwise().squaredNorm();
	Matrix<double, 4, 6> M;

	M << -U(1, 0) * X(0, 0), -U(1, 0) * X(1, 0), -U(1, 0), U(0, 0)* X(0, 0), U(0, 0)* X(1, 0), U(0, 0),
		-U(1, 1) * X(0, 1), -U(1, 1) * X(1, 1), -U(1, 1), U(0, 1)* X(0, 1), U(0, 1)* X(1, 1), U(0, 1),
		-U(1, 2) * X(0, 2), -U(1, 2) * X(1, 2), -U(1, 2), U(0, 2)* X(0, 2), U(0, 2)* X(1, 2), U(0, 2),
		-U(1, 3) * X(0, 3), -U(1, 3) * X(1, 3), -U(1, 3), U(0, 3)* X(0, 3), U(0, 3)* X(1, 3), U(0, 3);

	Matrix<double, 6, 6> Q = M.transpose().householderQr().householderQ();
	Matrix<double, 6, 2> N = Q.block<6, 2>(0, 4);


	Matrix<double, 3, 3> C;
	Matrix<double, 3, 4> D;

	C << U(0, 0) * X(0, 0), U(0, 0)* X(1, 0), U(0, 0),
		U(0, 1)* X(0, 1), U(0, 1)* X(1, 1), U(0, 1),
		U(0, 2)* X(0, 2), U(0, 2)* X(1, 2), U(0, 2);


	D << X(0, 0) * N(0, 0) + X(1, 0) * N(1, 0) + N(2, 0), r(0)* (X(0, 0) * N(0, 0) + X(1, 0) * N(1, 0) + N(2, 0)), r(0)* (X(0, 0) * N(0, 1) + X(1, 0) * N(1, 1) + N(2, 1)), X(0, 0)* N(0, 1) + X(1, 0) * N(1, 1) + N(2, 1),
		X(0, 1)* N(0, 0) + X(1, 1) * N(1, 0) + N(2, 0), r(1)* (X(0, 1) * N(0, 0) + X(1, 1) * N(1, 0) + N(2, 0)), r(1)* (X(0, 1) * N(0, 1) + X(1, 1) * N(1, 1) + N(2, 1)), X(0, 1)* N(0, 1) + X(1, 1) * N(1, 1) + N(2, 1),
		X(0, 2)* N(0, 0) + X(1, 2) * N(1, 0) + N(2, 0), r(2)* (X(0, 2) * N(0, 0) + X(1, 2) * N(1, 0) + N(2, 0)), r(2)* (X(0, 2) * N(0, 1) + X(1, 2) * N(1, 1) + N(2, 1)), X(0, 2)* N(0, 1) + X(1, 2) * N(1, 1) + N(2, 1);


	Matrix<double, 3, 4> CiD = C.partialPivLu().solve(D);


	double d11 = CiD(0, 0);
	double d12 = CiD(0, 1);
	double d13 = CiD(0, 2);
	double d14 = CiD(0, 3);
	double d21 = CiD(1, 0);
	double d22 = CiD(1, 1);
	double d23 = CiD(1, 2);
	double d24 = CiD(1, 3);
	double d31 = CiD(2, 0);
	double d32 = CiD(2, 1);
	double d33 = CiD(2, 2);
	double d34 = CiD(2, 3);
	double n11 = N(0, 0);
	double n12 = N(0, 1);
	double n21 = N(1, 0);
	double n22 = N(1, 1);
	double n31 = N(2, 0);
	double n32 = N(2, 1);
	double n41 = N(3, 0);
	double n42 = N(3, 1);
	double n51 = N(4, 0);
	double n52 = N(4, 1);
	// n61 = N(6, 0);
	// n62 = N(6, 1);

	double u4 = U(0, 3);
	double r4 = r(3);
	double x4 = X(0, 3);
	double y4 = X(1, 3);


	double knomy_b = n31 - d31 * u4 + n11 * x4 + n21 * y4 - d11 * u4 * x4 - d21 * u4 * y4;
	double knomy_1 = n32 - d34 * u4 + n12 * x4 + n22 * y4 - d14 * u4 * x4 - d24 * u4 * y4;
	double kdenny_b = d32 * u4 - n31 * r4 + d12 * u4 * x4 + d22 * u4 * y4 - n11 * r4 * x4 - n21 * r4 * y4;
	double kdenny_1 = d33 * u4 - n32 * r4 + d13 * u4 * x4 + d23 * u4 * y4 - n12 * r4 * x4 - n22 * r4 * y4;

	double c11_0 = n12 * n22 + n42 * n52;
	double c11_1 = n11 * n22 + n12 * n21 + n41 * n52 + n42 * n51;
	double c11_2 = n11 * n21 + n41 * n51;

	double c21_0 = n12 * n12 - n22 * n22 + n42 * n42 - n52 * n52;
	double c21_1 = 2 * n11 * n12 - 2 * n21 * n22 + 2 * n41 * n42 - 2 * n51 * n52;
	double c21_2 = n11 * n11 - n21 * n21 + n41 * n41 - n51 * n51;

	double c12_0 = (d14 * kdenny_1 + d13 * knomy_1) * (d24 * kdenny_1 + d23 * knomy_1);
	double c12_1 = (d24 * kdenny_1 + d23 * knomy_1) * (d11 * kdenny_1 + d14 * kdenny_b + d12 * knomy_1 + d13 * knomy_b) + (d14 * kdenny_1 + d13 * knomy_1) * (d21 * kdenny_1 + d24 * kdenny_b + d22 * knomy_1 + d23 * knomy_b);
	double c12_2 = (d11 * kdenny_1 + d14 * kdenny_b + d12 * knomy_1 + d13 * knomy_b) * (d21 * kdenny_1 + d24 * kdenny_b + d22 * knomy_1 + d23 * knomy_b) + (d14 * kdenny_1 + d13 * knomy_1) * (d21 * kdenny_b + d22 * knomy_b) + (d24 * kdenny_1 + d23 * knomy_1) * (d11 * kdenny_b + d12 * knomy_b);
	double c12_3 = (d21 * kdenny_b + d22 * knomy_b) * (d11 * kdenny_1 + d14 * kdenny_b + d12 * knomy_1 + d13 * knomy_b) + (d11 * kdenny_b + d12 * knomy_b) * (d21 * kdenny_1 + d24 * kdenny_b + d22 * knomy_1 + d23 * knomy_b);
	double c12_4 = (d11 * kdenny_b + d12 * knomy_b) * (d21 * kdenny_b + d22 * knomy_b);

	double c22_0 = (d14 * kdenny_1 + d24 * kdenny_1 + d13 * knomy_1 + d23 * knomy_1) * (d14 * kdenny_1 - d24 * kdenny_1 + d13 * knomy_1 - d23 * knomy_1);
	double c22_1 = (d14 * kdenny_1 - d24 * kdenny_1 + d13 * knomy_1 - d23 * knomy_1) * (d11 * kdenny_1 + d21 * kdenny_1 + d14 * kdenny_b + d24 * kdenny_b + d12 * knomy_1 + d22 * knomy_1 + d13 * knomy_b + d23 * knomy_b) + (d14 * kdenny_1 + d24 * kdenny_1 + d13 * knomy_1 + d23 * knomy_1) * (d11 * kdenny_1 - d21 * kdenny_1 + d14 * kdenny_b - d24 * kdenny_b + d12 * knomy_1 - d22 * knomy_1 + d13 * knomy_b - d23 * knomy_b);
	double c22_2 = (d11 * kdenny_1 + d21 * kdenny_1 + d14 * kdenny_b + d24 * kdenny_b + d12 * knomy_1 + d22 * knomy_1 + d13 * knomy_b + d23 * knomy_b) * (d11 * kdenny_1 - d21 * kdenny_1 + d14 * kdenny_b - d24 * kdenny_b + d12 * knomy_1 - d22 * knomy_1 + d13 * knomy_b - d23 * knomy_b) + (d14 * kdenny_1 + d24 * kdenny_1 + d13 * knomy_1 + d23 * knomy_1) * (d11 * kdenny_b - d21 * kdenny_b + d12 * knomy_b - d22 * knomy_b) + (d14 * kdenny_1 - d24 * kdenny_1 + d13 * knomy_1 - d23 * knomy_1) * (d11 * kdenny_b + d21 * kdenny_b + d12 * knomy_b + d22 * knomy_b);
	double c22_3 = (d11 * kdenny_b - d21 * kdenny_b + d12 * knomy_b - d22 * knomy_b) * (d11 * kdenny_1 + d21 * kdenny_1 + d14 * kdenny_b + d24 * kdenny_b + d12 * knomy_1 + d22 * knomy_1 + d13 * knomy_b + d23 * knomy_b) + (d11 * kdenny_b + d21 * kdenny_b + d12 * knomy_b + d22 * knomy_b) * (d11 * kdenny_1 - d21 * kdenny_1 + d14 * kdenny_b - d24 * kdenny_b + d12 * knomy_1 - d22 * knomy_1 + d13 * knomy_b - d23 * knomy_b);
	double c22_4 = (d11 * kdenny_b + d21 * kdenny_b + d12 * knomy_b + d22 * knomy_b) * (d11 * kdenny_b - d21 * kdenny_b + d12 * knomy_b - d22 * knomy_b);


	double poly[7];
	poly[0] = c11_2 * c22_4 - c12_4 * c21_2;
	poly[1] = c11_1 * c22_4 + c11_2 * c22_3 - c12_3 * c21_2 - c12_4 * c21_1;
	poly[2] = c11_0 * c22_4 + c11_1 * c22_3 + c11_2 * c22_2 - c12_2 * c21_2 - c12_3 * c21_1 - c12_4 * c21_0;
	poly[3] = c11_0 * c22_3 + c11_1 * c22_2 + c11_2 * c22_1 - c12_1 * c21_2 - c12_2 * c21_1 - c12_3 * c21_0;
	poly[4] = c11_0 * c22_2 + c11_1 * c22_1 + c11_2 * c22_0 - c12_0 * c21_2 - c12_1 * c21_1 - c12_2 * c21_0;
	poly[5] = c11_0 * c22_1 + c11_1 * c22_0 - c12_0 * c21_1 - c12_1 * c21_0;
	poly[6] = c11_0 * c22_0 - c12_0 * c21_0;

	for (int i = 1; i <= 6; ++i)
		poly[i] /= poly[0];

	// Setup companion matrix
	Matrix<double, 6, 6> Cp;
	Cp << -poly[1], -poly[2], -poly[3], -poly[4], -poly[5], -poly[6],
		1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 0;
		
	EigenSolver<Matrix<double,6,6>> es(Cp);
	Matrix<complex<double>, 6, 1> bsols = es.eigenvalues();
	
	for (int i = 0; i < 6; ++i) {
		if (std::fabs(bsols(i).imag()) > 1e-8)
			continue; // complex root

		double bsol = bsols(i).real();

		double bsol2 = bsol * bsol;
		double bsol3 = bsol2 * bsol;
		double bsol4 = bsol3 * bsol;

		double fsol2 = -(c11_0 + c11_1 * bsol + c11_2 * bsol2) / ((c12_0 + c12_1 * bsol + c12_2 * bsol2 + c12_3 * bsol3 + c12_4 * bsol4)) * (kdenny_b * bsol + kdenny_1) * (kdenny_b * bsol + kdenny_1);

		if (fsol2 < 0)
			continue; // complex focal length

		double fsol = std::sqrt(fsol2);

		double ksol = (knomy_b * bsol + knomy_1) / (kdenny_b * bsol + kdenny_1);

		Matrix<double, 6, 1> vsol = N.col(0) * bsol + N.col(1);
		Matrix<double, 3, 1> v2sol = CiD.col(0) * bsol + CiD.col(1) * (ksol * bsol) + CiD.col(2) * ksol + CiD.col(3);

		double nr2 = vsol(0) * vsol(0) + vsol(3) * vsol(3) + fsol2 * v2sol(0) * v2sol(0);
		double nr = std::sqrt(nr2);

		Matrix<double, 3, 3> R;

		R.col(0) << vsol(0) / nr, vsol(3) / nr, fsol* v2sol(0) / nr;
		R.col(1) << vsol(1) / nr, vsol(4) / nr, fsol* v2sol(1) / nr;

		
		R = R;
		R.col(2) = R.col(0).cross(R.col(1));

		Matrix<double, 3, 1> t;
		t << vsol(2), vsol(5), fsol* v2sol(2);
		t *= 1.0 / nr;

		// Correct sign
		if (R.determinant() < 0) {
			R = -R;
			t = -t;
		}

		Camera p(R, t, fsol);
		p.dist_params.push_back(ksol * fsol * fsol);
		poses->push_back(p);

		// Add flipped solution
		R = -R;
		t = -t;
		R.col(2) = -R.col(2);
		p.R = R;
		p.t = t;
		poses->push_back(p);
	}

	return poses->size();
}
