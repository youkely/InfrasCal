#include "refinement.h"

using namespace Eigen;
using namespace radialpose;

static const double SMALL_NUMBER = 1e-8;
static const double TOL_CONVERGENCE = 1e-10;
static const double INITIAL_LM_DAMP = 1e-6;
static const int MAX_ITER = 10;

/*
 TODO: add check that cost decreases in LM
 TODO: move radial refinement from larsson_iccv19.cc to here
 TODO: r_pow[] thing for refinement_dist as well...

*/

inline void drot(const Matrix3d &R, Matrix3d *dr1, Matrix3d *dr2, Matrix3d *dr3) {
	// skew = [0 -v(2) v(1); v(2) 0 -v(0); -v(1) v(0) 0]

	// dr1 = [0 0 0; 0 0 -1; 0 1 0]*R
	dr1->row(0).setZero();
	dr1->row(1) = -R.row(2);
	dr1->row(2) = R.row(1);

	// dr2 = [0 0 1; 0 0 0; -1 0 0]*R
	dr2->row(0) = R.row(2);
	dr2->row(1).setZero();
	dr2->row(2) = -R.row(0);

	// dr3 = [0 -1 0; 1 0 0; 0 0 0]*R
	dr3->row(0) = -R.row(1);
	dr3->row(1) = R.row(0);
	dr3->row(2).setZero();
}

inline void update_rot(Vector3d &v, Matrix3d &rot) {
	double stheta = v.norm();
	if (stheta < SMALL_NUMBER)
		return;
	v /= stheta;
	double theta = asin(stheta);
	Matrix3d K;
	K << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
	Matrix3d deltaR = Matrix3d::Identity() + stheta * K + (1 - cos(theta))*K*K;
	rot = deltaR * rot;
}

void radialpose::refinement_dist(const Eigen::Matrix<double, 2, Eigen::Dynamic> &x, const Eigen::Matrix<double, 3, Eigen::Dynamic> &X, radialpose::Camera &p, int Np, int Nd) {

	int n_pts = x.cols();
	int n_res = 2 * n_pts;
	int n_params = 6 + 1 + Np + Nd;
	double lm_damp = INITIAL_LM_DAMP;

	Matrix3d dr1, dr2, dr3;

	// Order for jacobian is: rotation, translation, focal, dist_params
	Matrix<double, Dynamic, Dynamic> J(n_res, n_params);
	J.setZero();
	Matrix<double, Dynamic, 1> res(n_res, 1);
	Matrix<double, Dynamic, 1> dx(n_params, 1);
	res.setZero();

	Matrix<double, 3, Dynamic> Z = X;

	// Change of variables to simplify equations
	for (int i = 0; i < Np; ++i)
		p.dist_params[i] *= p.focal;

	Matrix<double, Dynamic, Dynamic> H;
	Matrix<double, Dynamic, 1> g;

	for (int iter = 0; iter < MAX_ITER; ++iter) {
		// Z = R*Z + t 
		Z = p.R * X;
		Z.colwise() += p.t;

		drot(p.R, &dr1, &dr2, &dr3);

		for (int i = 0; i < n_pts; ++i) {
			double d = Z(2, i);
			double d2 = d * d;
			double r2 = Z.block<2, 1>(0, i).squaredNorm();

			double num = p.focal;
			double denom = d;

			double dnum_dt3 = 0.0;
			double ddenom_dt3 = 1.0;


			for (int k = 0; k < Np; k++) {
				double r2d2 = std::pow(r2 / d2, k + 1);

				num += p.dist_params[k] * r2d2;
				dnum_dt3 += -p.dist_params[k] * 2 * (k + 1) * r2d2 / d;

			}
			for (int k = 0; k < Nd; k++) {
				double r2d2 = std::pow(r2 / d2, k + 1);

				denom += d * p.dist_params[Np + k] * r2d2;
				ddenom_dt3 += -p.dist_params[Np + k] * (2 * k + 1)*r2d2;
			}

			double factor = num / denom;
			double dfactor_dt3 = (dnum_dt3*denom - num * ddenom_dt3) / (denom*denom);
			double dfactor_df = 1 / denom;

			res(2 * i + 0) = factor * Z(0, i) - x(0, i);
			res(2 * i + 1) = factor * Z(1, i) - x(1, i);

			// rotation
			// dfactor_dz = [0 0 dfactor_dt3]
			// dfactor_dr = dfactor_dz * dz_dr
			// d(factor*Z(0,i))_dr = dfactor_dr*Z(0,i) + factor * dZ(0,i)_dr

			// dZ_dr1
			Vector3d dZ_dr1 = dr1 * X.col(i);
			Vector3d dZ_dr2 = dr2 * X.col(i);
			Vector3d dZ_dr3 = dr3 * X.col(i);

			double dfactor_dr1 = dfactor_dt3 * dZ_dr1(2);
			double dfactor_dr2 = dfactor_dt3 * dZ_dr2(2);
			double dfactor_dr3 = dfactor_dt3 * dZ_dr3(2);

			J(2 * i + 0, 0) = dfactor_dr1 * Z(0, i) + factor * dZ_dr1(0);
			J(2 * i + 0, 1) = dfactor_dr2 * Z(0, i) + factor * dZ_dr2(0);
			J(2 * i + 0, 2) = dfactor_dr3 * Z(0, i) + factor * dZ_dr3(0);

			J(2 * i + 1, 0) = dfactor_dr1 * Z(1, i) + factor * dZ_dr1(1);
			J(2 * i + 1, 1) = dfactor_dr2 * Z(1, i) + factor * dZ_dr2(1);
			J(2 * i + 1, 2) = dfactor_dr3 * Z(1, i) + factor * dZ_dr3(1);

			// t_x
			J(2 * i + 0, 3) = factor;
			
			// t_y
			J(2 * i + 1, 4) = factor;

			// t_z
			J(2 * i + 0, 5) = dfactor_dt3 * Z(0, i);
			J(2 * i + 1, 5) = dfactor_dt3 * Z(1, i);
			
			// focal
			J(2 * i + 0, 6) = dfactor_df * Z(0, i);
			J(2 * i + 1, 6) = dfactor_df * Z(1, i);

			// dist_params[0..Np-1]
			for (int k = 0; k < Np; ++k) {
				double r2d2 = std::pow(r2 / d2, k + 1);
				double dfactor_dmu = r2d2 / denom;

				J(2 * i + 0, 7 + k) = dfactor_dmu * Z(0, i);
				J(2 * i + 1, 7 + k) = dfactor_dmu * Z(1, i);
			}

			// dist_params[Np..end]
			for (int k = 0; k < Nd; ++k) {
				double r2d2 = std::pow(r2 / d2, k + 1);
				double dfactor_dlambda = -d * r2d2 * num / (denom*denom);
				J(2 * i + 0, 7 + Np + k) = dfactor_dlambda * Z(0, i);
				J(2 * i + 1, 7 + Np + k) = dfactor_dlambda * Z(1, i);
			}
		}

		if (res.norm() < TOL_CONVERGENCE)
			break;

		H = J.transpose()*J;
		H.diagonal().array() += lm_damp; // LM dampening
		g = -J.transpose()*res;


		if (g.cwiseAbs().maxCoeff() < TOL_CONVERGENCE)
			break;

		//std::cout << "iter=" << iter << " res=" << res.squaredNorm() << ", g="<< g.squaredNorm() << "\n";
		//std::cout << res << "\n";

		dx = H.ldlt().solve(g);

		Vector3d dx_r = dx.block<3, 1>(0, 0);
		update_rot(dx_r, p.R);
		p.t(0) += dx(3);
		p.t(1) += dx(4);
		p.t(2) += dx(5);
		p.focal += dx(6);

		for (int i = 0; i < Np; ++i)
			p.dist_params[i] += dx(7 + i);
		for (int i = 0; i < Nd; ++i)
			p.dist_params[i + Np] += dx(Np + 7 + i);

		if (dx.array().abs().maxCoeff() < SMALL_NUMBER)
			break;
		lm_damp = std::max(1e-8, lm_damp / 10.0);
	}

	for (int i = 0; i < Np; ++i)
		p.dist_params[i] /= p.focal;

}

void radialpose::refinement_undist(const Eigen::Matrix<double, 2, Eigen::Dynamic> &x, const Eigen::Matrix<double, 3, Eigen::Dynamic> &X, radialpose::Camera &p, int Np, int Nd) {
	int n_pts = x.cols();
	int n_res = 2 * n_pts;
	int n_params = 6 + 1 + Np + Nd;

	Matrix3d dr1, dr2, dr3;
	double lm_damp = INITIAL_LM_DAMP;

	// Order for jacobian is: rotation, translation, focal, dist_params
	Matrix<double, Dynamic, Dynamic> J(n_res, n_params);
	J.setZero();
	Matrix<double, Dynamic, 1> res(n_res, 1);
	Matrix<double, Dynamic, 1> dx(n_params, 1);
	res.setZero();

	Matrix<double, 3, Dynamic> Z = X;

	// Change of variables to simplify equations
	double f2 = p.focal * p.focal;
	double f2k = f2;
	for (int i = 0; i < Np; ++i) {
		p.dist_params[i] /= f2k;
		f2k *= f2;
	}
	f2k = p.focal;
	for (int i = 0; i < Nd; ++i) {
		p.dist_params[Np + i] /= f2k;
		f2k *= f2;
	}

	double r_pow[3];
	Matrix<double, Dynamic, Dynamic> H;
	Matrix<double, Dynamic, 1> g;
	int iter;
	for (iter = 0; iter < MAX_ITER; ++iter) {
		// Z = R*Z + t 
		Z = p.R * X;
		Z.colwise() += p.t;

		drot(p.R, &dr1, &dr2, &dr3);

		for (int i = 0; i < n_pts; ++i) {
			double d = Z(2, i);
			double r2 = x.col(i).squaredNorm();

			double num = 1.0;
			double denom = p.focal;

			r_pow[0] = r2; // 2
			r_pow[1] = r2 * r2; // 4
			r_pow[2] = r2 * r_pow[1]; // 6


			for (int k = 0; k < Np; k++) {
				num += p.dist_params[k] * r_pow[k];
			}
			for (int k = 0; k < Nd; k++) {
				denom += p.dist_params[Np + k] * r_pow[k];
			}

			double factor = num / denom;

			double dfactor_df = -factor / denom;

			res(2 * i + 0) = factor * x(0, i) - Z(0, i) / d;
			res(2 * i + 1) = factor * x(1, i) - Z(1, i) / d;

			// Rotation
			
			// d(-Z(1)/Z(3)) = dZ(1)/Z(3) - Z(1)*dZ(3)/Z(3)^2

			Vector3d dZ_dr1 = dr1 * X.col(i);
			Vector3d dZ_dr2 = dr2 * X.col(i);
			Vector3d dZ_dr3 = dr3 * X.col(i);

			J(2 * i + 0, 0) = - dZ_dr1(0) / Z(2, i) + Z(0, i) * dZ_dr1(2) / (Z(2, i)*Z(2, i));
			J(2 * i + 0, 1) = - dZ_dr2(0) / Z(2, i) + Z(0, i) * dZ_dr2(2) / (Z(2, i)*Z(2, i));
			J(2 * i + 0, 2) = - dZ_dr3(0) / Z(2, i) + Z(0, i) * dZ_dr3(2) / (Z(2, i)*Z(2, i));

			J(2 * i + 1, 0) = -dZ_dr1(1) / Z(2, i) + Z(1, i) * dZ_dr1(2) / (Z(2, i)*Z(2, i));
			J(2 * i + 1, 1) = -dZ_dr2(1) / Z(2, i) + Z(1, i) * dZ_dr2(2) / (Z(2, i)*Z(2, i));
			J(2 * i + 1, 2) = -dZ_dr3(1) / Z(2, i) + Z(1, i) * dZ_dr3(2) / (Z(2, i)*Z(2, i));

			// tx
			J(2 * i + 0, 3) = - 1.0 / d; // tx
			J(2 * i + 1, 4) = - 1.0 / d; // ty

			// tz
			J(2 * i + 0, 5) = Z(0, i) / (d*d);
			J(2 * i + 1, 5) = Z(1, i) / (d*d);

			// focal
			J(2 * i + 0, 6) = dfactor_df * x(0, i);
			J(2 * i + 1, 6) = dfactor_df * x(1, i);

			// dist_params[0..Np-1]
			for (int k = 0; k < Np; ++k) {
				double dfactor_dmu = r_pow[k] / denom;
				J(2 * i + 0, 7 + k) = dfactor_dmu * x(0, i);
				J(2 * i + 1, 7 + k) = dfactor_dmu * x(1, i);
			}

			// dist_params[Np..end]
			for (int k = 0; k < Nd; ++k) {
				double dfactor_dlambda = -r_pow[k] * num / (denom*denom);
				J(2 * i + 0, 7 + Np + k) = dfactor_dlambda * x(0, i);
				J(2 * i + 1, 7 + Np + k) = dfactor_dlambda * x(1, i);
			}
		}

		if (res.norm() < TOL_CONVERGENCE)
			break;

		H = J.transpose()*J;
		H.diagonal().array() += lm_damp; // LM dampening
		g = -J.transpose()*res;


		if (g.cwiseAbs().maxCoeff() < TOL_CONVERGENCE)
			break;

		//std::cout << "iter=" << iter << " res=" << res.squaredNorm() << ", g="<< g.squaredNorm() << "\n";
		//std::cout << res << "\n";
		//std::cout << J << "\n";

		dx = H.ldlt().solve(g);

		Vector3d dx_r = dx.block<3, 1>(0, 0);
		update_rot(dx_r, p.R);
		p.t(0) += dx(3);
		p.t(1) += dx(4);
		p.t(2) += dx(5);
		p.focal += dx(6);

		for (int i = 0; i < Np; ++i)
			p.dist_params[i] += dx(7 + i);
		for (int i = 0; i < Nd; ++i)
			p.dist_params[i + Np] += dx(Np + 7 + i);

		if (dx.array().abs().maxCoeff() < SMALL_NUMBER)
			break;

		lm_damp = std::max(1e-8, lm_damp / 10.0);
	}

	//std::cout << "Local opt finished. iter=" << iter << ", res=" << res.norm() << ", g=" << g.norm() << "\n";

	// Revert change of variables
	f2 = p.focal * p.focal;
	f2k = f2;
	for (int i = 0; i < Np; ++i) {
		p.dist_params[i] *= f2k;
		f2k *= f2;
	}
	f2k = p.focal;
	for (int i = 0; i < Nd; ++i) {
		p.dist_params[Np + i] *= f2k;
		f2k *= f2;
	}
}
