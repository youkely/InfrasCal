#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include "larsson_iccv19.h"
#include "kukelova_iccv13.h"
#include "../misc/univariate.h"
#include "../misc/distortion.h"

using namespace Eigen;
using namespace radialpose;
using std::complex;

static const double SMALL_NUMBER = 1e-8;
static const double DAMP_FACTOR = 1e-8;



static void linsolve_known_pose_dist(const Points2D &x, const Points3D &X, double t3, int Np, int Nd, double damp, Camera* camera)
{
	int n_pts = x.cols();
	int n_param = 1 + Np + Nd;

	int n_rows = n_pts;
	if (damp > 0)
		n_rows += n_param - 1;

	// System matrix for normal equations
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(n_rows, n_param);
	Eigen::Matrix<double, Eigen::Dynamic, 1> b(n_rows, 1);
	A.setZero();  b.setZero();

	double r_pow[4];
	for (int i = 0; i < n_pts; i++) {

		double uu, uv;
		if (std::abs(x(0, i)) < SMALL_NUMBER) {
			uv = x(1, i);
			uu = X(1, i) / (t3 + X(2, i));
		}
		else {
			uv = x(0, i);
			uu = X(0, i) / (t3 + X(2, i));
		}

		//double rd2 = x(0, i) * x(0, i) + x(1, i) * x(1, i);
		double ru2 = (X(0, i) * X(0, i) + X(1, i) * X(1, i)) / ((X(2, i) + t3) * (X(2, i) + t3));				

		// compute powers
		r_pow[0] = ru2;            // r^2
		r_pow[1] = ru2 * ru2;      // r^4
		r_pow[2] = ru2 * r_pow[1]; // r^6
		r_pow[3] = ru2 * r_pow[2]; // r^8

		A(i, 0) = uu;
		for (int k = 0; k < Np; k++)
			A(i, 1 + k) = r_pow[k] * uu;
		for (int k = 0; k < Nd; k++)
			A(i, 1 + Np + k) = -uv * r_pow[k];
		b(i) = uv;
	}
	if (damp > 0) {
		for (int i = 1; i < n_param; i++)
			A(n_pts - 1 + i, i) += damp;
	}

	//Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> svd(A);	
	//std::cout << "A: " << A << "\n";
	//std::cout << "svd(A): " << svd.singularValues() << "\n";

	Eigen::Matrix<double, Eigen::Dynamic, 1> sol = A.fullPivHouseholderQr().solve(b);
	camera->focal = sol(0);
	for (int i = 1; i < n_param; i++) {
		if (i <= Np)
			camera->dist_params.push_back(sol(i) / camera->focal);
		else
			camera->dist_params.push_back(sol(i));
	}
}

static void linsolve_known_pose_undist(const Points2D &x, const Points3D &X, double t3, int Np, int Nd, double damp, Camera* camera)
{
	int n_pts = x.cols();
	int n_param = 1 + Np + Nd;

	int n_rows = n_pts;
	if (damp > 0)
		n_rows += n_param - 1;

	// System matrix for normal equations
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(n_rows, n_param);
	Eigen::Matrix<double, Eigen::Dynamic, 1> b(n_rows, 1);
	A.setZero();  b.setZero();

	double r_pow[4];
	for (int i = 0; i < n_pts; i++) {

		double uu, uv;
		if (std::abs(x(0, i)) < SMALL_NUMBER) {
			uv = x(1, i);
			uu = X(1, i) / (t3 + X(2, i));
		} else {
			uv = x(0, i);
			uu = X(0, i) / (t3 + X(2, i));
		}

		double rd2 = x(0, i) * x(0, i) + x(1, i) * x(1, i);		
			
		// compute powers
		r_pow[0] = rd2;            // r^2
		r_pow[1] = rd2 * rd2;      // r^4
		r_pow[2] = rd2 * r_pow[1]; // r^6
		r_pow[3] = rd2 * r_pow[2]; // r^8

		A(i, 0) = uu;
		for (int k = 0; k < Np; k++)
			A(i, 1 + k) = -r_pow[k] * uv;
		for (int k = 0; k < Nd; k++)
			A(i, 1 + Np + k) = uu * r_pow[k];
		b(i) = uv;
	}
	if (damp > 0) {
		for (int i = 1; i < n_param; i++)
			A(n_pts - 1 + i, i) += damp;
	}


	Eigen::Matrix<double, Eigen::Dynamic, 1> sol = A.fullPivHouseholderQr().solve(b);
	camera->focal = sol(0);
	double f2 = camera->focal * camera->focal;
	double f2k = f2;
	// We have the real mu_k = sol * f^2k and lambda_k = sol * f^2k-1
	for (int i = 1; i < n_param; i++) {
		if (i == Np+1)
			f2k = camera->focal;
		camera->dist_params.push_back(sol(i) * f2k);
		f2k = f2k * f2;
	}
}


// Small refinement on the minimal sample. TODO: Move this to refinement.cc
template<int Np, int Nd>
void radial_refinement_dist(Camera &p, const Points2D &x, const Points3D &X, double damp_factor) {
	// It is assumed that X is already rotated by p
	constexpr int n_pts = std::max(2 + Np + Nd, 5);

	Matrix<double, 2 * n_pts, 2 + Np + Nd> J;
	Matrix<double, 2 * n_pts, 1> res;
	Matrix<double, 2 + Np + Nd, 1> dx;

	for (int i = 0; i < Np; ++i)
		p.dist_params[i] *= p.focal;

	for (int iter = 0; iter < 5; ++iter) {
		for (int i = 0; i < n_pts; ++i) {
			double d = X(2, i) + p.t(2);
			double d2 = d * d;
			double r2 = X.block<2, 1>(0, i).squaredNorm();

			double num = p.focal;
			double denom = d;

			double dnum_dt = 0.0;
			double ddenom_dt = 1.0;


			for (int k = 0; k < Np; k++) {
				double r2d2 = std::pow(r2 / d2, k + 1);

				num += p.dist_params[k] * r2d2;
				dnum_dt += - p.dist_params[k] * 2 * (k + 1) * r2d2 / d;

			}
			for (int k = 0; k < Nd; k++) {
				double r2d2 = std::pow(r2 / d2, k + 1);

				denom += d * p.dist_params[Np + k] * r2d2;
				ddenom_dt += -p.dist_params[Np + k] * (2 * k + 1)*r2d2;
			}

			double factor = num / denom;
			double dfactor_dt = (dnum_dt*denom - num * ddenom_dt) / (denom*denom);
			double dfactor_df = 1 / denom;

			res(2 * i + 0) = factor * X(0, i) - x(0, i);
			res(2 * i + 1) = factor * X(1, i) - x(1, i);

			J(2 * i + 0, 0) = dfactor_dt * X(0, i);
			J(2 * i + 1, 0) = dfactor_dt * X(1, i);
			J(2 * i + 0, 1) = dfactor_df * X(0, i);
			J(2 * i + 1, 1) = dfactor_df * X(1, i);

			for (int k = 0; k < Np; ++k) {
				double r2d2 = std::pow(r2 / d2, k + 1);
				double dfactor_dmu = r2d2 / denom;

				J(2 * i + 0, 2 + k) = dfactor_dmu * X(0, i);
				J(2 * i + 1, 2 + k) = dfactor_dmu * X(1, i);
			}

			for (int k = 0; k < Nd; ++k) {
				double r2d2 = std::pow(r2 / d2, k + 1);
				double dfactor_dlambda = -d * r2d2 * num / (denom*denom);
				J(2 * i + 0, 2 + Np + k) = dfactor_dlambda * X(0, i);
				J(2 * i + 1, 2 + Np + k) = dfactor_dlambda * X(1, i);
			}

		}

		if (res.norm() < SMALL_NUMBER)
			break;

		//std::cout << "res = " << res << "\n";
		//std::cout << "jac = " << J << "\n";

		Matrix<double, 2 + Np + Nd, 2 + Np + Nd> H = J.transpose()*J;
		H.diagonal().array() += 1e-6; // LM dampening
		Matrix<double, 2 + Np + Nd, 1> g = -J.transpose()*res;

		if (Nd > 0 && Np > 0 && damp_factor > 0) {
			// For rational models we add a small dampening factor			
			H.template block<Np + Nd,Np + Nd>(2, 2).diagonal().array() += damp_factor;
			for(int i = 0; i < Np+Nd; i++)
				g(2+i) -= damp_factor * p.dist_params[i];
		}


		dx = H.ldlt().solve(g);

		p.t(2) += dx(0);
		p.focal += dx(1);
		for (int i = 0; i < Np; ++i)
			p.dist_params[i] += dx(2 + i);
		for (int i = 0; i < Nd; ++i)
			p.dist_params[i + Np] += dx(Np + 2 + i);

		if (dx.array().abs().maxCoeff() < SMALL_NUMBER)
			break;
	}

	for (int i = 0; i < Np; ++i)
		p.dist_params[i] /= p.focal;
}


template<int Np, int Nd>
void radial_refinement_undist(Camera &p, const Points2D &x, const Points3D &X, double damp_factor) {
	// It is assumed that X is already rotated by p
	constexpr int n_pts = std::max(2 + Np + Nd, 5);

	Matrix<double, 2 * n_pts, 2 + Np + Nd> J;
	Matrix<double, 2 * n_pts, 1> res;
	Matrix<double, 2 + Np + Nd, 1> dx;

	// Change of variables
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

	for (int iter = 0; iter < 5; ++iter) {
		for (int i = 0; i < n_pts; ++i) {
			double d = X(2, i) + p.t(2);
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

			res(2 * i + 0) = factor * x(0, i) - X(0, i) / d;
			res(2 * i + 1) = factor * x(1, i) - X(1, i) / d;

			J(2 * i + 0, 0) = X(0, i) / (d*d); // t
			J(2 * i + 1, 0) = X(1, i) / (d*d);
			J(2 * i + 0, 1) = dfactor_df * x(0, i);
			J(2 * i + 1, 1) = dfactor_df * x(1, i);

			for (int k = 0; k < Np; ++k) {
				double dfactor_dmu = r_pow[k] / denom;
				J(2 * i + 0, 2 + k) = dfactor_dmu * x(0, i);
				J(2 * i + 1, 2 + k) = dfactor_dmu * x(1, i);
			}
			for (int k = 0; k < Nd; ++k) {
				double dfactor_dlambda = -r_pow[k] * num / (denom*denom);
				J(2 * i + 0, 2 + Np + k) = dfactor_dlambda * x(0, i);
				J(2 * i + 1, 2 + Np + k) = dfactor_dlambda * x(1, i);
			}
		}

		//std::cout << "res = " << res << "\n";
		//std::cout << "jac = " << J << "\n";
		if (res.norm() < SMALL_NUMBER)
			break;

		Matrix<double, 2 + Np + Nd, 2 + Np + Nd> H = J.transpose()*J;
		H.diagonal().array() += 1e-6; // LM dampening
		Matrix<double, 2 + Np + Nd, 1> g = -J.transpose()*res;

		if (Nd > 0 && Np > 0 && damp_factor > 0) {
			// For rational models we add a small dampening factor
			H.template block<Np + Nd,Np + Nd>(2, 2).diagonal().array() += damp_factor;
			for (int i = 0; i < Np + Nd; i++)
				g(2 + i) -= damp_factor * p.dist_params[i];
		}


		dx = H.ldlt().solve(g);

		p.t(2) += dx(0);
		p.focal += dx(1);
		for (int i = 0; i < Np; ++i)
			p.dist_params[i] += dx(2 + i);
		for (int i = 0; i < Nd; ++i)
			p.dist_params[i + Np] += dx(Np + 2 + i);

		if (dx.array().abs().maxCoeff() < SMALL_NUMBER)
			break;
	}

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

inline double simple_preconditioner(const Points2D &x, const Points3D &X) {
	// Simple preconditioner using the first two points
	double nx1 = x.col(0).squaredNorm();
	double nx2 = x.col(1).squaredNorm();
	double sx1 = X.block<2, 1>(0, 0).dot(x.col(0));
	double sx2 = X.block<2, 1>(0, 1).dot(x.col(1));
	return (X(2, 1)*nx2*sx1 - X(2, 0)*nx1*sx2) / (nx1 * sx2 - nx2 * sx1);
}

template<int Np, int Nd, bool DistortionModel>
int radialpose::larsson_iccv19::Solver<Np,Nd,DistortionModel>::solve(const Points2D& image_points, const Points3D& world_points, std::vector<Camera>* poses) const
{

	std::vector<Camera> initial_poses;
	std::vector<double> t3;

	if (use_radial_solver) {
		kukelova_iccv13::Radial1DSolver::p5p_radial_impl(image_points, world_points, &initial_poses);
	} else {
		initial_poses.push_back(Camera(Matrix3d::Identity(), Vector3d::Zero()));
	}
	Matrix<double, 3, Dynamic> X;

	for (int k = 0; k < initial_poses.size(); k++) {
		t3.clear();

		X = initial_poses[k].R * world_points;
		X.colwise() += initial_poses[k].t;

		//std::cout << "Initial pose " << k + 1 << "/" << initial_poses.size() << "\n";
		//std::cout << "R=" << initial_poses[k].R << "\nt=" << initial_poses[k].t << "\n";
		//std::cout << "X=" << X << "\n";

		double t0 = 0;
		if (use_precond) {
			t0 = simple_preconditioner(image_points, X);
			X.row(2).array() += t0;
		}

		solver_impl(image_points, X, &t3);

		for (int i = 0; i < t3.size(); ++i) {
			Camera pose;
			pose.R = initial_poses[k].R;
			pose.t = initial_poses[k].t;
			pose.t(2) = t3[i];

			if (DistortionModel) {
				linsolve_known_pose_dist(image_points, X, t3[i], Np, Nd, damp_factor, &pose);
				if(root_refinement)
					radial_refinement_dist<Np, Nd>(pose, image_points, X, damp_factor);
			} else {
				linsolve_known_pose_undist(image_points, X, t3[i], Np, Nd, damp_factor, &pose);
				if(root_refinement)
					radial_refinement_undist<Np, Nd>(pose, image_points, X, damp_factor);
			}			

			if (pose.focal < 0) {
				// flipped solution
				pose.focal = -pose.focal;
				pose.R.row(0) = -pose.R.row(0);
				pose.R.row(1) = -pose.R.row(1);
				pose.t(0) = -pose.t(0);
				pose.t(1) = -pose.t(1);
			}

			
			// Revert precond
			pose.t(2) += t0;

			//std::cout << "solution[" << i << "], t3=" << pose.t(2) << "\n";
			poses->push_back(pose);
		}
	}
	return poses->size();
}




// Template instantiations
template class radialpose::larsson_iccv19::Solver<1, 0, true>;
template class radialpose::larsson_iccv19::Solver<2, 0, true>;
template class radialpose::larsson_iccv19::Solver<3, 0, true>;
template class radialpose::larsson_iccv19::Solver<3, 3, true>;
template class radialpose::larsson_iccv19::Solver<1, 0, false>;
//template class radialpose::larsson_iccv19::Solver<2, 0, false>;
//template class radialpose::larsson_iccv19::Solver<3, 0, false>;
//template class radialpose::larsson_iccv19::Solver<3, 3, false>;

template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<1, 0, true>>;
template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<2, 0, true>>;
template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<3, 0, true>>;
template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<3, 3, true>>;
template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<1, 0, false>>;

/*
 This is broken?
// These are implemented in larsson_iccv19_impl.cc
extern template int radialpose::larsson_iccv19::Solver<1, 0, true>::solver_impl(Eigen::Matrix<double, 2, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>, std::vector<double>*);
extern template int radialpose::larsson_iccv19::Solver<2, 0, true>::solver_impl(Eigen::Matrix<double, 2, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>, std::vector<double>*);
extern template int radialpose::larsson_iccv19::Solver<3, 0, true>::solver_impl(Eigen::Matrix<double, 2, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>, std::vector<double>*);
extern template int radialpose::larsson_iccv19::Solver<3, 3, true>::solver_impl(Eigen::Matrix<double, 2, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>, std::vector<double>*);
extern template int radialpose::larsson_iccv19::Solver<1, 0, false>::solver_impl(Eigen::Matrix<double, 2, Eigen::Dynamic>, Eigen::Matrix<double, 3, Eigen::Dynamic>, std::vector<double>*);
*/