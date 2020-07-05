/* C++ implementation of the solvers from Larsson et al. ICCV 2019
   Please cite
	Larsson et al., Revisiting Radial Distortion Absolute Pose, ICCV 2019
   if you use this solver.
	*/
#pragma once
#include <Eigen/Dense>
#include "pose_estimator.h"
#include "../misc/refinement.h"
#include "../misc/distortion.h"

namespace radialpose {
	namespace larsson_iccv19 {
		template<int Np,int Nd, bool DistortionModel = true>
		class Solver : public PoseEstimator<Solver<Np,Nd,DistortionModel>> {
		public:
			Solver() = default;
			int solve(const Points2D& image_points, const Points3D& world_points, std::vector<Camera>* poses) const;
			
			int minimal_sample_size() const {
				return std::max(5, 2 + Np + Nd);
			}

			inline void distort(const std::vector<double>& dist_params, const Eigen::Matrix<double, 2, Eigen::Dynamic> xu, Eigen::Matrix<double, 2, Eigen::Dynamic>* xd) const {
				if(DistortionModel) {
					forward_rational_model(dist_params, Np, Nd, xu, xd);
				} else {
					inverse_rational_model(dist_params, Np, Nd, xu, xd);
				}
			}

			inline void refine(Camera &pose, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x, const Eigen::Matrix<double, 3, Eigen::Dynamic> &X) const {
				if (DistortionModel)
					radialpose::refinement_dist(x, X, pose, Np, Nd);
				else
					radialpose::refinement_undist(x, X, pose, Np, Nd);
			}

			bool use_qz_solver = false;
			bool use_rescaling = true;
			bool use_precond = true;
			bool use_radial_solver = true;
			bool root_refinement = true;
			double damp_factor = 0.0;
		private:			
			int solver_impl(const Points2D& x, const Points3D& X, std::vector<double>* t3) const;			
		};
	};
};

