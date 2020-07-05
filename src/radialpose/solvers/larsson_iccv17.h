#pragma once
/* C++ implementation of one of the non-planar solvers from Larsson et al. (ICCV17)
   Please cite
	Larsson et al., Making Minimal Solvers for Absolute Pose Estimation Compact and Robust, ICCV 2017
   if you use this solver.
	*/
#include <Eigen/Dense>
#include "pose_estimator.h"
#include "../misc/distortion.h"
#include "../misc/refinement.h"

namespace radialpose {
	namespace larsson_iccv17 {

		class NonPlanarSolver : public PoseEstimator<NonPlanarSolver> {
		public:
			NonPlanarSolver() = default;
			int solve(const Points2D &image_points, const Points3D &world_points, std::vector<Camera>* poses) const;
			
			int minimal_sample_size() const {
				return 4;
			}
			void distort(const std::vector<double>& dist_params, const Eigen::Matrix<double, 2, Eigen::Dynamic> xu, Eigen::Matrix<double, 2, Eigen::Dynamic>* xd) const {
				inverse_1param_division_model(dist_params[0], xu, xd);
			}

			inline void refine(Camera &pose, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x, const Eigen::Matrix<double, 3, Eigen::Dynamic> &X) const {
				radialpose::refinement_undist(x, X, pose, 0, 1);
			}
		};
	};

};