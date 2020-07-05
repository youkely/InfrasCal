/* C++ Port of the planar solver from Magnus Oskarsson.
   See https://github.com/hamburgerlady/fast_planar_camera_pose/blob/master/solver_planar_p4pfr_fast.m
   for the original Matlab implementation.
   Please cite 
    Oskarsson, A fast minimal solver for absolute camera pose with unknown focal length and radial distortion from four planar points, arxiv
   if you use this solver.   

   Note that this solver assumes that coordinate system is chosen such that X(3,:) = 0 !

    */
#pragma once
#include "pose_estimator.h"
#include "../misc/distortion.h"
#include "../misc/refinement.h"

namespace radialpose {
	namespace oskarsson_arxiv18 {
		class PlanarSolver : public PoseEstimator<PlanarSolver> {
			
		public:
			PlanarSolver() {}

			int solve(const Points2D& image_points, const Points3D& world_points, std::vector<Camera>* poses) const;

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
	}
};