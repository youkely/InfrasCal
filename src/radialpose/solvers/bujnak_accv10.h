/* C++ implementation of the solver from Bujnak et al. ACCV 2010
   Please cite
	Bujnak et al., New efficient solution to the absolute pose problem for camera with unknown focal length and radial distortion, ACCV 2010
   if you use this solver.
	*/
#pragma once
#include <Eigen/Dense>
#include "pose_estimator.h"
#include "../misc/distortion.h"
#include "../misc/refinement.h"
namespace radialpose {
	
	namespace bujnak_accv10 {

		class NonPlanarSolver : public PoseEstimator<NonPlanarSolver> {
		public:
			NonPlanarSolver() = default;
			int solve(Points2D& image_points, Points3D& world_points, std::vector<Camera>* poses) const;
			int minimal_sample_size() const {
				return 4;
			}	
			inline void distort(const std::vector<double>& dist_params, const Eigen::Matrix<double, 2, Eigen::Dynamic> xu, Eigen::Matrix<double, 2, Eigen::Dynamic>* xd) const {
				inverse_1param_division_model(dist_params[0], xu, xd);
			}

			inline void refine(Camera &pose, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x, const Eigen::Matrix<double, 3, Eigen::Dynamic> &X) const {
				radialpose::refinement_undist(x, X, pose, 0, 1);
			}
		};

	};
};