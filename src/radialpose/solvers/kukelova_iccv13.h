/* C++ implementation of the solvers from Kukelova et al. ICCV 2013
   Please cite
	Kukelova et al., Real-time solution to the absolute pose problem with unknown radial distortion and focal length, ICCV 2013
   if you use this solver.
	*/
#pragma once
#include "pose_estimator.h"
#include "../misc/distortion.h"
#include "../misc/refinement.h"

namespace radialpose {
	namespace kukelova_iccv13 {
		class Solver : public PoseEstimator<Solver> {
		public:
			Solver() : n_d(1) {}
			Solver(int model_div) : n_d(model_div) {}

			int solve(const Points2D& image_points, const Points3D& world_points, std::vector<Camera>* poses) const;

			int minimal_sample_size() const {
				return 5;
			}

			// TODO: Refactor distort. Cannot be static since we need n_d?
			inline void distort(const std::vector<double>& dist_params, const Eigen::Matrix<double, 2, Eigen::Dynamic> xu, Eigen::Matrix<double, 2, Eigen::Dynamic>* xd) const {
				inverse_rational_model(dist_params, 0, dist_params.size(), xu, xd);
			}

			inline void refine(Camera &pose, const Eigen::Matrix<double, 2, Eigen::Dynamic> &x, const Eigen::Matrix<double, 3, Eigen::Dynamic> &X) const {
				radialpose::refinement_undist(x, X, pose, 0, n_d);
			}

			bool use_radial_solver = true;
		private:
			int n_d;
		};

		class Radial1DSolver : public PoseEstimator<Radial1DSolver> {
		public:
			Radial1DSolver() {
				// Chirality is meaningless for radial cameras.
				check_chirality = false;
				// Radial reprojection error not implemented.
				check_reprojection_error = false;
			}

			int solve(const Points2D& image_points, const Points3D& world_points, std::vector<Camera>* poses) const;

			int minimal_sample_size() const {
				return 5;
			}

			inline void distort(const std::vector<double>& dist_params, const Eigen::Matrix<double, 2, Eigen::Dynamic> xu, Eigen::Matrix<double, 2, Eigen::Dynamic>* xd) const {
				*xd = xu; // No distortion model
			}

			inline void refine(Camera &pose, Eigen::Matrix<double, 2, Eigen::Dynamic> &x, Eigen::Matrix<double, 3, Eigen::Dynamic> &X) const {
				// TODO: implement
			}

			// This is made public since a lot of others solves build on this. TODO: better design choices
			static void p5p_radial_impl(const Points2D& x, const Points3D& X, std::vector<Camera>* poses);
		};

		
	};
};