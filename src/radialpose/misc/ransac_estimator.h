#pragma once

#include <Eigen/Dense>
#include "camera.h"
#include "../solvers/pose_estimator.h"

namespace radialpose {

template<class Solver>
class RansacEstimator {
public:
	RansacEstimator(Eigen::Matrix<double,2,Eigen::Dynamic> im_pts, Eigen::Matrix<double,3,Eigen::Dynamic> w_pts, Solver est) {
		image_points = im_pts;
		world_points = w_pts;
		solver = est;
	}

	inline int min_sample_size() const {
		return solver.minimal_sample_size();
	}
	inline int non_minimal_sample_size() const {
		return solver.minimal_sample_size()*2;
	}
	inline int num_data() const {
		return image_points.cols();
	}

	int MinimalSolver(const std::vector<int>& sample,
		std::vector<Camera>* poses) const {
		Points2D x(2, sample.size());
		Points3D X(3, sample.size());

		for (int i = 0; i < sample.size(); i++) {
			x.col(i) = image_points.col(sample[i]);
			X.col(i) = world_points.col(sample[i]);
		}
		solver.estimate(x, X, poses);
				
		return poses->size();
	}

	// Returns 0 if no model could be estimated and 1 otherwise.
	int NonMinimalSolver(const std::vector<int>& sample,
		Camera* pose) const {
		if (!use_non_minimal)
			return 0;

		Eigen::Matrix<double, 2, Eigen::Dynamic> x(2, sample.size());
		Eigen::Matrix<double, 3, Eigen::Dynamic> X(3, sample.size());		

		for (int i = 0; i < sample.size(); i++) {
			x.col(i) = image_points.col(sample[i]);
			X.col(i) = world_points.col(sample[i]);
		}

		// Call minimal solver
		std::vector<Camera> poses;
		Points2D xs = x.block(0, 0, 2, min_sample_size());
		Points3D Xs = X.block(0, 0, 3, min_sample_size());
		solver.estimate(xs, Xs, &poses);

		// for all pose candidates compute score
		double best_score = std::numeric_limits<double>::max();
		int best_idx = -1;

		for (int i = 0; i < poses.size(); ++i) {
			double score = 0;
			for (int j = 0; j < sample.size(); ++j)
				score += EvaluateModelOnPoint(poses[i], sample[j]);
			if (score < best_score) {
				best_score = score;
				best_idx = i;
			}
		}

		if (best_idx != -1) {
			*pose = poses[best_idx];

			return 1;
		} else {
			return 0;
		}
	}

	// Evaluates the line on the i-th data point.
	double EvaluateModelOnPoint(const Camera& pose, int i) const {
		// Computer reprojection error
		Eigen::Vector3d Z = pose.R * world_points.col(i) + pose.t;

		Eigen::Matrix<double, 2, Eigen::Dynamic> z(2, 1);
		z << Z(0) / Z(2), Z(1) / Z(2);
		
		solver.distort(pose.dist_params, z, &z);
		z = pose.focal * z;

		return (z - image_points.col(i)).squaredNorm();
	}

	// Linear least squares solver. Calls NonMinimalSolver.
	inline void LeastSquares(const std::vector<int>& sample,
		Camera* p) const {
		if (!use_local_opt)
			return;
		Eigen::Matrix<double, 2, Eigen::Dynamic> x(2, sample.size());
		Eigen::Matrix<double, 3, Eigen::Dynamic> X(3, sample.size());

		for (int i = 0; i < sample.size(); i++) {
			x.col(i) = image_points.col(sample[i]);
			X.col(i) = world_points.col(sample[i]);
		}
		solver.refine(*p, x, X);
	}




	bool use_non_minimal = true;
	bool use_local_opt = true;
private:
	Solver solver;
	Eigen::Matrix<double, 2, Eigen::Dynamic> image_points;
	Eigen::Matrix<double, 3, Eigen::Dynamic> world_points;

};

}