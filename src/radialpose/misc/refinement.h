#pragma once
#include "camera.h"
#include <Eigen/Dense>

namespace radialpose {

	void refinement_dist(const Eigen::Matrix<double, 2, Eigen::Dynamic> &x, const Eigen::Matrix<double, 3, Eigen::Dynamic> &X, radialpose::Camera &p, int Np, int Nd);
	void refinement_undist(const Eigen::Matrix<double, 2, Eigen::Dynamic> &x, const Eigen::Matrix<double, 3, Eigen::Dynamic> &X, radialpose::Camera &p, int Np, int Nd);

}