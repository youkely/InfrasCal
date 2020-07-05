#pragma once

#include <Eigen/Dense>
#include <vector>
namespace radialpose {

	struct Camera {

		Camera() : focal(1.0) {}
		Camera(Eigen::Matrix3d rot, Eigen::Vector3d trans, double f) : R(rot), t(trans), focal(f) {};
		Camera(Eigen::Matrix3d rot, Eigen::Vector3d trans) : R(rot), t(trans), focal(1.0) {};

		Eigen::Matrix3d R;
		Eigen::Vector3d t;
		double focal;
		std::vector<double> dist_params;
	};
}