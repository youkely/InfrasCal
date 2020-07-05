#ifndef COSTFUNCTIONFACTORY_H
#define COSTFUNCTIONFACTORY_H

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

#include "infrascal/camera_models/Camera.h"

namespace ceres
{
    class CostFunction;
}

namespace infrascal
{

enum
{
    CAMERA_INTRINSICS =         1 << 0,
    CAMERA_POSE =               1 << 1,
    POINT_3D =                  1 << 2,
    ODOMETRY_INTRINSICS =       1 << 3,
    ODOMETRY_3D_POSE =          1 << 4,
    ODOMETRY_6D_POSE =          1 << 5,
    CAMERA_ODOMETRY_TRANSFORM = 1 << 6,
    PRINCIPLE_TRANSLATION =     1 << 7
};

class CostFunctionFactory
{
public:
    CostFunctionFactory();

    static boost::shared_ptr<CostFunctionFactory> instance(void);

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Vector3d& observed_P,
                                              const Eigen::Vector2d& observed_p,
                                              int flags) const;

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Vector3d& observed_P,
                                              const Eigen::Vector2d& observed_p,
                                              const Eigen::Matrix2d& sqrtPrecisionMat,
                                              int flags) const;

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Vector2d& observed_p,
                                              int flags, bool optimize_cam_odo_z = true) const;

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Vector2d& observed_p,
                                              const Eigen::Matrix2d& sqrtPrecisionMat,
                                              int flags, bool optimize_cam_odo_z = true) const;

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Vector3d& odo_pos,
                                              const Eigen::Vector3d& odo_att,
                                              const Eigen::Vector2d& observed_p,
                                              int flags, bool optimize_cam_odo_z = true) const;

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Quaterniond& cam_odo_q,
                                              const Eigen::Vector3d& cam_odo_t,
                                              const Eigen::Vector3d& odo_pos,
                                              const Eigen::Vector3d& odo_att,
                                              const Eigen::Vector2d& observed_p,
                                              int flags) const;

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Quaterniond& cam_ref_q,
                                              const Eigen::Vector3d& cam_ref_t,
                                              const Eigen::Vector3d& odo_pos,
                                              const Eigen::Vector3d& odo_att,
                                              const Eigen::Vector2d& observed_p,
                                              int flags, bool optimize_cam_odo_z) const;

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Quaterniond& cam_ref_q,
                                              const Eigen::Vector3d& cam_ref_t,
                                              const Eigen::Vector2d& observed_p,
                                              int flags, bool optimize_cam_odo_z = true) const;

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Quaterniond& cam_ref_q,
                                              const Eigen::Vector3d& cam_ref_t,
                                              const Eigen::Vector2d& observed_p,
                                              const Eigen::Matrix2d& sqrtPrecisionMat,
                                              int flags, bool optimize_cam_odo_z = true) const;

    ceres::CostFunction* generateCostFunction(const CameraConstPtr& cameraLeft,
                                              const CameraConstPtr& cameraRight,
                                              const Eigen::Vector3d& observed_P,
                                              const Eigen::Vector2d& observed_p_left,
                                              const Eigen::Vector2d& observed_p_right) const;

    ceres::CostFunction* generateCostFunction(const std::vector<double> weights) const;

    ceres::CostFunction* pentagonCostFunction(const std::vector<double> weights, const int edge_id) const;

    ceres::CostFunction* baselineCostFunction(const double weight, const double baseline) const;

    ceres::CostFunction* baselineCostFunction(const double weight) const;

    ceres::CostFunction* baselineCostFunction2(const double weight) const;

    ceres::CostFunction* radialPoseCostFunction(const Eigen::Quaterniond& world_cam_q,
                                                const Eigen::Vector3d& world_cam_t) const;
    ceres::CostFunction* radialPoseCostFunction(const double cx, const double cy,
                                                const Eigen::Vector3d& observed_P,
                                                const Eigen::Vector2d& observed_p) const;
    ceres::CostFunction* radialPoseCostFunction(const Eigen::Vector3d& observed_P,
                                                const Eigen::Vector2d& observed_p) const;
    ceres::CostFunction* radialPoseCostFunction2(const double cx, const double cy,
                                                const Eigen::Vector3d& observed_P,
                                                const Eigen::Vector2d& observed_p) const;
    ceres::CostFunction* generateCostFunction(const CameraConstPtr& camera,
                                              const Eigen::Vector3d& observed_P,
                                              const Eigen::Vector2d& observed_p,
                                              const Eigen::Vector2d& cxcy) const;
private:
    static boost::shared_ptr<CostFunctionFactory> m_instance;
};

}

#endif
