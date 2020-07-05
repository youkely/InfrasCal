#include "CostFunctionFactory.h"

#include "ceres/ceres.h"
#include "infrascal/camera_models/CataCamera.h"
#include "infrascal/camera_models/EquidistantCamera.h"
#include "infrascal/camera_models/PinholeCamera.h"
#include "infrascal/camera_models/ScaramuzzaCamera.h"

namespace infrascal
{



template<typename T>
void
inverseTransform(const T* const q, const T* const t,
                       T*  q_inv,  T* t_inv)
{
    T q_inv_[4] = {q[0], -q[1], -q[2], -q[3]};
    ceres::QuaternionRotatePoint(q_inv_, t, t_inv);
    t_inv[0] = -t_inv[0]; t_inv[1] = -t_inv[1]; t_inv[2] = -t_inv[2];
    q_inv[0] = q_inv_[0]; q_inv[1] = q_inv_[1]; q_inv[2] = q_inv_[2]; q_inv[3] = q_inv_[3];
}

template<typename T>
void
concatTransform(const T* const q1, const T* const t1,
                const T* const q2, const T* const t2,
                T* q12, T* t12)
{
    T q1_[4] = {q1[0], q1[1], q1[2], q1[3]};
    T q2_[4] = {q2[0], q2[1], q2[2], q2[3]};
    T q12_[4];
    ceres::QuaternionProduct(q1_, q2_, q12_);
    ceres::QuaternionRotatePoint(q1_, t2, t12);
    t12[0] += t1[0];    t12[1] += t1[1];    t12[2] += t1[2];
    q12[0] = q12_[0];    q12[1] = q12_[1];    q12[2] = q12_[2];    q12[3] = q12_[3];
}

template<typename T>
void
worldToCameraTransform(const T* const q_cam_odo, const T* const t_cam_odo,
                       const T* const p_odo, const T* const att_odo,
                       T* q, T* t, bool optimize_cam_odo_z = true)
{
    Eigen::Quaternion<T> q_z_inv(cos(att_odo[0] / T(2)), T(0), T(0), -sin(att_odo[0] / T(2)));
    Eigen::Quaternion<T> q_y_inv(cos(att_odo[1] / T(2)), T(0), -sin(att_odo[1] / T(2)), T(0));
    Eigen::Quaternion<T> q_x_inv(cos(att_odo[2] / T(2)), -sin(att_odo[2] / T(2)), T(0), T(0));

    Eigen::Quaternion<T> q_zyx_inv = q_x_inv * q_y_inv * q_z_inv;

    T q_odo[4] = {q_zyx_inv.w(), q_zyx_inv.x(), q_zyx_inv.y(), q_zyx_inv.z()};

    T q_odo_cam[4] = {q_cam_odo[3], -q_cam_odo[0], -q_cam_odo[1], -q_cam_odo[2]};

    T q0[4];
    ceres::QuaternionProduct(q_odo_cam, q_odo, q0);

    T t0[3];
    T t_odo[3] = {p_odo[0], p_odo[1], p_odo[2]};

    ceres::QuaternionRotatePoint(q_odo, t_odo, t0);

    t0[0] += t_cam_odo[0];
    t0[1] += t_cam_odo[1];

    if (optimize_cam_odo_z)
    {
        t0[2] += t_cam_odo[2];
    }

    ceres::QuaternionRotatePoint(q_odo_cam, t0, t);
    t[0] = -t[0]; t[1] = -t[1]; t[2] = -t[2];

    // Convert quaternion from Ceres convention (w, x, y, z)
    // to Eigen convention (x, y, z, w)
    // return cam_T_world
    q[0] = q0[1]; q[1] = q0[2]; q[2] = q0[3]; q[3] = q0[0];
}

template<typename T>
void
worldToCameraTransformFromLocal(const T* const q_camref_odo, const T* const t_camref_odo,
                                const T* const q_cam_camref, const T* const t_cam_camref,
                                const T* const p_odo, const T* const att_odo,
                                T* q, T* t, bool optimize_cam_odo_z = true)
{
    Eigen::Quaternion<T> q_z_inv(cos(att_odo[0] / T(2)), T(0), T(0), -sin(att_odo[0] / T(2)));
    Eigen::Quaternion<T> q_y_inv(cos(att_odo[1] / T(2)), T(0), -sin(att_odo[1] / T(2)), T(0));
    Eigen::Quaternion<T> q_x_inv(cos(att_odo[2] / T(2)), -sin(att_odo[2] / T(2)), T(0), T(0));

    Eigen::Quaternion<T> q_zyx_inv = q_x_inv * q_y_inv * q_z_inv;

    T q_odo[4] = {q_zyx_inv.w(), q_zyx_inv.x(), q_zyx_inv.y(), q_zyx_inv.z()};

    T q_camref_odo_[4] = {q_camref_odo[3], q_camref_odo[0], q_camref_odo[1], q_camref_odo[2]};

    T q_cam_camref_[4] = {q_cam_camref[3], q_cam_camref[0], q_cam_camref[1], q_cam_camref[2]};

    T q_cam_odo[4];
    ceres::QuaternionProduct(q_camref_odo_, q_cam_camref_, q_cam_odo);

    T t_cam_odo[3];
    T t_cam_camref_[3] = {t_cam_camref[0], t_cam_camref[1], t_cam_camref[2]};

    ceres::QuaternionRotatePoint(q_camref_odo_, t_cam_camref_, t_cam_odo);

    t_cam_odo[0] += t_camref_odo[0];
    t_cam_odo[1] += t_camref_odo[1];
    t_cam_odo[2] += t_camref_odo[2];

    T q_odo_cam[4] = {q_cam_odo[0], -q_cam_odo[1], -q_cam_odo[2], -q_cam_odo[3]};

    T q0[4];
    ceres::QuaternionProduct(q_odo_cam, q_odo, q0);

    T t0[3];
    T t_odo[3] = {p_odo[0], p_odo[1], p_odo[2]};

    ceres::QuaternionRotatePoint(q_odo, t_odo, t0);

    t0[0] += t_cam_odo[0];
    t0[1] += t_cam_odo[1];

    if (optimize_cam_odo_z)
    {
        t0[2] += t_cam_odo[2];
    }

    ceres::QuaternionRotatePoint(q_odo_cam, t0, t);
    t[0] = -t[0]; t[1] = -t[1]; t[2] = -t[2];

    // Convert quaternion from Ceres convention (w, x, y, z)
    // to Eigen convention (x, y, z, w)
    q[0] = q0[1]; q[1] = q0[2]; q[2] = q0[3]; q[3] = q0[0];
}

// given 3d points optimize others
template<class CameraT>
class ReprojectionError1
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionError1(const Eigen::Vector3d& observed_P,
                       const Eigen::Vector2d& observed_p)
     : m_observed_P(observed_P), m_observed_p(observed_p)
     , m_sqrtPrecisionMat(Eigen::Matrix2d::Identity()) {}

    ReprojectionError1(const Eigen::Vector3d& observed_P,
                       const Eigen::Vector2d& observed_p,
                       const Eigen::Matrix2d& sqrtPrecisionMat)
     : m_observed_P(observed_P), m_observed_p(observed_p)
     , m_sqrtPrecisionMat(sqrtPrecisionMat) {}

    ReprojectionError1(const std::vector<double>& intrinsic_params,
                       const Eigen::Vector3d& observed_P,
                       const Eigen::Vector2d& observed_p)
     : m_intrinsic_params(intrinsic_params)
     , m_observed_P(observed_P), m_observed_p(observed_p) {}

    // variables: camera intrinsics
    template <typename T>
    bool operator()(const T* const intrinsic_params,
                    T* residuals) const
    {
        Eigen::Matrix<T,3,1> P = m_observed_P.cast<T>();

        Eigen::Matrix<T,2,1> predicted_p;
        T q[4] = {T(0), T(0), T(0), T(1)};
        T t[3] = {T(0), T(0), T(0)};

        CameraT::spaceToPlane(intrinsic_params, q, t, P, predicted_p);

        Eigen::Matrix<T,2,1> e = predicted_p - m_observed_p.cast<T>();

        Eigen::Matrix<T,2,1> e_weighted = m_sqrtPrecisionMat.cast<T>() * e;

        residuals[0] = e_weighted(0);
        residuals[1] = e_weighted(1);

        return true;
    }

    // variables: camera intrinsics, t3
    template <typename T>
    bool operator()(const T* const intrinsic_params,
                    const T* const t3,
                    T* residuals) const
    {
        Eigen::Matrix<T,3,1> P = m_observed_P.cast<T>();

        Eigen::Matrix<T,2,1> predicted_p;
        T q[4] = {T(0), T(0), T(0), T(1)};
        T t[3] = {T(0), T(0), t3[0]};

        CameraT::spaceToPlane(intrinsic_params, q, t, P, predicted_p);

        Eigen::Matrix<T,2,1> e = predicted_p - m_observed_p.cast<T>();

        Eigen::Matrix<T,2,1> e_weighted = m_sqrtPrecisionMat.cast<T>() * e;

        residuals[0] = e_weighted(0);
        residuals[1] = e_weighted(1);

        return true;
    }

    // variables: camera intrinsics and camera extrinsics
    template <typename T>
    bool operator()(const T* const intrinsic_params,
                    const T* const q,
                    const T* const t,
                    T* residuals) const
    {
        Eigen::Matrix<T,3,1> P = m_observed_P.cast<T>();

        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params, q, t, P, predicted_p);

        Eigen::Matrix<T,2,1> e = predicted_p - m_observed_p.cast<T>();

        Eigen::Matrix<T,2,1> e_weighted = m_sqrtPrecisionMat.cast<T>() * e;

        residuals[0] = e_weighted(0);
        residuals[1] = e_weighted(1);

        return true;
    }

    // variables: camera-odometry transforms and odometry poses
    template <typename T>
    bool operator()(const T* const q_cam_odo, const T* const t_cam_odo,
                    const T* const p_odo, const T* const att_odo,
                    T* residuals) const
    {
        T q[4], t[3];
        worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q, t);

        Eigen::Matrix<T,3,1> P = m_observed_P.cast<T>();

        std::vector<T> intrinsic_params(m_intrinsic_params.begin(), m_intrinsic_params.end());

        // project 3D object point to the image plane
        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

        residuals[0] = predicted_p(0) - T(m_observed_p(0));
        residuals[1] = predicted_p(1) - T(m_observed_p(1));

        return true;
    }

    // variables: camera intrinsics, camera-odometry transforms and odometry poses
    template <typename T>
    bool operator()(const T* const intrinsic_params,
                    const T* const q_cam_odo, const T* const t_cam_odo,
                    const T* const p_odo, const T* const att_odo,
                    T* residuals) const
    {
        T q[4], t[3];
        worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q, t);

        Eigen::Matrix<T,3,1> P = m_observed_P.cast<T>();

        // project 3D object point to the image plane
        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params, q, t, P, predicted_p);

        residuals[0] = predicted_p(0) - T(m_observed_p(0));
        residuals[1] = predicted_p(1) - T(m_observed_p(1));

        return true;
    }

private:
    // camera intrinsics
    std::vector<double> m_intrinsic_params;

    // observed 3D point
    Eigen::Vector3d m_observed_P;

    // observed 2D point
    Eigen::Vector2d m_observed_p;

    // square root of precision matrix
    Eigen::Matrix2d m_sqrtPrecisionMat;
};

// variables: camera extrinsics, 3D point
template<class CameraT>
class ReprojectionError2
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionError2(const std::vector<double>& intrinsic_params,
                       const Eigen::Vector2d& observed_p)
     : m_intrinsic_params(intrinsic_params), m_observed_p(observed_p) {}

    template <typename T>
    bool operator()(const T* const q, const T* const t,
                    const T* const point, T* residuals) const
    {
        Eigen::Matrix<T,3,1> P;
        P(0) = T(point[0]);
        P(1) = T(point[1]);
        P(2) = T(point[2]);

        std::vector<T> intrinsic_params(m_intrinsic_params.begin(), m_intrinsic_params.end());

        // project 3D object point to the image plane
        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

        residuals[0] = predicted_p(0) - T(m_observed_p(0));
        residuals[1] = predicted_p(1) - T(m_observed_p(1));

        return true;
    }

private:
    // camera intrinsics
    std::vector<double> m_intrinsic_params;

    // observed 2D point
    Eigen::Vector2d m_observed_p;
};

// variable cam-odo, odom, 3d points, intrinsics
template<class CameraT>
class ReprojectionError3
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionError3(const Eigen::Vector2d& observed_p)
     : m_observed_p(observed_p)
     , m_sqrtPrecisionMat(Eigen::Matrix2d::Identity())
     , m_optimize_cam_odo_z(true) {}

    ReprojectionError3(const Eigen::Vector2d& observed_p,
                       const Eigen::Matrix2d& sqrtPrecisionMat)
     : m_observed_p(observed_p)
     , m_sqrtPrecisionMat(sqrtPrecisionMat)
     , m_optimize_cam_odo_z(true){}

    ReprojectionError3(const std::vector<double>& intrinsic_params,
                       const Eigen::Vector2d& observed_p)
     : m_intrinsic_params(intrinsic_params)
     , m_observed_p(observed_p)
     , m_sqrtPrecisionMat(Eigen::Matrix2d::Identity())
     , m_optimize_cam_odo_z(true) {}

    ReprojectionError3(const std::vector<double>& intrinsic_params,
                       const Eigen::Vector2d& observed_p,
                       const Eigen::Matrix2d& sqrtPrecisionMat)
     : m_intrinsic_params(intrinsic_params)
     , m_observed_p(observed_p)
     , m_sqrtPrecisionMat(sqrtPrecisionMat)
     , m_optimize_cam_odo_z(true) {}


    ReprojectionError3(const std::vector<double>& intrinsic_params,
                       const Eigen::Vector3d& odo_pos,
                       const Eigen::Vector3d& odo_att,
                       const Eigen::Vector2d& observed_p,
                       bool optimize_cam_odo_z)
     : m_intrinsic_params(intrinsic_params)
     , m_odo_pos(odo_pos), m_odo_att(odo_att)
     , m_observed_p(observed_p)
     , m_optimize_cam_odo_z(optimize_cam_odo_z) {}

    ReprojectionError3(const std::vector<double>& intrinsic_params,
                       const Eigen::Quaterniond& cam_odo_q,
                       const Eigen::Vector3d& cam_odo_t,
                       const Eigen::Vector3d& odo_pos,
                       const Eigen::Vector3d& odo_att,
                       const Eigen::Vector2d& observed_p)
     : m_intrinsic_params(intrinsic_params)
     , m_cam_odo_q(cam_odo_q), m_cam_odo_t(cam_odo_t)
     , m_odo_pos(odo_pos), m_odo_att(odo_att)
     , m_observed_p(observed_p)
     , m_optimize_cam_odo_z(true) {}

    // variables: camera intrinsics, camera-to-odometry transform,
    //            odometry extrinsics, 3D point
    template <typename T>
    bool operator()(const T* const intrinsic_params,
                    const T* const q_cam_odo, const T* const t_cam_odo,
                    const T* const p_odo, const T* const att_odo,
                    const T* const point, T* residuals) const
    {
        T q[4], t[3];
        worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q, t, m_optimize_cam_odo_z);

        Eigen::Matrix<T,3,1> P(point[0], point[1], point[2]);

        // project 3D object point to the image plane
        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params, q, t, P, predicted_p);

        Eigen::Matrix<T,2,1> err = predicted_p - m_observed_p.cast<T>();
        Eigen::Matrix<T,2,1> err_weighted = m_sqrtPrecisionMat.cast<T>() * err;

        residuals[0] = err_weighted(0);
        residuals[1] = err_weighted(1);

        return true;
    }

    // variables: camera-to-odometry transform, 3D point
    template <typename T>
    bool operator()(const T* const q_cam_odo, const T* const t_cam_odo,
                    const T* const point, T* residuals) const
    {
        T p_odo[3] = {T(m_odo_pos(0)), T(m_odo_pos(1)), T(m_odo_pos(2))};
        T att_odo[3] = {T(m_odo_att(0)), T(m_odo_att(1)), T(m_odo_att(2))};
        T q[4], t[3];

        worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q, t, m_optimize_cam_odo_z);

        std::vector<T> intrinsic_params(m_intrinsic_params.begin(), m_intrinsic_params.end());
        Eigen::Matrix<T,3,1> P(point[0], point[1], point[2]);

        // project 3D object point to the image plane
        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

        residuals[0] = predicted_p(0) - T(m_observed_p(0));
        residuals[1] = predicted_p(1) - T(m_observed_p(1));

        return true;
    }

    // variables: camera-to-odometry transform, odometry extrinsics, 3D point
    template <typename T>
    bool operator()(const T* const q_cam_odo, const T* const t_cam_odo,
                    const T* const p_odo, const T* const att_odo,
                    const T* const point, T* residuals) const
    {
        T q[4], t[3];
        worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q, t, m_optimize_cam_odo_z);

        std::vector<T> intrinsic_params(m_intrinsic_params.begin(), m_intrinsic_params.end());
        Eigen::Matrix<T,3,1> P(point[0], point[1], point[2]);

        // project 3D object point to the image plane
        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

        Eigen::Matrix<T,2,1> err = predicted_p - m_observed_p.cast<T>();
        Eigen::Matrix<T,2,1> err_weighted = m_sqrtPrecisionMat.cast<T>() * err;

        residuals[0] = err_weighted(0);
        residuals[1] = err_weighted(1);

        return true;
    }

    // variables: 3D point
    template <typename T>
    bool operator()(const T* const point, T* residuals) const
    {
        T q_cam_odo[4] = {T(m_cam_odo_q.coeffs()(0)), T(m_cam_odo_q.coeffs()(1)), T(m_cam_odo_q.coeffs()(2)), T(m_cam_odo_q.coeffs()(3))};
        T t_cam_odo[3] = {T(m_cam_odo_t(0)), T(m_cam_odo_t(1)), T(m_cam_odo_t(2))};
        T p_odo[3] = {T(m_odo_pos(0)), T(m_odo_pos(1)), T(m_odo_pos(2))};
        T att_odo[3] = {T(m_odo_att(0)), T(m_odo_att(1)), T(m_odo_att(2))};
        T q[4], t[3];

        worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q, t, m_optimize_cam_odo_z);

        std::vector<T> intrinsic_params(m_intrinsic_params.begin(), m_intrinsic_params.end());
        Eigen::Matrix<T,3,1> P(point[0], point[1], point[2]);

        // project 3D object point to the image plane
        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

        residuals[0] = predicted_p(0) - T(m_observed_p(0));
        residuals[1] = predicted_p(1) - T(m_observed_p(1));

        return true;
    }

private:
    // camera intrinsics
    std::vector<double> m_intrinsic_params;

    // observed camera-odometry transform
    Eigen::Quaterniond m_cam_odo_q;
    Eigen::Vector3d m_cam_odo_t;

    // observed odometry
    Eigen::Vector3d m_odo_pos;
    Eigen::Vector3d m_odo_att;

    // observed 2D point
    Eigen::Vector2d m_observed_p;

    Eigen::Matrix2d m_sqrtPrecisionMat;

    bool m_optimize_cam_odo_z;
};

// fix camera relative extrinsics
template<class CameraT>
class ReprojectionError4
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionError4(const std::vector<double>& intrinsic_params,
                       const Eigen::Quaterniond& cam_ref_q,
                       const Eigen::Vector3d& cam_ref_t,
                       const Eigen::Vector3d& odo_pos,
                       const Eigen::Vector3d& odo_att,
                       const Eigen::Vector2d& observed_p,
                       bool optimize_cam_odo_z)
            : m_intrinsic_params(intrinsic_params)
            , m_cam_ref_q(cam_ref_q), m_cam_ref_t(cam_ref_t)
            , m_odo_pos(odo_pos), m_odo_att(odo_att)
            , m_observed_p(observed_p)
            , m_optimize_cam_odo_z(optimize_cam_odo_z) {}

    ReprojectionError4(const std::vector<double>& intrinsic_params,
                       const Eigen::Quaterniond& cam_ref_q,
                       const Eigen::Vector3d& cam_ref_t,
                       const Eigen::Vector2d& observed_p,
                       bool optimize_cam_odo_z)
            : m_intrinsic_params(intrinsic_params)
            , m_cam_ref_q(cam_ref_q), m_cam_ref_t(cam_ref_t)
            , m_observed_p(observed_p)
            , m_sqrtPrecisionMat(Eigen::Matrix2d::Identity())
            , m_optimize_cam_odo_z(optimize_cam_odo_z) {}


    ReprojectionError4(const std::vector<double>& intrinsic_params,
                       const Eigen::Quaterniond& cam_ref_q,
                       const Eigen::Vector3d& cam_ref_t,
                       const Eigen::Vector2d& observed_p,
                       const Eigen::Matrix2d& sqrtPrecisionMat,
                       bool optimize_cam_odo_z)
            : m_intrinsic_params(intrinsic_params)
            , m_cam_ref_q(cam_ref_q), m_cam_ref_t(cam_ref_t)
            , m_observed_p(observed_p)
            , m_sqrtPrecisionMat(sqrtPrecisionMat)
            , m_optimize_cam_odo_z(optimize_cam_odo_z) {}

    // variables: camera-to-odometry transform, 3D point
    template <typename T>
    bool operator()(const T* const q_cam_odo, const T* const t_cam_odo,
                    const T* const point, T* residuals) const
    {
        T p_odo[3] = {T(m_odo_pos(0)), T(m_odo_pos(1)), T(m_odo_pos(2))};
        T att_odo[3] = {T(m_odo_att(0)), T(m_odo_att(1)), T(m_odo_att(2))};
        T q_cam_camref[4] = {T(m_cam_ref_q.coeffs()(0)), T(m_cam_ref_q.coeffs()(1)), T(m_cam_ref_q.coeffs()(2)), T(m_cam_ref_q.coeffs()(3))};
        T t_cam_camref[3] = {T(m_cam_ref_t(0)), T(m_cam_ref_t(1)), T(m_cam_ref_t(2))};

        T q[4], t[3];

        worldToCameraTransformFromLocal(q_cam_odo, t_cam_odo, q_cam_camref, t_cam_camref, p_odo, att_odo, q, t, m_optimize_cam_odo_z);

        std::vector<T> intrinsic_params(m_intrinsic_params.begin(), m_intrinsic_params.end());
        Eigen::Matrix<T,3,1> P(point[0], point[1], point[2]);

        // project 3D object point to the image plane
        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

        residuals[0] = predicted_p(0) - T(m_observed_p(0));
        residuals[1] = predicted_p(1) - T(m_observed_p(1));

        return true;
    }

    // variables: camera-to-odometry transform, odometry extrinsics, 3D point
    template <typename T>
    bool operator()(const T* const q_cam_odo, const T* const t_cam_odo,
                    const T* const p_odo, const T* const att_odo,
                    const T* const point, T* residuals) const
    {
        T q_cam_camref[4] = {T(m_cam_ref_q.coeffs()(0)), T(m_cam_ref_q.coeffs()(1)), T(m_cam_ref_q.coeffs()(2)), T(m_cam_ref_q.coeffs()(3))};
        T t_cam_camref[3] = {T(m_cam_ref_t(0)), T(m_cam_ref_t(1)), T(m_cam_ref_t(2))};

        T q[4], t[3];

        worldToCameraTransformFromLocal(q_cam_odo, t_cam_odo, q_cam_camref, t_cam_camref, p_odo, att_odo, q, t, m_optimize_cam_odo_z);


        std::vector<T> intrinsic_params(m_intrinsic_params.begin(), m_intrinsic_params.end());
        Eigen::Matrix<T,3,1> P(point[0], point[1], point[2]);

        // project 3D object point to the image plane
        Eigen::Matrix<T,2,1> predicted_p;
        CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

        Eigen::Matrix<T,2,1> err = predicted_p - m_observed_p.cast<T>();
        Eigen::Matrix<T,2,1> err_weighted = m_sqrtPrecisionMat.cast<T>() * err;

        residuals[0] = err_weighted(0);
        residuals[1] = err_weighted(1);

        return true;
    }


private:
    // camera intrinsics
    std::vector<double> m_intrinsic_params;

    // observed reference-camera transform
    Eigen::Quaterniond m_cam_ref_q;
    Eigen::Vector3d m_cam_ref_t;

    // observed odometry
    Eigen::Vector3d m_odo_pos;
    Eigen::Vector3d m_odo_att;

    // observed 2D point
    Eigen::Vector2d m_observed_p;

    Eigen::Matrix2d m_sqrtPrecisionMat;

    bool m_optimize_cam_odo_z;
};


// given 3d points and principal point optimize others
template<class CameraT>
class ReprojectionError5
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionError5(const Eigen::Vector3d& observed_P,
                       const Eigen::Vector2d& observed_p,
                       const Eigen::Vector2d& cxcy)
            : m_observed_P(observed_P), m_observed_p(observed_p)
            , m_sqrtPrecisionMat(Eigen::Matrix2d::Identity()),
            m_cxcy(cxcy){}


    // variables: camera intrinsics, t3
    template <typename T>
    bool operator()(const T* const intrinsic_params,
                    const T* const t3,
                    T* residuals) const
    {
        Eigen::Matrix<T,3,1> P = m_observed_P.cast<T>();

        Eigen::Matrix<T,2,1> predicted_p;
        T q[4] = {T(0), T(0), T(0), T(1)};
        T t[3] = {T(0), T(0), t3[0]};
        T intrinsic_params_applied[8] = {intrinsic_params[0], intrinsic_params[1], intrinsic_params[2],
                                         intrinsic_params[3], intrinsic_params[4], intrinsic_params[5],
                                         T(m_cxcy[0]), T(m_cxcy[1])};
        CameraT::spaceToPlane(intrinsic_params_applied, q, t, P, predicted_p);

        Eigen::Matrix<T,2,1> e = predicted_p - m_observed_p.cast<T>();

        Eigen::Matrix<T,2,1> e_weighted = m_sqrtPrecisionMat.cast<T>() * e;

        residuals[0] = e_weighted(0);
        residuals[1] = e_weighted(1);

        return true;
    }

private:
    // principal point
    Eigen::Vector2d m_cxcy;

    // observed 3D point
    Eigen::Vector3d m_observed_P;

    // observed 2D point
    Eigen::Vector2d m_observed_p;

    // square root of precision matrix
    Eigen::Matrix2d m_sqrtPrecisionMat;
};

// variables: camera intrinsics and camera extrinsics
template<class CameraT>
class StereoReprojectionError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoReprojectionError(const Eigen::Vector3d& observed_P,
                            const Eigen::Vector2d& observed_p_l,
                            const Eigen::Vector2d& observed_p_r)
     : m_observed_P(observed_P)
     , m_observed_p_l(observed_p_l)
     , m_observed_p_r(observed_p_r)
    {

    }

    template <typename T>
    bool operator()(const T* const intrinsic_params_l,
                    const T* const intrinsic_params_r,
                    const T* const q_l,
                    const T* const t_l,
                    const T* const q_l_r,
                    const T* const t_l_r,
                    T* residuals) const
    {
        Eigen::Matrix<T,3,1> P;
        P(0) = T(m_observed_P(0));
        P(1) = T(m_observed_P(1));
        P(2) = T(m_observed_P(2));

        Eigen::Matrix<T,2,1> predicted_p_l;
        CameraT::spaceToPlane(intrinsic_params_l, q_l, t_l, P, predicted_p_l);

        Eigen::Quaternion<T> q_r = Eigen::Quaternion<T>(q_l_r) * Eigen::Quaternion<T>(q_l);

        Eigen::Matrix<T,3,1> t_r;
        t_r(0) = t_l[0];
        t_r(1) = t_l[1];
        t_r(2) = t_l[2];

        t_r = Eigen::Quaternion<T>(q_l_r) * t_r;
        t_r(0) += t_l_r[0];
        t_r(1) += t_l_r[1];
        t_r(2) += t_l_r[2];

        Eigen::Matrix<T,2,1> predicted_p_r;
        CameraT::spaceToPlane(intrinsic_params_r, q_r.coeffs().data(), t_r.data(), P, predicted_p_r);

        residuals[0] = predicted_p_l(0) - T(m_observed_p_l(0));
        residuals[1] = predicted_p_l(1) - T(m_observed_p_l(1));
        residuals[2] = predicted_p_r(0) - T(m_observed_p_r(0));
        residuals[3] = predicted_p_r(1) - T(m_observed_p_r(1));

        return true;
    }

private:
    // observed 3D point
    Eigen::Vector3d m_observed_P;

    // observed 2D point
    Eigen::Vector2d m_observed_p_l;
    Eigen::Vector2d m_observed_p_r;
};

boost::shared_ptr<CostFunctionFactory> CostFunctionFactory::m_instance;

CostFunctionFactory::CostFunctionFactory()
{

}

boost::shared_ptr<CostFunctionFactory>
CostFunctionFactory::instance(void)
{
    if (m_instance.get() == 0)
    {
        m_instance.reset(new CostFunctionFactory);
    }

    return m_instance;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Vector3d& observed_P,
                                          const Eigen::Vector2d& observed_p,
                                          int flags) const
{
    ceres::CostFunction* costFunction = 0;

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);

    switch (flags)
    {

    case CAMERA_INTRINSICS:
        switch (camera->modelType())
        {
            case Camera::KANNALA_BRANDT:
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError1<EquidistantCamera>, 2, 8>(
                                new ReprojectionError1<EquidistantCamera>(observed_P, observed_p));
                break;
            case Camera::PINHOLE:
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 8>(
                                new ReprojectionError1<PinholeCamera>(observed_P, observed_p));
                break;
            case Camera::MEI:
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 9>(
                                new ReprojectionError1<CataCamera>(observed_P, observed_p));
                break;
            case Camera::SCARAMUZZA:
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS>(
                                new ReprojectionError1<OCAMCamera>(observed_P, observed_p));
                break;
        }
        break;
    case CAMERA_INTRINSICS | PRINCIPLE_TRANSLATION:
        switch (camera->modelType())
        {
            case Camera::KANNALA_BRANDT:
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError1<EquidistantCamera>, 2, 8, 1>(
                                new ReprojectionError1<EquidistantCamera>(observed_P, observed_p));
                break;
            case Camera::PINHOLE:
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 8, 1>(
                                new ReprojectionError1<PinholeCamera>(observed_P, observed_p));
                break;
            case Camera::MEI:
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 9, 1>(
                                new ReprojectionError1<CataCamera>(observed_P, observed_p));
                break;
            case Camera::SCARAMUZZA:
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 1>(
                                new ReprojectionError1<OCAMCamera>(observed_P, observed_p));
                break;
        }
        break;
    case CAMERA_INTRINSICS | CAMERA_POSE:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<EquidistantCamera>, 2, 8, 4, 3>(
                    new ReprojectionError1<EquidistantCamera>(observed_P, observed_p));
            break;
        case Camera::PINHOLE:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 8, 4, 3>(
                    new ReprojectionError1<PinholeCamera>(observed_P, observed_p));
            break;
        case Camera::MEI:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 9, 4, 3>(
                    new ReprojectionError1<CataCamera>(observed_P, observed_p));
            break;
        case Camera::SCARAMUZZA:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3>(
                    new ReprojectionError1<OCAMCamera>(observed_P, observed_p));
            break;
        }
        break;
    case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<EquidistantCamera>, 2, 4, 3, 3, 3>(
                    new ReprojectionError1<EquidistantCamera>(intrinsic_params, observed_P, observed_p));
            break;
        case Camera::PINHOLE:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 4, 3, 3, 3>(
                    new ReprojectionError1<PinholeCamera>(intrinsic_params, observed_P, observed_p));
            break;
        case Camera::MEI:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 4, 3, 3, 3>(
                    new ReprojectionError1<CataCamera>(intrinsic_params, observed_P, observed_p));
            break;
        case Camera::SCARAMUZZA:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2, 4, 3, 3, 3>(
                    new ReprojectionError1<OCAMCamera>(intrinsic_params, observed_P, observed_p));
            break;
        }
        break;
    case CAMERA_INTRINSICS | CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<EquidistantCamera>, 2, 8, 4, 3, 3, 3>(
                    new ReprojectionError1<EquidistantCamera>(observed_P, observed_p));
            break;
        case Camera::PINHOLE:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 8, 4, 3, 3, 3>(
                    new ReprojectionError1<PinholeCamera>(observed_P, observed_p));
            break;
        case Camera::MEI:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 9, 4, 3, 3, 3>(
                    new ReprojectionError1<CataCamera>(observed_P, observed_p));
            break;
        case Camera::SCARAMUZZA:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3, 3, 3>(
                    new ReprojectionError1<OCAMCamera>(observed_P, observed_p));
            break;
        }
        break;
    }

    return costFunction;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Vector3d& observed_P,
                                          const Eigen::Vector2d& observed_p,
                                          const Eigen::Matrix2d& sqrtPrecisionMat,
                                          int flags) const
{
    ceres::CostFunction* costFunction = 0;

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);

    switch (flags)
    {
    case CAMERA_INTRINSICS | CAMERA_POSE:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<EquidistantCamera>, 2, 8, 4, 3>(
                    new ReprojectionError1<EquidistantCamera>(observed_P, observed_p, sqrtPrecisionMat));
            break;
        case Camera::PINHOLE:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 8, 4, 3>(
                    new ReprojectionError1<PinholeCamera>(observed_P, observed_p, sqrtPrecisionMat));
            break;
        case Camera::MEI:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 9, 4, 3>(
                    new ReprojectionError1<CataCamera>(observed_P, observed_p, sqrtPrecisionMat));
            break;
        case Camera::SCARAMUZZA:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3>(
                    new ReprojectionError1<OCAMCamera>(observed_P, observed_p, sqrtPrecisionMat));
            break;
        }
        break;
    }

    return costFunction;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Vector2d& observed_p,
                                          int flags, bool optimize_cam_odo_z) const
{
    ceres::CostFunction* costFunction = 0;

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);

    switch (flags)
    {
    case CAMERA_POSE | POINT_3D:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError2<EquidistantCamera>, 2, 4, 3, 3>(
                    new ReprojectionError2<EquidistantCamera>(intrinsic_params, observed_p));
            break;
        case Camera::PINHOLE:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError2<PinholeCamera>, 2, 4, 3, 3>(
                    new ReprojectionError2<PinholeCamera>(intrinsic_params, observed_p));
            break;
        case Camera::MEI:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError2<CataCamera>, 2, 4, 3, 3>(
                    new ReprojectionError2<CataCamera>(intrinsic_params, observed_p));
            break;
        case Camera::SCARAMUZZA:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError2<OCAMCamera>, 2, 4, 3, 3>(
                    new ReprojectionError2<OCAMCamera>(intrinsic_params, observed_p));
            break;
        }
        break;
    case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_3D_POSE | POINT_3D:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 3, 2, 1, 3>(
                        new ReprojectionError3<EquidistantCamera>(intrinsic_params, observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 2, 1, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, observed_p));
            }
            break;
        case Camera::PINHOLE:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 3, 2, 1, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 2, 1, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, observed_p));
            }
            break;
        case Camera::MEI:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 3, 2, 1, 3>(
                        new ReprojectionError3<CataCamera>(intrinsic_params, observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 2, 2, 1, 3>(
                        new ReprojectionError3<CataCamera>(intrinsic_params, observed_p));
            }
            break;
        case Camera::SCARAMUZZA:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 3, 2, 1, 3>(
                        new ReprojectionError3<OCAMCamera>(intrinsic_params, observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 2, 2, 1, 3>(
                        new ReprojectionError3<OCAMCamera>(intrinsic_params, observed_p));
            }
            break;
        }
        break;
    case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<EquidistantCamera>(intrinsic_params, observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, observed_p));
            }
            break;
        case Camera::PINHOLE:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, observed_p));
            }
            break;
        case Camera::MEI:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<CataCamera>(intrinsic_params, observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<CataCamera>(intrinsic_params, observed_p));
            }
            break;
        case Camera::SCARAMUZZA:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<OCAMCamera>(intrinsic_params, observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<OCAMCamera>(intrinsic_params, observed_p));
            }
            break;
        }
        break;
    case CAMERA_INTRINSICS | CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_3D_POSE | POINT_3D:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 8, 4, 3, 2, 1, 3>(
                        new ReprojectionError3<EquidistantCamera>(observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 2, 1, 3>(
                        new ReprojectionError3<PinholeCamera>(observed_p));
            }
            break;
        case Camera::PINHOLE:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 3, 2, 1, 3>(
                        new ReprojectionError3<PinholeCamera>(observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 2, 1, 3>(
                        new ReprojectionError3<PinholeCamera>(observed_p));
            }
            break;
        case Camera::MEI:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 3, 2, 1, 3>(
                        new ReprojectionError3<CataCamera>(observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 2, 2, 1, 3>(
                        new ReprojectionError3<CataCamera>(observed_p));
            }
            break;
        case Camera::SCARAMUZZA:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3, 2, 1, 3>(
                        new ReprojectionError3<OCAMCamera>(observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 2, 2, 1, 3>(
                        new ReprojectionError3<OCAMCamera>(observed_p));
            }
            break;
        }
        break;
    case CAMERA_INTRINSICS | CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 8, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<EquidistantCamera>(observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(observed_p));
            }
            break;
        case Camera::PINHOLE:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(observed_p));
            }
            break;
        case Camera::MEI:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<CataCamera>(observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<CataCamera>(observed_p));
            }
            break;
        case Camera::SCARAMUZZA:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<OCAMCamera>(observed_p));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<OCAMCamera>(observed_p));
            }
            break;
        }
        break;
    }

    return costFunction;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Vector2d& observed_p,
                                          const Eigen::Matrix2d& sqrtPrecisionMat,
                                          int flags, bool optimize_cam_odo_z) const
{
    ceres::CostFunction* costFunction = 0;

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);

    switch (flags)
    {
    case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<EquidistantCamera>(intrinsic_params, observed_p, sqrtPrecisionMat));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, observed_p, sqrtPrecisionMat));
            }
            break;
        case Camera::PINHOLE:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, observed_p, sqrtPrecisionMat));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, observed_p, sqrtPrecisionMat));
            }
            break;
        case Camera::MEI:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<CataCamera>(intrinsic_params, observed_p, sqrtPrecisionMat));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<CataCamera>(intrinsic_params, observed_p, sqrtPrecisionMat));
            }
            break;
        case Camera::SCARAMUZZA:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<OCAMCamera>(intrinsic_params, observed_p, sqrtPrecisionMat));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<OCAMCamera>(intrinsic_params, observed_p, sqrtPrecisionMat));
            }
            break;
        }
        break;
    case CAMERA_INTRINSICS | CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 8, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<EquidistantCamera>(observed_p, sqrtPrecisionMat));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(observed_p, sqrtPrecisionMat));
            }
            break;
        case Camera::PINHOLE:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(observed_p, sqrtPrecisionMat));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(observed_p, sqrtPrecisionMat));
            }
            break;
        case Camera::MEI:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<CataCamera>(observed_p, sqrtPrecisionMat));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<CataCamera>(observed_p, sqrtPrecisionMat));
            }
            break;
        case Camera::SCARAMUZZA:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3, 3, 3, 3>(
                        new ReprojectionError3<OCAMCamera>(observed_p, sqrtPrecisionMat));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 2, 3, 3, 3>(
                        new ReprojectionError3<OCAMCamera>(observed_p, sqrtPrecisionMat));
            }
            break;
        }
        break;
    }

    return costFunction;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Vector3d& odo_pos,
                                          const Eigen::Vector3d& odo_att,
                                          const Eigen::Vector2d& observed_p,
                                          int flags, bool optimize_cam_odo_z) const
{
    ceres::CostFunction* costFunction = 0;

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);

    switch (flags)
    {
    case CAMERA_ODOMETRY_TRANSFORM | POINT_3D:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 3, 3>(
                        new ReprojectionError3<EquidistantCamera>(intrinsic_params, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 2, 3>(
                        new ReprojectionError3<EquidistantCamera>(intrinsic_params, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
            }
            break;
        case Camera::PINHOLE:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 3, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3>(
                        new ReprojectionError3<PinholeCamera>(intrinsic_params, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
            }
            break;
        case Camera::MEI:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 3, 3>(
                        new ReprojectionError3<CataCamera>(intrinsic_params, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 2, 3>(
                        new ReprojectionError3<CataCamera>(intrinsic_params, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
            }
            break;
        case Camera::SCARAMUZZA:
            if (optimize_cam_odo_z)
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 3, 3>(
                        new ReprojectionError3<OCAMCamera>(intrinsic_params, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 2, 3>(
                        new ReprojectionError3<OCAMCamera>(intrinsic_params, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
            }
            break;
        }
        break;
    }

    return costFunction;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Quaterniond& cam_odo_q,
                                          const Eigen::Vector3d& cam_odo_t,
                                          const Eigen::Vector3d& odo_pos,
                                          const Eigen::Vector3d& odo_att,
                                          const Eigen::Vector2d& observed_p,
                                          int flags) const
{
    ceres::CostFunction* costFunction = 0;

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);

    switch (flags)
    {
    case POINT_3D:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 3>(
                    new ReprojectionError3<EquidistantCamera>(intrinsic_params, cam_odo_q, cam_odo_t, odo_pos, odo_att, observed_p));
            break;
        case Camera::PINHOLE:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 3>(
                    new ReprojectionError3<PinholeCamera>(intrinsic_params, cam_odo_q, cam_odo_t, odo_pos, odo_att, observed_p));
            break;
        case Camera::MEI:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 3>(
                    new ReprojectionError3<CataCamera>(intrinsic_params, cam_odo_q, cam_odo_t, odo_pos, odo_att, observed_p));
            break;
        case Camera::SCARAMUZZA:
            costFunction =
                new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 3>(
                    new ReprojectionError3<OCAMCamera>(intrinsic_params, cam_odo_q, cam_odo_t, odo_pos, odo_att, observed_p));
            break;
        }
        break;
    }

    return costFunction;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Quaterniond& cam_ref_q,
                                          const Eigen::Vector3d& cam_ref_t,
                                          const Eigen::Vector3d& odo_pos,
                                          const Eigen::Vector3d& odo_att,
                                          const Eigen::Vector2d& observed_p,
                                          int flags, bool optimize_cam_odo_z) const
{
    ceres::CostFunction* costFunction = 0;

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);

    switch (flags)
    {
    case CAMERA_ODOMETRY_TRANSFORM | POINT_3D:
        switch (camera->modelType())
        {
            case Camera::KANNALA_BRANDT:
                if (optimize_cam_odo_z)
                {
                    costFunction =
                            new ceres::AutoDiffCostFunction<ReprojectionError4<EquidistantCamera>, 2, 4, 3, 3>(
                                    new ReprojectionError4<EquidistantCamera>(intrinsic_params, cam_ref_q, cam_ref_t, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
                }
                else
                {
                    costFunction =
                            new ceres::AutoDiffCostFunction<ReprojectionError4<EquidistantCamera>, 2, 4, 2, 3>(
                                    new ReprojectionError4<EquidistantCamera>(intrinsic_params, cam_ref_q, cam_ref_t, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
                }
                break;
            case Camera::PINHOLE:
                if (optimize_cam_odo_z)
                {
                    costFunction =
                            new ceres::AutoDiffCostFunction<ReprojectionError4<PinholeCamera>, 2, 4, 3, 3>(
                                    new ReprojectionError4<PinholeCamera>(intrinsic_params, cam_ref_q, cam_ref_t, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
                }
                else
                {
                    costFunction =
                            new ceres::AutoDiffCostFunction<ReprojectionError4<PinholeCamera>, 2, 4, 2, 3>(
                                    new ReprojectionError4<PinholeCamera>(intrinsic_params, cam_ref_q, cam_ref_t, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
                }
                break;
            case Camera::MEI:
                if (optimize_cam_odo_z)
                {
                    costFunction =
                            new ceres::AutoDiffCostFunction<ReprojectionError4<CataCamera>, 2, 4, 3, 3>(
                                    new ReprojectionError4<CataCamera>(intrinsic_params, cam_ref_q, cam_ref_t, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
                }
                else
                {
                    costFunction =
                            new ceres::AutoDiffCostFunction<ReprojectionError4<CataCamera>, 2, 4, 2, 3>(
                                    new ReprojectionError4<CataCamera>(intrinsic_params, cam_ref_q, cam_ref_t, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
                }
                break;
            case Camera::SCARAMUZZA:
                if (optimize_cam_odo_z)
                {
                    costFunction =
                            new ceres::AutoDiffCostFunction<ReprojectionError4<OCAMCamera>, 2, 4, 3, 3>(
                                    new ReprojectionError4<OCAMCamera>(intrinsic_params, cam_ref_q, cam_ref_t, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
                }
                else
                {
                    costFunction =
                            new ceres::AutoDiffCostFunction<ReprojectionError4<OCAMCamera>, 2, 4, 2, 3>(
                                    new ReprojectionError4<OCAMCamera>(intrinsic_params, cam_ref_q, cam_ref_t, odo_pos, odo_att, observed_p, optimize_cam_odo_z));
                }
                break;
        }
        break;
    }

    return costFunction;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Quaterniond& cam_ref_q,
                                          const Eigen::Vector3d& cam_ref_t,
                                          const Eigen::Vector2d& observed_p,
                                          int flags, bool optimize_cam_odo_z) const
{
    ceres::CostFunction* costFunction = 0;

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);

    switch (flags)
    {
    case CAMERA_ODOMETRY_TRANSFORM | POINT_3D | ODOMETRY_6D_POSE:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            if (optimize_cam_odo_z)
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<EquidistantCamera>, 2, 4, 3, 3, 3, 3>(
                                new ReprojectionError4<EquidistantCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<EquidistantCamera>, 2, 4, 2, 3, 3, 3>(
                                new ReprojectionError4<EquidistantCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, optimize_cam_odo_z));
            }
            break;
        case Camera::PINHOLE:
            if (optimize_cam_odo_z)
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<PinholeCamera>, 2, 4, 3, 3, 3, 3>(
                                new ReprojectionError4<PinholeCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
                                new ReprojectionError4<PinholeCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, optimize_cam_odo_z));
            }
            break;
        case Camera::MEI:
            if (optimize_cam_odo_z)
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<CataCamera>, 2, 4, 3, 3, 3, 3>(
                                new ReprojectionError4<CataCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<CataCamera>, 2, 4, 2, 3, 3, 3>(
                                new ReprojectionError4<CataCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, optimize_cam_odo_z));
            }
            break;
        case Camera::SCARAMUZZA:
            if (optimize_cam_odo_z)
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<OCAMCamera>, 2, 4, 3, 3, 3, 3>(
                                new ReprojectionError4<OCAMCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<OCAMCamera>, 2, 4, 2, 3, 3, 3>(
                                new ReprojectionError4<OCAMCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, optimize_cam_odo_z));
            }
            break;
        }
        break;
    }

    return costFunction;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Quaterniond& cam_ref_q,
                                          const Eigen::Vector3d& cam_ref_t,
                                          const Eigen::Vector2d& observed_p,
                                          const Eigen::Matrix2d& sqrtPrecisionMat,
                                          int flags, bool optimize_cam_odo_z) const
{
    ceres::CostFunction* costFunction = 0;

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);

    switch (flags)
    {
    case CAMERA_ODOMETRY_TRANSFORM | POINT_3D | ODOMETRY_6D_POSE:
        switch (camera->modelType())
        {
        case Camera::KANNALA_BRANDT:
            if (optimize_cam_odo_z)
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<EquidistantCamera>, 2, 4, 3, 3, 3, 3>(
                                new ReprojectionError4<EquidistantCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, sqrtPrecisionMat, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<EquidistantCamera>, 2, 4, 2, 3, 3, 3>(
                                new ReprojectionError4<EquidistantCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, sqrtPrecisionMat, optimize_cam_odo_z));
            }
            break;
        case Camera::PINHOLE:
            if (optimize_cam_odo_z)
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<PinholeCamera>, 2, 4, 3, 3, 3, 3>(
                                new ReprojectionError4<PinholeCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, sqrtPrecisionMat, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
                                new ReprojectionError4<PinholeCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, sqrtPrecisionMat, optimize_cam_odo_z));
            }
            break;
        case Camera::MEI:
            if (optimize_cam_odo_z)
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<CataCamera>, 2, 4, 3, 3, 3, 3>(
                                new ReprojectionError4<CataCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, sqrtPrecisionMat, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<CataCamera>, 2, 4, 2, 3, 3, 3>(
                                new ReprojectionError4<CataCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, sqrtPrecisionMat, optimize_cam_odo_z));
            }
            break;
        case Camera::SCARAMUZZA:
            if (optimize_cam_odo_z)
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<OCAMCamera>, 2, 4, 3, 3, 3, 3>(
                                new ReprojectionError4<OCAMCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, sqrtPrecisionMat, optimize_cam_odo_z));
            }
            else
            {
                costFunction =
                        new ceres::AutoDiffCostFunction<ReprojectionError4<OCAMCamera>, 2, 4, 2, 3, 3, 3>(
                                new ReprojectionError4<OCAMCamera>(intrinsic_params, cam_ref_q, cam_ref_t, observed_p, sqrtPrecisionMat, optimize_cam_odo_z));
            }
            break;
        }
        break;
    }

    return costFunction;
}




ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& cameraL,
                                          const CameraConstPtr& cameraR,
                                          const Eigen::Vector3d& observed_P,
                                          const Eigen::Vector2d& observed_p_l,
                                          const Eigen::Vector2d& observed_p_r) const
{
    ceres::CostFunction* costFunction = 0;

    if (cameraL->modelType() != cameraR->modelType())
    {
        return costFunction;
    }

    switch (cameraL->modelType())
    {
    case Camera::KANNALA_BRANDT:
        costFunction =
            new ceres::AutoDiffCostFunction<StereoReprojectionError<EquidistantCamera>, 4, 8, 8, 4, 3, 4, 3>(
                new StereoReprojectionError<EquidistantCamera>(observed_P, observed_p_l, observed_p_r));
        break;
    case Camera::PINHOLE:
        costFunction =
            new ceres::AutoDiffCostFunction<StereoReprojectionError<PinholeCamera>, 4, 8, 8, 4, 3, 4, 3>(
                new StereoReprojectionError<PinholeCamera>(observed_P, observed_p_l, observed_p_r));
        break;
    case Camera::MEI:
        costFunction =
            new ceres::AutoDiffCostFunction<StereoReprojectionError<CataCamera>, 4, 9, 9, 4, 3, 4, 3>(
                new StereoReprojectionError<CataCamera>(observed_P, observed_p_l, observed_p_r));
        break;
    case Camera::SCARAMUZZA:
        costFunction =
            new ceres::AutoDiffCostFunction<StereoReprojectionError<OCAMCamera>, 4, SCARAMUZZA_CAMERA_NUM_PARAMS, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3, 4, 3>(
                new StereoReprojectionError<OCAMCamera>(observed_P, observed_p_l, observed_p_r));
        break;
    }

    return costFunction;
}

template<class CameraT>
class IntrinsicDiffError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IntrinsicDiffError(const std::vector<double>& weights)
            : m_weights(weights) {}

    template <typename T>
    bool operator()(const T* const intrinsic_params, const T* const intrinsic_params_ref, T* residuals) const
    {
        for(int i = 0; i < 8; i++){
            residuals[i] = (intrinsic_params[i] - intrinsic_params_ref[i])
                    * m_weights[i];
        }
        return true;
    }

private:
    std::vector<double> m_weights;

};


class PentagonExtrinsicsDiffError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PentagonExtrinsicsDiffError(const std::vector<double>& weights, const int& edge_id)
            : m_weights(weights), m_edge_id(edge_id) {}

    template <typename T>
    bool operator()(const T* const q_cam_ref, const T* const t_cam_ref,
                    const T* const q_rig_ref, const T* const t_rig_ref,
                    const T* const linear_offsets,
                    const T* const rig_scale,
                    T* residuals) const
    {

        // calculate t_cam_rig, q_cam_rig from estimate
        T q_ref_rig[4] = {q_rig_ref[3], -q_rig_ref[0], -q_rig_ref[1], -q_rig_ref[2]};
        T t_rig_ref_[3] = {t_rig_ref[0], t_rig_ref[1], t_rig_ref[2]};
        T t_ref_rig[3];
        ceres::QuaternionRotatePoint(q_ref_rig, t_rig_ref_, t_ref_rig);
        t_ref_rig[0] = -t_ref_rig[0]; t_ref_rig[1] = -t_ref_rig[1]; t_ref_rig[2] = -t_ref_rig[2];

        T q_cam_ref_[4] = {q_cam_ref[3], q_cam_ref[0], q_cam_ref[1], q_cam_ref[2]};
        T t_cam_ref_[3] = {t_cam_ref[0], t_cam_ref[1], t_cam_ref[2]};

        T t_cam_rig[3];
        ceres::QuaternionRotatePoint(q_ref_rig, t_cam_ref_, t_cam_rig);
        t_cam_rig[0] += t_ref_rig[0];
        t_cam_rig[1] += t_ref_rig[1];
        t_cam_rig[2] += t_ref_rig[2];

        T q_cam_rig[4];
        ceres::QuaternionProduct(q_ref_rig, q_cam_ref_, q_cam_rig);


        // calculate t_cam_rig, q_cam_rig from model

        T t_rig_edge[3] = {T(0), T(0), -rig_scale[0]};

        T theta = T(m_edge_id * 2 * M_PI / 5.0);
        Eigen::Quaternion<T> q_y(cos(theta / T(2)), T(0), -sin(theta / T(2)), T(0));
        T q_rig_edge[4] = {q_y.w(), q_y.x(), q_y.y(), q_y.z()};
        T q_edge_rig[4] = {q_rig_edge[0], -q_rig_edge[1], -q_rig_edge[2], -q_rig_edge[3]};
        T t_cam_edge[3] = {linear_offsets[0], T(0), T(0)};
//        T t_cam_edge[3] = {T(0), T(0), T(0)};
        T q_cam_edge[4] = {T(1), T(0), T(0), T(0)};
        T t_edge_rig[3];
        ceres::QuaternionRotatePoint(q_edge_rig, t_rig_edge, t_edge_rig);
        t_edge_rig[0] = -t_edge_rig[0]; t_edge_rig[1] = -t_edge_rig[1];t_edge_rig[2] = -t_edge_rig[2];
        T t_cam_rig_model[3];
        ceres::QuaternionRotatePoint(q_edge_rig, t_cam_edge, t_cam_rig_model);
        t_cam_rig_model[0] += t_edge_rig[0];
        t_cam_rig_model[1] += t_edge_rig[1];
        t_cam_rig_model[2] += t_edge_rig[2];

        T q_cam_rig_model[4];
        ceres::QuaternionProduct(q_edge_rig, q_cam_edge, q_cam_rig_model);


        T angleAxis[3];
        ceres::QuaternionToAngleAxis(q_cam_rig, angleAxis);
        T angleAxis_model[3];
        ceres::QuaternionToAngleAxis(q_cam_rig_model, angleAxis_model);

        residuals[0] = (t_cam_rig[0] - t_cam_rig_model[0]) * m_weights[0];
        residuals[1] = (t_cam_rig[1] - t_cam_rig_model[1]) * m_weights[0];
        residuals[2] = (t_cam_rig[2] - t_cam_rig_model[2]) * m_weights[0];
        residuals[3] = (angleAxis[0] - angleAxis_model[0]) * m_weights[1];
        residuals[4] = (angleAxis[1] - angleAxis_model[1]) * m_weights[1];
        residuals[5] = (angleAxis[2] - angleAxis_model[2]) * m_weights[1];
        return true;
    }

private:
    std::vector<double> m_weights;
    int m_edge_id;

};

class BaselineDiffError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BaselineDiffError(const double& weight)
            : m_weight(weight) {}

    template <typename T>
    bool operator()(const T* const q_cam_ref_1, const T* const t_cam_ref_1,
                    const T* const q_cam_ref_2, const T* const t_cam_ref_2,
                    const T* const baseline,
                    T* residuals) const
    {
        T q_cam1_ref_[4] = {q_cam_ref_1[3], q_cam_ref_1[0], q_cam_ref_1[1], q_cam_ref_1[2]};
        T t_cam1_ref_[3] = {t_cam_ref_1[0], t_cam_ref_1[1], t_cam_ref_1[2]};
        T q_cam2_ref_[4] = {q_cam_ref_2[3], q_cam_ref_2[0], q_cam_ref_2[1], q_cam_ref_2[2]};
        T t_cam2_ref_[3] = {t_cam_ref_2[0], t_cam_ref_2[1], t_cam_ref_2[2]};

        T q_ref_cam2[4] = {q_cam2_ref_[0], q_cam2_ref_[1], q_cam2_ref_[2], q_cam2_ref_[3]};
        T t_ref_cam2[3];
        ceres::QuaternionRotatePoint(q_ref_cam2, t_cam2_ref_, t_ref_cam2);
        t_ref_cam2[0] = -t_ref_cam2[0]; t_ref_cam2[1] = -t_ref_cam2[1]; t_ref_cam2[2] = -t_ref_cam2[2];

        T t_cam1_cam2[3];
        ceres::QuaternionRotatePoint(q_ref_cam2, t_cam1_ref_, t_cam1_cam2);
        t_cam1_cam2[0] += t_ref_cam2[0];
        t_cam1_cam2[1] += t_ref_cam2[1];
        t_cam1_cam2[2] += t_ref_cam2[2];

        residuals[0] = (t_cam1_cam2[0]- baseline[0])* m_weight;
        residuals[1] = (t_cam1_cam2[1]- T(0))* m_weight;
        residuals[2] = (t_cam1_cam2[2]- T(0))* m_weight;

        return true;
    }

private:
    double m_weight;

};


class BaselineDiffError2
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BaselineDiffError2(const double& weight, const double& baseline)
            : m_weight(weight), m_baseline(baseline) {}

    template <typename T>
    bool operator()(const T* const linear_offsets1, const T* const linear_offsets2, T* residuals) const
    {
        residuals[0] = (linear_offsets2[0] - linear_offsets1[0] - m_baseline)* m_weight;

        return true;
    }

private:
    double m_weight;
    double m_baseline;

};

class BaselineDiffError3
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BaselineDiffError3(const double& weight)
            : m_weight(weight) {}

    template <typename T>
    bool operator()(const T* const linear_offsets0,
                    const T* const linear_offsets1,
                    const T* const linear_offsets2,
                    const T* const linear_offsets3,
                    const T* const linear_offsets4,
                    const T* const linear_offsets5,
                    const T* const linear_offsets6,
                    const T* const linear_offsets7,
                    const T* const linear_offsets8,
                    const T* const linear_offsets9,
                    T* residuals) const
    {
        T baselines[5];
        baselines[0] = linear_offsets0[0]-linear_offsets1[0];
        baselines[1] = linear_offsets2[0]-linear_offsets3[0];
        baselines[2] = linear_offsets4[0]-linear_offsets5[0];
        baselines[3] = linear_offsets6[0]-linear_offsets7[0];
        baselines[4] = linear_offsets8[0]-linear_offsets9[0];

        residuals[0] = (baselines[0] - baselines[1]) * m_weight;
        residuals[1] = (baselines[1] - baselines[2]) * m_weight;
        residuals[2] = (baselines[2] - baselines[3]) * m_weight;
        residuals[3] = (baselines[3] - baselines[4]) * m_weight;
        residuals[4] = (baselines[4] - baselines[0]) * m_weight;

        return true;
    }

private:
    double m_weight;

};

class RadialPoseDiffError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RadialPoseDiffError(const Eigen::Quaterniond& world_cam_q,
                        const Eigen::Vector3d& world_cam_t)
                        : m_world_cam_q(world_cam_q),
                          m_world_cam_t(world_cam_t){}

    template <typename T>
    bool operator()(const T* const q_cam_ref, const T* const t_cam_ref,
                    const T* const p_ref, const T* const att_ref,
                    T* residuals) const
    {
        T q_cam_ref_[4] = {q_cam_ref[3], q_cam_ref[0], q_cam_ref[1], q_cam_ref[2]};
        T q_ref_cam[4];
        T t_ref_cam[3];
        inverseTransform(q_cam_ref_, t_cam_ref, q_ref_cam, t_ref_cam);

        // prepare T_world_cam
        T q_world_cam[4] = {T(m_world_cam_q.w()), T(m_world_cam_q.x()),
                          T(m_world_cam_q.y()), T(m_world_cam_q.z())};
        T t_world_cam[3] = {T(m_world_cam_t(0)), T(m_world_cam_t(1)), T(m_world_cam_t(2))};

        // prepare T_ref_world
        Eigen::Quaternion<T> q_z_inv(cos(att_ref[0] / T(2)), T(0), T(0), -sin(att_ref[0] / T(2)));
        Eigen::Quaternion<T> q_y_inv(cos(att_ref[1] / T(2)), T(0), -sin(att_ref[1] / T(2)), T(0));
        Eigen::Quaternion<T> q_x_inv(cos(att_ref[2] / T(2)), -sin(att_ref[2] / T(2)), T(0), T(0));
        Eigen::Quaternion<T> q_zyx_inv = q_x_inv * q_y_inv * q_z_inv;

        T q_ref_world[4] = {q_zyx_inv.w(), -q_zyx_inv.x(), -q_zyx_inv.y(), -q_zyx_inv.z()};
        T t_ref_world[3] = {p_ref[0], p_ref[1], p_ref[2]};

        // prepare T_cam_ref_proj
        T q_ref_cam_proj[4];
        T t_ref_cam_proj[3];
        concatTransform(q_world_cam, t_world_cam, q_ref_world, t_ref_world, q_ref_cam_proj, t_ref_cam_proj);

        // error is ||(T_cam_ref - T_cam_ref_proj)[0:2,:]||
        T R[3];
        ceres::QuaternionToAngleAxis(q_ref_cam, R);
        T R_proj[3];
        ceres::QuaternionToAngleAxis(q_ref_cam_proj, R_proj);

        residuals[0] = (R[0] - R_proj[0]);
        residuals[1] = (R[1] - R_proj[1]);
        residuals[2] = (R[2] - R_proj[2]);
        residuals[3] = (t_ref_cam[0] - t_ref_cam_proj[0])*0.5;
        residuals[4] = (t_ref_cam[1] - t_ref_cam_proj[1])*0.5;
        return true;
    }

private:

    Eigen::Quaterniond m_world_cam_q;
    Eigen::Vector3d m_world_cam_t;
};

class RadialPoseDiffError2
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    RadialPoseDiffError2(const double cx,
                        const double cy,
                        const Eigen::Vector3d& observed_P,
                        const Eigen::Vector2d& observed_p)
            :  m_cx(cx), m_cy(cy),
            m_observed_P(observed_P), m_observed_p(observed_p){}

    RadialPoseDiffError2(const Eigen::Vector3d& observed_P,
                         const Eigen::Vector2d& observed_p)
            :  m_observed_P(observed_P), m_observed_p(observed_p){}

    // variables: camera-odometry transforms and odometry poses
    template <typename T>
    bool operator()(const T* const q_cam_ref, const T* const t_cam_ref,
                    const T* const p_ref, const T* const att_ref,
                    T* residuals) const
    {
        // project 3D object point to the image plane
//        T q_world_cam[4], t_world_cam[3];
//        worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q_world_cam, t_world_cam);
//        T q_world_cam_[4] = {q_world_cam[3], q_world_cam[0], q_world_cam[1], q_world_cam[2]};

        // prepare T_ref_cam
        T q_cam_ref_[4] = {q_cam_ref[3], q_cam_ref[0], q_cam_ref[1], q_cam_ref[2]};
        T q_ref_cam[4];
        T t_ref_cam[3];
        inverseTransform(q_cam_ref_, t_cam_ref, q_ref_cam, t_ref_cam);

        // prepare T_world_ref
        Eigen::Quaternion<T> q_z_inv(cos(att_ref[0] / T(2)), T(0), T(0), -sin(att_ref[0] / T(2)));
        Eigen::Quaternion<T> q_y_inv(cos(att_ref[1] / T(2)), T(0), -sin(att_ref[1] / T(2)), T(0));
        Eigen::Quaternion<T> q_x_inv(cos(att_ref[2] / T(2)), -sin(att_ref[2] / T(2)), T(0), T(0));
        Eigen::Quaternion<T> q_zyx_inv = q_x_inv * q_y_inv * q_z_inv;
        T q_ref_world[4] = {q_zyx_inv.w(), -q_zyx_inv.x(), -q_zyx_inv.y(), -q_zyx_inv.z()};
        T t_ref_world[3] = {p_ref[0], p_ref[1], p_ref[2]};
        T q_world_ref[4];
        T t_world_ref[3];
        inverseTransform(q_ref_world, t_ref_world, q_world_ref, t_world_ref);

        // prepare T_world_cam
        T q_world_cam[4];
        T t_world_cam[3];
        concatTransform(q_ref_cam, t_ref_cam, q_world_ref, t_world_ref, q_world_cam, t_world_cam);

        //compute reprojected point
        T point_in_cam[3];
        T point_in_world[3] = {T(m_observed_P[0]), T(m_observed_P[1]), T(m_observed_P[2])};
        ceres::QuaternionRotatePoint(q_world_cam, point_in_world, point_in_cam);
        point_in_cam[0] += t_world_cam[0]; point_in_cam[1] += t_world_cam[1]; point_in_cam[2] += t_world_cam[2];
        T p1_pred[2] = {point_in_cam[0]/point_in_cam[2], point_in_cam[1]/point_in_cam[2]};
        T p1_pred_norm = sqrt(p1_pred[0]*p1_pred[0] + p1_pred[1]*p1_pred[1]);
        T p1_pixel[2] = {T(m_observed_p(0))-T(m_cx), T(m_observed_p(1))-T(m_cy)};
        T p1_pixel_norm = sqrt(p1_pixel[0]*p1_pixel[0] + p1_pixel[1]*p1_pixel[1]);


        T p1_pred_proj[2];
        p1_pred_proj[0] = (p1_pred[0]*p1_pred[0]*p1_pixel[0]
                          + p1_pred[0]*p1_pred[1]*p1_pixel[1])/(p1_pred_norm*p1_pred_norm);
        p1_pred_proj[1] = (p1_pred[1]*p1_pred[1]*p1_pixel[1]
                          + p1_pred[0]*p1_pred[1]*p1_pixel[0])/(p1_pred_norm*p1_pred_norm);

        residuals[0] = p1_pred_proj[0] - p1_pixel[0];
        residuals[1] = p1_pred_proj[1] - p1_pixel[1];

        return true;
    }


    // variables: camera-odometry transforms and odometry poses and principle point
    template <typename T>
    bool operator()(const T* const q_cam_ref, const T* const t_cam_ref,
                    const T* const p_ref, const T* const att_ref,
                    const T* const cx, const T* const cy,
                    T* residuals) const
    {
        // project 3D object point to the image plane
//        T q_world_cam[4], t_world_cam[3];
//        worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q_world_cam, t_world_cam);
//        T q_world_cam_[4] = {q_world_cam[3], q_world_cam[0], q_world_cam[1], q_world_cam[2]};

        // prepare T_ref_cam
        T q_cam_ref_[4] = {q_cam_ref[3], q_cam_ref[0], q_cam_ref[1], q_cam_ref[2]};
        T q_ref_cam[4];
        T t_ref_cam[3];
        inverseTransform(q_cam_ref_, t_cam_ref, q_ref_cam, t_ref_cam);

        // prepare T_world_ref
        Eigen::Quaternion<T> q_z_inv(cos(att_ref[0] / T(2)), T(0), T(0), -sin(att_ref[0] / T(2)));
        Eigen::Quaternion<T> q_y_inv(cos(att_ref[1] / T(2)), T(0), -sin(att_ref[1] / T(2)), T(0));
        Eigen::Quaternion<T> q_x_inv(cos(att_ref[2] / T(2)), -sin(att_ref[2] / T(2)), T(0), T(0));
        Eigen::Quaternion<T> q_zyx_inv = q_x_inv * q_y_inv * q_z_inv;
        T q_ref_world[4] = {q_zyx_inv.w(), -q_zyx_inv.x(), -q_zyx_inv.y(), -q_zyx_inv.z()};
        T t_ref_world[3] = {p_ref[0], p_ref[1], p_ref[2]};
        T q_world_ref[4];
        T t_world_ref[3];
        inverseTransform(q_ref_world, t_ref_world, q_world_ref, t_world_ref);

        // prepare T_world_cam
        T q_world_cam[4];
        T t_world_cam[3];
        concatTransform(q_ref_cam, t_ref_cam, q_world_ref, t_world_ref, q_world_cam, t_world_cam);

        //compute reprojected point
        T point_in_cam[3];
        T point_in_world[3] = {T(m_observed_P[0]), T(m_observed_P[1]), T(m_observed_P[2])};
        ceres::QuaternionRotatePoint(q_world_cam, point_in_world, point_in_cam);
        point_in_cam[0] += t_world_cam[0]; point_in_cam[1] += t_world_cam[1]; point_in_cam[2] += t_world_cam[2];
        T p1_pred[2] = {point_in_cam[0]/point_in_cam[2], point_in_cam[1]/point_in_cam[2]};
        T p1_pred_norm = sqrt(p1_pred[0]*p1_pred[0] + p1_pred[1]*p1_pred[1]);
        T p1_pixel[2] = {T(m_observed_p(0))-T(cx[0]), T(m_observed_p(1))-T(cy[0])};
        T p1_pixel_norm = sqrt(p1_pixel[0]*p1_pixel[0] + p1_pixel[1]*p1_pixel[1]);


        T p1_pred_proj[2];
        p1_pred_proj[0] = (p1_pred[0]*p1_pred[0]*p1_pixel[0]
                           + p1_pred[0]*p1_pred[1]*p1_pixel[1])/(p1_pred_norm*p1_pred_norm);
        p1_pred_proj[1] = (p1_pred[1]*p1_pred[1]*p1_pixel[1]
                           + p1_pred[0]*p1_pred[1]*p1_pixel[0])/(p1_pred_norm*p1_pred_norm);

        residuals[0] = p1_pred_proj[0] - p1_pixel[0];
        residuals[1] = p1_pred_proj[1] - p1_pixel[1];

        return true;
    }


    // variables: camera poses
    template <typename T>
    bool operator()(const T* const q_world_cam, const T* const t_world_cam,
                    T* residuals) const
    {
        // prepare T_world_cam
        T q_world_cam_[4] = {q_world_cam[3], q_world_cam[0], q_world_cam[1], q_world_cam[2]};

        //compute reprojected point
        T point_in_cam[3];
        T point_in_world[3] = {T(m_observed_P[0]), T(m_observed_P[1]), T(m_observed_P[2])};
        ceres::QuaternionRotatePoint(q_world_cam_, point_in_world, point_in_cam);
        point_in_cam[0] += t_world_cam[0]; point_in_cam[1] += t_world_cam[1]; point_in_cam[2] += t_world_cam[2];
        T p1_pred[2] = {point_in_cam[0]/point_in_cam[2], point_in_cam[1]/point_in_cam[2]};
        T p1_pred_norm = sqrt(p1_pred[0]*p1_pred[0] + p1_pred[1]*p1_pred[1]);
        T p1_pixel[2] = {T(m_observed_p(0))-T(m_cx), T(m_observed_p(1))-T(m_cy)};
        T p1_pixel_norm = sqrt(p1_pixel[0]*p1_pixel[0] + p1_pixel[1]*p1_pixel[1]);


        T p1_pred_proj[2];
        p1_pred_proj[0] = (p1_pred[0]*p1_pred[0]*p1_pixel[0]
                           + p1_pred[0]*p1_pred[1]*p1_pixel[1])/(p1_pred_norm*p1_pred_norm);
        p1_pred_proj[1] = (p1_pred[1]*p1_pred[1]*p1_pixel[1]
                           + p1_pred[0]*p1_pred[1]*p1_pixel[0])/(p1_pred_norm*p1_pred_norm);

        residuals[0] = p1_pred_proj[0] - p1_pixel[0];
        residuals[1] = p1_pred_proj[1] - p1_pixel[1];

        return true;
    }


private:
    double m_cx;
    double m_cy;
    // observed 3D point
    Eigen::Vector3d m_observed_P;

    // observed 2D point
    Eigen::Vector2d m_observed_p;

};

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const std::vector<double> weights) const
{
    ceres::CostFunction* costFunction = 0;
    costFunction =
            new ceres::AutoDiffCostFunction<IntrinsicDiffError<PinholeCamera>, 8, 8, 8>(
                    new IntrinsicDiffError<PinholeCamera>(weights));
    return costFunction;

}

ceres::CostFunction*
CostFunctionFactory::pentagonCostFunction(const std::vector<double> weights, const int edge_id) const
{
    ceres::CostFunction* costFunction = 0;
    costFunction =
            new ceres::AutoDiffCostFunction<PentagonExtrinsicsDiffError, 6, 4, 3, 4, 3, 1, 1>(
                    new PentagonExtrinsicsDiffError(weights, edge_id));
    return costFunction;

}

ceres::CostFunction*
CostFunctionFactory::baselineCostFunction(const double weight, const double baseline) const
{
    ceres::CostFunction* costFunction = 0;
    costFunction =
            new ceres::AutoDiffCostFunction<BaselineDiffError2, 1, 1, 1>(
                    new BaselineDiffError2(weight, baseline));
    return costFunction;

}

ceres::CostFunction*
CostFunctionFactory::baselineCostFunction(const double weight) const
{
    ceres::CostFunction* costFunction = 0;
    costFunction =
            new ceres::AutoDiffCostFunction<BaselineDiffError, 3, 4, 3, 4, 3, 1>(
                    new BaselineDiffError(weight));
    return costFunction;

}

ceres::CostFunction*
CostFunctionFactory::baselineCostFunction2(const double weight) const
{
    ceres::CostFunction* costFunction = 0;
    costFunction =
            new ceres::AutoDiffCostFunction<BaselineDiffError3, 5, 1,1,1,1,1,1,1,1,1,1>(
                    new BaselineDiffError3(weight));
    return costFunction;

}

ceres::CostFunction*
CostFunctionFactory::radialPoseCostFunction(const Eigen::Quaterniond& world_cam_q,
                                            const Eigen::Vector3d& world_cam_t) const
{
    ceres::CostFunction* costFunction = 0;
    costFunction =
            new ceres::AutoDiffCostFunction<RadialPoseDiffError, 5, 4, 3, 3, 3>(
                    new RadialPoseDiffError(world_cam_q, world_cam_t));
    return costFunction;

}

ceres::CostFunction*
CostFunctionFactory::radialPoseCostFunction(const double cx, const double cy,
                                            const Eigen::Vector3d& observed_P,
                                            const Eigen::Vector2d& observed_p) const
{
    ceres::CostFunction* costFunction = 0;
    costFunction =
            new ceres::AutoDiffCostFunction<RadialPoseDiffError2, 2, 4, 3, 3, 3>(
                    new RadialPoseDiffError2(cx, cy, observed_P, observed_p));
    return costFunction;

}

ceres::CostFunction*
CostFunctionFactory::radialPoseCostFunction(const Eigen::Vector3d& observed_P,
                                            const Eigen::Vector2d& observed_p) const
{
    ceres::CostFunction* costFunction = 0;
    costFunction =
            new ceres::AutoDiffCostFunction<RadialPoseDiffError2, 2, 4, 3, 3, 3, 1, 1>(
                    new RadialPoseDiffError2(observed_P, observed_p));
    return costFunction;

}


//variable camera poses
ceres::CostFunction*
CostFunctionFactory::radialPoseCostFunction2(const double cx, const double cy,
                                            const Eigen::Vector3d& observed_P,
                                            const Eigen::Vector2d& observed_p) const
{
    ceres::CostFunction* costFunction = 0;
    costFunction =
            new ceres::AutoDiffCostFunction<RadialPoseDiffError2, 2, 4, 3>(
                    new RadialPoseDiffError2(cx, cy, observed_P, observed_p));
    return costFunction;

}


ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const CameraConstPtr& camera,
                                          const Eigen::Vector3d& observed_P,
                                          const Eigen::Vector2d& observed_p,
                                          const Eigen::Vector2d& cxcy) const
{
    ceres::CostFunction* costFunction = 0;

    switch (camera->modelType())
    {
        case Camera::KANNALA_BRANDT:
            costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError5<EquidistantCamera>, 2, 8, 1>(
                            new ReprojectionError5<EquidistantCamera>(observed_P, observed_p, cxcy));
            break;
        case Camera::PINHOLE:
            costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError5<PinholeCamera>, 2, 8, 1>(
                            new ReprojectionError5<PinholeCamera>(observed_P, observed_p, cxcy));
            break;
        case Camera::MEI:
            costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 9, 1>(
                            new ReprojectionError1<CataCamera>(observed_P, observed_p));
            break;
        case Camera::SCARAMUZZA:
            costFunction =
                    new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 1>(
                            new ReprojectionError1<OCAMCamera>(observed_P, observed_p));
            break;
    }
    return costFunction;
}

}