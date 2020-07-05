#ifndef INFRASTRUCTURECALIBRATION_H
#define INFRASTRUCTURECALIBRATION_H

#include <boost/thread.hpp>
#include "src/SiftGPU/SiftGPU.h"

#include "infrascal/camera_systems/CameraSystem.h"
#include "infrascal/sparse_graph/SparseGraph.h"
#include "src/radialpose/radialpose.h"

namespace infrascal
{

// forward declaration
class LocationRecognition;

class InfrastructureCalibration
{
public:
    enum
    {
        CAMERA,
        ODOMETRY
    };

    enum
    {
        PRUNE_BEHIND_CAMERA = 0x1,
        PRUNE_FARAWAY = 0x2,
        PRUNE_HIGH_REPROJ_ERR = 0x4
    };

    class Options
    {
    public:
        Options()
         : verbose(false) {};

        std::string dataDir;
        std::string outputDir;
        bool optimizePoints;
        bool optimizeIntrinsics;
        bool verbose;
        std::string calibMode;
    };

    typedef struct
    {
        uint64_t timestamp = 0;
        std::vector<FramePtr> frames;
    } FrameSet;

    //create calib object
    InfrastructureCalibration(std::vector<CameraPtr>& cameras,
                              const Options& options);
    //load map
    bool loadMap(const std::string& mapDirectory, const std::string& databaseDirectory, const std::string& vocFilename);

    //add one frameset and estimate camera poses for the frame
    bool addFrameSet(const std::vector<cv::Mat>& images,
                     uint64_t timestamp, bool preprocess);

    //add odom
    void addOdometry(double x, double y, double yaw, uint64_t timestamp);

    void reset(void);
    void run(void);

    void loadFrameSets(const std::string& filename);
    void saveFrameSets(const std::string& filename) const;
    bool writeToColmap(const std::string& filename);

    const CameraSystem& cameraSystem(void) const;

private:
    void extractFeatures(const cv::Mat &image, uint64_t timestamp, FramePtr &frame, bool preprocess = false);

    void estimateCameraPose(uint64_t timestamp, FramePtr& frame);
    void prune(int flags, int poseType);
    void optimize(int flags);
    void optimizeIntrinsics();
    void optimizeRadialRigpose();
    void optimizeRadialRigposeBA();
    void upgradeRadialCamera();
    void upgradeOptiRadialCamera();

    cv::Mat buildDescriptorMat(const std::vector<Point2DFeaturePtr>& features,
                               std::vector<size_t>& indices,
                               bool hasScenePoint) const;
    std::vector<cv::DMatch> matchFeatures(const std::vector<Point2DFeaturePtr>& queryFeatures,
                                          const std::vector<Point2DFeaturePtr>& trainFeatures) const;

    void solveP3PRansac(const FrameConstPtr& frame1,
                        const std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > corr2D3Ds,
                        Eigen::Matrix4d& H,
                        std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> >& inliers) const;

    void solvePnPRansac(const FrameConstPtr& frame1,
                        const std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > corr2D3Ds,
                        Eigen::Matrix4d& H,
                        std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> >& inliers,
                        radialpose::Camera& cam_param);
                                          
    void solvePnPRadialRansac(const FrameConstPtr& frame1,
                              const std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > corr2D3Ds,
                              Eigen::Matrix4d& H,
                              std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> >& inliers) const;

    double reprojectionError(const CameraConstPtr& camera,
                             const Eigen::Vector3d& P,
                             const Eigen::Quaterniond& cam_ref_q,
                             const Eigen::Vector3d& cam_ref_t,
                             const Eigen::Vector3d& ref_p,
                             const Eigen::Vector3d& ref_att,
                             const Eigen::Vector2d& observed_p) const;

    void frameReprojectionError(const FramePtr& frame,
                                const CameraConstPtr& camera,
                                const Pose& T_cam_ref,
                                double& minError, double& maxError, double& avgError,
                                size_t& featureCount) const;

    void frameReprojectionError(const FramePtr& frame,
                                const CameraConstPtr& camera,
                                double& minError, double& maxError, double& avgError,
                                size_t& featureCount) const;

    void reprojectionError(double& minError, double& maxError,
                           double& avgError, size_t& featureCount) const;

    double radialReprojectionError(const CameraConstPtr& camera,
                             const Eigen::Vector3d& P,
                             const Eigen::Quaterniond& cam_ref_q,
                             const Eigen::Vector3d& cam_ref_t,
                             const Eigen::Vector3d& ref_p,
                             const Eigen::Vector3d& ref_att,
                             const Eigen::Vector2d& observed_p) const;

    void frameRadialReprojectionError(const FramePtr& frame,
                                const CameraConstPtr& camera,
                                const Pose& T_cam_ref,
                                double& minError, double& maxError, double& avgError,
                                size_t& featureCount) const;

    void frameRadialReprojectionError(const FramePtr& frame,
                                const CameraConstPtr& camera,
                                double& minError, double& maxError, double& avgError,
                                size_t& featureCount) const;

    void radialReprojectionError(double& minError, double& maxError,
                           double& avgError, size_t& featureCount) const;



    // Compute the quaternion average using the Markley SVD method
    template <typename FloatT>
    Eigen::Quaternion<FloatT> quaternionAvg(const std::vector<Eigen::Quaternion<FloatT> >& points) const;


    OdometryPtr solveRigPoses(Eigen::MatrixXd cam_T_ref_stack,
                              Eigen::MatrixXd cam_T_world_stack,
                              FrameSet& frameset) const;
    void setRigPoses(std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref);
    void printCameraExtrinsics(const std::string& header) const;

    // inputs
    std::vector<CameraPtr> m_cameras;
    SparseGraph m_refGraph;

    // working data
    boost::shared_ptr<LocationRecognition> m_locrec;
    boost::mutex m_feature3DMapMutex;
    boost::unordered_map<Point3DFeature*, Point3DFeaturePtr> m_feature3DMap;
    std::vector<FrameSet> m_framesets;
    double m_x_last;
    double m_y_last;
    double m_distance;
    bool m_verbose;
    std::vector<radialpose::Camera> m_cam_param_best;
    std::vector<int> m_inlierCounts_best;
    // output
    CameraSystem m_cameraSystem;

    // parameters
    const float k_maxDistanceRatio;
    const float k_maxPoint3DDistance;
    const float k_maxReprojErr;
    const int k_minCorrespondences2D3D;
    const double k_minKeyFrameDistance;
    const int k_nearestImageMatches;
    const double k_reprojErrorThresh;

    Options m_options;

};

template <typename FloatT>
Eigen::Quaternion<FloatT>
InfrastructureCalibration::quaternionAvg(const std::vector<Eigen::Quaternion<FloatT> >& points) const
{
    using namespace Eigen;

    Matrix<FloatT, 3, 3> sum;
    sum.setZero();
    for (int i = 0, end = points.size(); i != end; ++i)
    {
        sum += points[i].toRotationMatrix();
    }

    JacobiSVD<Matrix<FloatT, 3, 3> > svd(sum, ComputeFullU | ComputeFullV);

    Matrix<FloatT, 3, 3> result = svd.matrixU()
        * (Matrix<FloatT, 3, 1>() << 1, 1, svd.matrixU().determinant()*svd.matrixV().determinant()).finished().asDiagonal()
        * svd.matrixV().transpose();
    return Quaternion<FloatT>(result);
}

}

#endif
