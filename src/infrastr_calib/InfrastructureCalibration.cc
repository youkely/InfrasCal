#include "infrascal/infrastr_calib/InfrastructureCalibration.h"

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "../camera_models/CostFunctionFactory.h"
#include "../features2d/SurfGPU.h"
#include "../gpl/EigenQuaternionParameterization.h"
#include "../../gpl/gpl.h"
#include "infrascal/EigenUtils.h"
#ifndef HAVE_OPENCV3
#include "../gpl/OpenCVUtils.h"
#endif // HAVE_OPENCV3
#include "../location_recognition/LocationRecognition.h"
#include "../pose_estimation/P3P.h"
#include "../radialpose/radialpose.h"
#include "infrascal/sparse_graph/SparseGraphUtils.h"
#include "ceres/ceres.h"
#include <GL/gl.h>
#include "infrascal/camera_models/CameraFactory.h"


namespace infrascal
{

InfrastructureCalibration::InfrastructureCalibration(std::vector<CameraPtr>& cameras,
                                                     const Options& options)
 : m_cameras(cameras)
 , m_x_last(0.0)
 , m_y_last(0.0)
 , m_distance(0.0)
 , m_verbose(options.verbose)
 , m_options(options)
 , m_cameraSystem(cameras.size())
 , k_maxDistanceRatio(0.7f)
 , k_maxPoint3DDistance(20.0)
 , k_maxReprojErr(20.0)
 , k_minCorrespondences2D3D(25)
 , k_minKeyFrameDistance(0.1)
 , k_nearestImageMatches(10)
 , k_reprojErrorThresh(10.0)
{

}

bool
InfrastructureCalibration::loadMap(const std::string& mapDirectory, const std::string& databaseDirectory, const std::string& vocFilename)
{
    if (m_verbose)
    {
        std::cout << "# INFO: Loading map... " << std::flush;
    }

    boost::filesystem::path graphPath(mapDirectory);
    if (!m_refGraph.readFromColmap(mapDirectory, databaseDirectory))
    {
        std::cout << std::endl << "# ERROR: Cannot read graph file " << graphPath.string() << "." << std::endl;
        return false;
    }

    if (m_verbose)
    {
        std::cout << "Finished." << std::endl;
        std::cout << "# INFO: Setting up location recognition... " << std::flush;
    }

    m_locrec = boost::make_shared<LocationRecognition>();
    m_locrec->setup(m_refGraph, vocFilename);

    if (m_verbose)
    {
        std::cout << "Finished." << std::endl;
    }

    reset();

    return true;
}

bool
InfrastructureCalibration::addFrameSet(const std::vector<cv::Mat>& images,
                                       uint64_t timestamp,
                                       bool preprocess)
{
    if (images.size() != m_cameras.size())
    {
        std::cout << "# WARNING: Number of images does not match camera count." << std::endl;
        return false;
    }

    std::vector<boost::shared_ptr<boost::thread> > threads(m_cameras.size());
    std::vector<FramePtr> frames(m_cameras.size());
    // estimate camera pose corresponding to each image
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        frames.at(i) = boost::make_shared<Frame>();
        frames.at(i)->cameraId() = i;
        extractFeatures(images.at(i), timestamp,
                           frames.at(i), preprocess);
    }
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        threads.at(i) = boost::make_shared<boost::thread>(&InfrastructureCalibration::estimateCameraPose,
                                                          this,timestamp, frames.at(i));
    }

    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        threads.at(i)->join();
    }

    FrameSet frameset;
    frameset.timestamp = timestamp;

    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        if (frames.at(i)->cameraPose().get() == nullptr)
        {
            continue;
        }

        frameset.frames.push_back(frames.at(i));
    }
    if (frameset.frames.size() < 2)
    {
        return false;
    }

    std::vector<FramePtr> framePrev(m_cameras.size());
    std::vector<FramePtr> frameCurr(m_cameras.size());

    bool addFrameSet = false;

    if (m_framesets.empty())
    {
        addFrameSet = true;
    }

    if (!addFrameSet)
    {
        // estimate keyframe distance by using the minimum of the norm of the
        // translation of the ith camera between two frames
        for (size_t i = 0; i < m_framesets.back().frames.size(); ++i)
        {
            FramePtr& frame = m_framesets.back().frames.at(i);

            framePrev.at(frame->cameraId()) = frame;
        }

        for (size_t i = 0; i < frameset.frames.size(); ++i)
        {
            FramePtr& frame = frameset.frames.at(i);

            frameCurr.at(frame->cameraId()) = frame;
        }

        double keyFrameDist = std::numeric_limits<double>::min();
        for (size_t i = 0; i < m_cameras.size(); ++i)
        {
            if (framePrev.at(i).get() == nullptr || frameCurr.at(i).get() == nullptr)
            {
                continue;
            }

            double d = (frameCurr.at(i)->cameraPose()->toMatrix().inverse().block<3,1>(0,3) -
                        framePrev.at(i)->cameraPose()->toMatrix().inverse().block<3,1>(0,3)).norm();

            if (d > keyFrameDist)
            {
                keyFrameDist = d;
            }
        }

        if (keyFrameDist == std::numeric_limits<double>::min() ||
            keyFrameDist > k_minKeyFrameDistance)
        {
            addFrameSet = true;
        }
        else
        {
            if (m_verbose)
            {
                std::cout << "# INFO: Skipping frame set as inter-frame distance is too small." << std::endl;
            }
        }
    }

    if (!addFrameSet)
    {
        return false;
    }

    m_framesets.push_back(frameset);

    if (m_verbose)
    {
        std::cout << "# INFO: Added frame set " << m_framesets.size()
                  << " [ ";
        for (size_t i = 0; i < frameset.frames.size(); ++i)
        {
            std::cout << frameset.frames.at(i)->cameraId() << " ";
        }
        std::cout << "] ts = " << frameset.timestamp << std::endl;
    }
    return true;
}

void
InfrastructureCalibration::addOdometry(double x, double y, double yaw,
                                       uint64_t timestamp)
{
    if (m_x_last != 0.0 || m_y_last != 0.0)
    {
        m_distance += hypot(x - m_x_last, y - m_y_last);
    }

    m_x_last = x;
    m_y_last = y;
}

void
InfrastructureCalibration::reset(void)
{
    m_feature3DMap.clear();
    m_framesets.clear();
    m_x_last = 0.0;
    m_y_last = 0.0;
    m_distance = 0.0;

    m_cameraSystem = CameraSystem(m_cameras.size());

    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        m_cameraSystem.setCamera(i, m_cameras.at(i));
    }
    m_cameraSystem.setReferenceCamera(0);
    m_inlierCounts_best.resize(m_cameras.size());
    m_cam_param_best.resize(m_cameras.size());
}

void
InfrastructureCalibration::run(void)
{
    if (m_verbose)
    {
        double sumError = 0.0;
        size_t sumFeatureCount = 0;
        for (size_t i = 0; i < m_framesets.size(); ++i)
        {
            FrameSet& frameset = m_framesets.at(i);

            for (size_t j = 0; j < frameset.frames.size(); ++j)
            {
                FramePtr& frame = frameset.frames.at(j);

                double minError, maxError, avgError;
                size_t featureCount;
                if (m_options.calibMode == "InRaSU"){
                    frameRadialReprojectionError(frame,
                                                 m_cameraSystem.getCamera(frame->cameraId()),
                                                 minError, maxError, avgError,
                                                 featureCount);
                } else{
                    frameReprojectionError(frame,
                                                 m_cameraSystem.getCamera(frame->cameraId()),
                                                 minError, maxError, avgError,
                                                 featureCount);
                }


                sumError += avgError * featureCount;
                sumFeatureCount += featureCount;
            }
        }
        if (m_options.calibMode == "InRaSU") {
            std::cout << "# INFO: Average radial reprojection error over all frames: "
                      << sumError / sumFeatureCount << " px" << std::endl;
        }
        else{
            std::cout << "# INFO: Average reprojection error over all frames: "
                      << sumError / sumFeatureCount << " px" << std::endl;
        }

        size_t nFrames = 0;
        for (size_t i = 0; i < m_framesets.size(); ++i)
        {
            FrameSet& frameset = m_framesets.at(i);

            nFrames += frameset.frames.size();
        }

        std::cout << "# INFO: Average number of frames per set: "
                  << static_cast<double>(nFrames) / static_cast<double>(m_framesets.size())
                  << std::endl;
    }

    // without loss of generality, mark camera 0 as the reference frame
    m_cameraSystem.setGlobalCameraPose(0, Eigen::Matrix4d::Identity());

    // find initial estimates for camera extrinsics

    // in each iteration over complete frame sets,
    // compute the relative camera poses with respect to camera 0,
    // and use these extrinsics to compute the average reprojection error
    // over all frame sets. We use the extrinsics with the lowest average
    // reprojection error as the initial estimate.
    double minReprojError = std::numeric_limits<double>::max();
    std::vector<Pose, Eigen::aligned_allocator<Pose> > best_T_cam_ref;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > poses(m_cameras.size());//cam_T_ref
    std::unordered_map<int, int> camHasPose;

    // first iteration, to find cam_T_rig for all cameras
    bool init = false;
    for (size_t i = 0; i < 2*m_framesets.size(); ++i)
    {
        FrameSet& frameset = m_framesets.at(i%m_framesets.size());

        if (m_verbose)
        {
            if (i>m_framesets.size())
                continue;
            std::cout << "# INFO: Added frame set " << i
                      << " [ ";
            for (size_t i = 0; i < frameset.frames.size(); ++i)
            {
                std::cout << frameset.frames.at(i)->cameraId() << " ";
            }
            std::cout << "] ts = " << frameset.timestamp << std::endl;
        }

        if ((m_options.calibMode =="InRa"||m_options.calibMode =="In"||m_options.calibMode =="InRI")
            && frameset.frames.size() < m_cameras.size())
        {
            continue;
        }

        if (frameset.frames.size() < 2)
        {
            continue;
        }
        // set system pose to first camera
        if (!init)
        {
            init = true;
            camHasPose.clear();
            OdometryPtr odometry = boost::make_shared<Odometry>();
            odometry->timeStamp() = frameset.frames.at(0)->cameraPose()->timeStamp();
            Eigen::Matrix4d H;

            H = frameset.frames.at(0)->cameraPose()->toMatrix().inverse();
            odometry->position() = H.block<3,1>(0,3);

            Eigen::Quaterniond qAvg = Eigen::Quaterniond(H.block<3,3>(0,0));

            double roll, pitch, yaw;
            mat2RPY(qAvg.toRotationMatrix(), roll, pitch, yaw);

            odometry->attitude() = Eigen::Vector3d(yaw, pitch, roll);
            for (size_t j = 0; j < frameset.frames.size(); ++j)
            {
                frameset.frames.at(j)->systemPose() = odometry;
            }
        }
        //Eigen::Matrix4d prev_T_cur = Eigen::Matrix4d::Identity();

        if(!camHasPose.empty())
        {
            int numberOverlap = 0;
            int cameraIdx;
            Eigen::MatrixXd cam_T_ref_stack(4,4);
            Eigen::MatrixXd cam_T_world_stack(4,4);

            for (size_t j = 0; j < frameset.frames.size(); ++j) {
                cameraIdx = frameset.frames.at(j)->cameraId();
                if (camHasPose.count(cameraIdx) > 0) {
                    cam_T_world_stack.conservativeResize(numberOverlap * 2 + 2, 4);
                    cam_T_ref_stack.conservativeResize(numberOverlap * 2 + 2, 4);
                    cam_T_world_stack.block<2, 4>(numberOverlap * 2, 0) =
                            frameset.frames.at(j)->cameraPose()->toMatrix().block<2, 4>(0, 0);
                    cam_T_ref_stack.block<2, 4>(numberOverlap * 2, 0) =
                            poses.at(cameraIdx).block<2, 4>(0, 0);
                    numberOverlap++;
                }
            }
            if(numberOverlap < 2){
                continue;
            }

            OdometryPtr odometry = solveRigPoses(cam_T_ref_stack, cam_T_world_stack, frameset);

            for (size_t j = 0; j < frameset.frames.size(); ++j)
            {
                frameset.frames.at(j)->systemPose() = odometry;
            }
        }

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            int cameraIdx = frameset.frames.at(j)->cameraId();
            poses.at(cameraIdx) = frameset.frames.at(j)->cameraPose()->toMatrix() *
                    frameset.frames.at(j)->systemPose()->toMatrix();
            camHasPose[cameraIdx]++;
        }

        if(camHasPose.size() < m_cameras.size()){
            continue;
        }
        camHasPose.clear();
        init = false;

        //set camera reference extrinsics
        std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref(m_cameras.size());
        for (size_t j = 0; j < m_cameras.size(); ++j)
        {
            if(m_options.calibMode=="InRaSU"){
                poses.at(j)(2,3) = 0;
            }
            Eigen::Matrix4d H_cam_ref = poses.at(j).inverse();
            T_cam_ref.at(j).rotation() = Eigen::Quaterniond(H_cam_ref.block<3,3>(0,0));
            T_cam_ref.at(j).translation() = H_cam_ref.block<3,1>(0,3);

            m_cameraSystem.setGlobalCameraPose(j, H_cam_ref);
        }

        setRigPoses(T_cam_ref);

        // compute average reprojection error over all frame sets
        double minError, maxError, avgError;
        size_t featureCount;
        if(m_options.calibMode=="InRaSU") {
            radialReprojectionError(minError, maxError, avgError, featureCount);
        } else{
            reprojectionError(minError, maxError, avgError, featureCount);
        }

        if (avgError < minReprojError)
        {
            minReprojError = avgError;
            best_T_cam_ref = T_cam_ref;
        }
    }

    if (minReprojError == std::numeric_limits<double>::max())
    {
        std::cout << "# ERROR: No complete frame sets were found." << std::endl;
        return;
    }

    for (size_t j = 0; j < m_cameras.size(); ++j)
    {
        m_cameraSystem.setGlobalCameraPose(j, best_T_cam_ref.at(j).toMatrix());
    }
    setRigPoses(best_T_cam_ref);
    if (m_verbose)
    {
        printCameraExtrinsics("best guess");
        double minError, maxError, avgError;
        size_t featureCount;
        if(m_options.calibMode=="InRaSU") {
            radialReprojectionError(minError, maxError, avgError, featureCount);
            std::cout << "# INFO: Radial reprojection error: avg = " << avgError
                      << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
        } else{
            reprojectionError(minError, maxError, avgError, featureCount);
            std::cout << "# INFO: Reprojection error: avg = " << avgError
                      << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
        }
    }

    if(m_options.calibMode=="InRaSU") {
        // optimization P and Q for all frames
        optimizeRadialRigpose();
        if (m_verbose) {
            printCameraExtrinsics("after cam pose optimization, cam ref");
            double minError, maxError, avgError;
            size_t featureCount;
            radialReprojectionError(minError, maxError, avgError, featureCount);
            std::cout << "# INFO: Radial reprojection error: avg = " << avgError
                      << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
        }

        // optimization P and Q for all frames
        optimizeRadialRigposeBA();
        if (m_verbose) {
            printCameraExtrinsics("after cam pose BA, cam ref");
            double minError, maxError, avgError;
            size_t featureCount;
            radialReprojectionError(minError, maxError, avgError, featureCount);
            std::cout << "# INFO: Radial reprojection error: avg = " << avgError
                      << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
        }

        // upgrade each camera
        upgradeRadialCamera();
        if (m_verbose) {
            printCameraExtrinsics("after cam upgrade");
            double minError, maxError, avgError;
            size_t featureCount;
            reprojectionError(minError, maxError, avgError, featureCount);
            std::cout << "# INFO: Reprojection error: avg = " << avgError
                      << " px | max = " << maxError << " px | count = " << featureCount << std::endl;

        }
        upgradeOptiRadialCamera();
        if (m_verbose) {
            printCameraExtrinsics("after cam upgrade BA");
            double minError, maxError, avgError;
            size_t featureCount;
            reprojectionError(minError, maxError, avgError, featureCount);
            std::cout << "# INFO: Reprojection error: avg = " << avgError
                      << " px | max = " << maxError << " px | count = " << featureCount << std::endl;

        }
    }
    else if(m_options.calibMode == "InRa" || m_options.calibMode == "InRaS")
    {
        boost::filesystem::path cameraSystemBeforeBAPath(m_options.outputDir);
        cameraSystemBeforeBAPath /= "infrastr_beforeBA.xml";
        m_cameraSystem.writeToXmlFile(cameraSystemBeforeBAPath.string());
        boost::filesystem::path extrinsics_init(m_options.outputDir);
        extrinsics_init /= "extrinsic_init.txt";
        m_cameraSystem.writePosesToTextFile(extrinsics_init.string());
    }


    int optimizeFlags = CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE;
    if (m_options.optimizePoints)
    {
        optimizeFlags = optimizeFlags | POINT_3D;
    }
    if (m_options.optimizeIntrinsics)
    {
        optimizeFlags = optimizeFlags | CAMERA_INTRINSICS;
    }
    //prune(PRUNE_FARAWAY | PRUNE_BEHIND_CAMERA, ODOMETRY);

    //optimizeIntrinsics();
    optimize(optimizeFlags);

    prune(PRUNE_FARAWAY | PRUNE_BEHIND_CAMERA | PRUNE_HIGH_REPROJ_ERR, ODOMETRY);

    // run non-linear optimization to optimize odometry poses and camera extrinsics
    optimize(optimizeFlags);
    if (m_verbose)
    {
        for (size_t i = 0; i < m_framesets.size(); ++i) {
            FrameSet &frameset = m_framesets.at(i);
            addOdometry(frameset.frames.at(0)->systemPose()->x(),
                        frameset.frames.at(0)->systemPose()->y(),
                        frameset.frames.at(0)->systemPose()->yaw(),
                        frameset.timestamp);
        }
        std::cout << "# INFO: Odometry distance: " << m_distance << " m" << std::endl;
    }

    // write extrinsic data
    boost::filesystem::path cameraSystemPath(m_options.outputDir);
    cameraSystemPath /= "infrastr.xml";
    m_cameraSystem.writeToXmlFile(cameraSystemPath.string());
    m_cameraSystem.writeToDirectory(m_options.outputDir);
    if (m_verbose)
    {
        printCameraExtrinsics("final result");
    }

}

void
InfrastructureCalibration::loadFrameSets(const std::string& filename)
{
    m_framesets.clear();

    SparseGraph graph;

    graph.readFromBinaryFile(filename);

    for (size_t i = 0; i < graph.frameSetSegment(0).size(); ++i)
    {
        FrameSet frameset;
        uint64_t timestamp = 0;

        for (int j = 0; j < (int)graph.frameSetSegment(0).at(i)->frames().size(); ++j)
        {
            FramePtr& frame = graph.frameSetSegment(0).at(i)->frames().at(j);

            if (frame.get() == 0)
            {
                continue;
            }

            timestamp = frame->cameraPose()->timeStamp();

            frameset.frames.push_back(frame);
        }

        frameset.timestamp = timestamp;

        m_framesets.push_back(frameset);
    }

    if (m_verbose)
    {
        std::cout << "# INFO: Loaded " << m_framesets.size() << " frame sets from " << filename << std::endl;
    }

}

void
InfrastructureCalibration::saveFrameSets(const std::string& filename) const
{
    SparseGraph graph;

    graph.frameSetSegments().resize(1);
    graph.frameSetSegment(0).resize(m_framesets.size());

    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        const FrameSet& frameset = m_framesets.at(i);

        graph.frameSetSegment(0).at(i) = boost::make_shared<infrascal::FrameSet>();
        graph.frameSetSegment(0).at(i)->frames().resize(m_cameras.size());

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            int cameraId = frameset.frames.at(j)->cameraId();

            graph.frameSetSegment(0).at(i)->frames().at(cameraId) = frameset.frames.at(j);
        }
    }

    graph.writeToBinaryFile(filename);

    if (m_verbose)
    {
        std::cout << "# INFO: Wrote " << m_framesets.size() << " frame sets to " << filename << std::endl;
    }
}

bool
InfrastructureCalibration::writeToColmap(const std::string& filename) {
    boost::filesystem::path imageFilePath(filename);
    imageFilePath /= "images.txt";
    std::ofstream file_images(imageFilePath.string());
    file_images << "# Image list with two lines of data per image:"<<std::endl;
    file_images << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"<<std::endl;
    file_images << "#   POINTS2D[] as (X, Y, POINT3D_ID)"<<std::endl;
    int count = 0;
    std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref(m_cameras.size());
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        T_cam_ref.at(i) = Pose(m_cameraSystem.getGlobalCameraPose(i));
    }

    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        const FrameSet& frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            const FramePtr frame = frameset.frames.at(j);
            if (frame.get() == 0)
            {
                continue;
            }
            int cameraId = frame->cameraId();
            Eigen::Matrix4d T_ref_cam = T_cam_ref.at(cameraId).toMatrix().inverse();
            Eigen::Matrix4d T_world_ref = frame->systemPose()->toMatrix().inverse();

            Eigen::Matrix4d T_world_cam = T_ref_cam * T_world_ref;
            Eigen::Matrix3d R_world_cam = Eigen::Matrix3d(T_world_cam.block<3,3>(0,0));
            Eigen::Quaterniond q_world_cam = Eigen::Quaterniond(R_world_cam);

            file_images <<i*m_cameras.size()+cameraId <<" "
                        <<q_world_cam.w()<<" "
                        <<q_world_cam.x()<<" "
                        <<q_world_cam.y()<<" "
                        <<q_world_cam.z()<<" "
                        <<T_world_cam(0,3)<<" "
                        <<T_world_cam(1,3)<<" "
                        <<T_world_cam(2,3)<<" "
                        <<cameraId<<" "
                        <<i*m_cameras.size()+cameraId
                        <<std::endl;
            for (size_t k = 0; k < frame->features2D().size(); ++k)
            {
                Point2DFeaturePtr &feature2D = frame->features2D().at(k);
                if (feature2D->feature3D().get() == 0)
                    file_images <<feature2D->keypoint().pt.x<<" "<<feature2D->keypoint().pt.y<<" -1 ";
                else
                    file_images <<feature2D->keypoint().pt.x<<" "<<feature2D->keypoint().pt.y<<" "<<
                    feature2D->feature3D()->attributes()<<" ";
            }
            file_images << std::endl;
            count++;

        }
    }
    file_images.close();

    boost::filesystem::path cameraFilePath(filename);
    cameraFilePath /= "cameras.txt";
    std::ofstream file_cameras(cameraFilePath.string());
    file_cameras << "# # Camera list with one line of data per camera:"<<std::endl;
    file_cameras << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"<<std::endl;
    file_cameras << "# Number of cameras: "<<m_cameras.size()<<std::endl;
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        const CameraConstPtr& camera = m_cameraSystem.getCamera(i);
        std::vector<double> intrinsicParams;
        camera->writeParameters(intrinsicParams);
        if(m_cameraSystem.getCamera(i)->modelType() == Camera::KANNALA_BRANDT){
            file_cameras << i << " OPENCV_FISHEYE "<< camera->imageWidth()<<" "
                        << camera->imageHeight()<<" "
                        << intrinsicParams[4] <<" "
                        << intrinsicParams[5] <<" "
                        << intrinsicParams[6] <<" "
                        << intrinsicParams[7] <<" "
                        << intrinsicParams[0] <<" "
                        << intrinsicParams[1] <<" "
                        << intrinsicParams[2] <<" "
                        << intrinsicParams[3] <<std::endl;
        }
        if(m_cameraSystem.getCamera(i)->modelType() == Camera::PINHOLE){
            file_cameras << i << " OPENCV "<< camera->imageWidth()<<" "
                        << camera->imageHeight()<<" "
                        << intrinsicParams[4] <<" "
                        << intrinsicParams[5] <<" "
                        << intrinsicParams[6] <<" "
                        << intrinsicParams[7] <<" "
                        << intrinsicParams[0] <<" "
                        << intrinsicParams[1] <<" "
                        << intrinsicParams[2] <<" "
                        << intrinsicParams[3] <<std::endl;
        }
    }
    file_cameras.close();


}


void
InfrastructureCalibration::extractFeatures(const cv::Mat& image,
                                           uint64_t timestamp,
                                           FramePtr& frame,
                                           bool preprocess)
{

    cv::Mat imageProc;
    if (preprocess)
    {


#ifdef HAVE_CUDA
        //////////////////
        // CUDA + OPENCV2 + OPENCV3
        //////////////////

#ifdef HAVE_OPENCV3
        typedef cv::cuda::GpuMat CUDAMat;
#else // HAVE_OPENCV3
        typedef cv::gpu::GpuMat CUDAMat;
#endif // HAVE_OPENCV3


        CUDAMat gpuImage, gpuImageProc;
        gpuImage.upload(image);
        cv::equalizeHist(gpuImage, gpuImageProc);


        gpuImageProc.download(imageProc);
#else // HAVE_CUDA
        //////////////////
    // OPENCV2 + OPENCV3
    //////////////////
        cv::Mat gpuImage, gpuImageProc;
        cv::equalizeHist(gpuImage, gpuImageProc);
#endif
    }
    else if (image.channels() > 1)
    {
        cv::cvtColor(image, imageProc, CV_BGR2GRAY);
    }
    else
    {
        image.copyTo(imageProc);
    }


    // compute keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

#define  SIFT
#ifdef SIFT
    SiftGPU sift_gpu;
    const char* myargv[20] ={ "./sift_gpu", "-v", "0", "-maxd", "4000", "-tc2", "1000", "-cuda", "0",
                              "-fo", "-1", "-d", "3", "-t", "0.0067", "-e", "10.0", "-ofix", "-mo", "1" };
    sift_gpu.ParseParam(20, myargv);


    if(sift_gpu.VerifyContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED){
        std::cerr<<"SiftGPU is not supported!"<<endl;
    }

    sift_gpu.RunSIFT( imageProc.cols, imageProc.rows, imageProc.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);

    const size_t num_features = static_cast<size_t>(sift_gpu.GetFeatureNum());
    vector<float> siftDescriptors(128*num_features);
    vector<SiftGPU::SiftKeypoint> siftKeypoints(num_features);
    keypoints.resize(num_features);

    // Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            descriptors_float(num_features, 128);

    // Download the extracted keypoints and descriptors.
    sift_gpu.GetFeatureVector(siftKeypoints.data(),
                              descriptors_float.data());

    Eigen::MatrixXf descriptors_normalized(descriptors_float.rows(),
                                           descriptors_float.cols());
    for (Eigen::MatrixXf::Index r = 0; r < descriptors_float.rows(); ++r) {
        const float norm = descriptors_float.row(r).lpNorm<1>();
        descriptors_normalized.row(r) = descriptors_float.row(r) / norm;
        descriptors_normalized.row(r) =
                descriptors_normalized.row(r).array().sqrt();
    }

    eigen2cv(descriptors_normalized, descriptors);

    for(int i = 0; i < num_features; i++){
        keypoints[i].pt.x = siftKeypoints[i].x;
        keypoints[i].pt.y = siftKeypoints[i].y;
        keypoints[i].size = siftKeypoints[i].s;
        keypoints[i].angle = siftKeypoints[i].o;
    }

#else
    cv::Ptr<SurfGPU> surf = SurfGPU::instance(300.0, 4, 2, true, 0.01);
    surf->detect(imageProc, keypoints);
    surf->compute(imageProc, keypoints, descriptors);


#endif

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        Point2DFeaturePtr feature2D = boost::make_shared<Point2DFeature>();

        feature2D->keypoint() = keypoints.at(i);
        descriptors.row(i).copyTo(feature2D->descriptor());
        feature2D->index() = i;
        feature2D->frame() = frame;

        frame->features2D().push_back(feature2D);
    }
}


void
InfrastructureCalibration::estimateCameraPose(uint64_t timestamp,
                                              FramePtr& frame)
{
    double tsStart = timeInSeconds();

    // find k closest matches in vocabulary tree
    std::vector<FrameTag> candidates;
    m_locrec->knnMatch_maploc(frame, k_nearestImageMatches, candidates);

    // find match with highest number of inlier 2D-2D correspondences
    int bestInlierCount = 0;
    std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > bestCorr2D3D;
    std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > allCorr2D3D;
    Eigen::Matrix4d bestH;
    radialpose::Camera best_cam_param;
    if (m_options.calibMode == "In" || m_options.calibMode == "InRI")
    {
        std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > Corr2D3Ds;

        for (size_t i = 0; i < candidates.size(); ++i)
        {
            FrameTag tag = candidates.at(i);

            FramePtr &trainFrame = m_refGraph.frameSetSegment(tag.frameSetSegmentId).at(tag.frameSetId)->frames().at(
                    tag.frameId);
            // find 2D-3D correspondences
            std::vector<cv::DMatch> matches = matchFeatures(frame->features2D(), trainFrame->features2D());

            if ((int) matches.size() < k_minCorrespondences2D3D)
            {
                continue;
            }
            for (size_t j = 0; j < matches.size(); ++j)
            {
                cv::DMatch match = matches.at(j);
                Corr2D3Ds.push_back(std::make_pair(frame->features2D().at(match.queryIdx),
                                                   trainFrame->features2D().at(match.trainIdx)->feature3D()));
            }

            // find camera pose from P3P RANSAC
            Eigen::Matrix4d H;
            std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > inliers;

            solveP3PRansac(frame, Corr2D3Ds, H, inliers);

            int nInliers = inliers.size();

            if (nInliers < k_minCorrespondences2D3D)
            {
                continue;
            }

            if (nInliers > bestInlierCount)
            {
                bestInlierCount = nInliers;
                bestH = H;
                bestCorr2D3D.clear();
                for (size_t j = 0; j < inliers.size(); ++j)
                {
                    bestCorr2D3D.push_back(inliers.at(j));
                }
            }
        }
    }
    else
    {
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            FrameTag tag = candidates.at(i);

            FramePtr &trainFrame = m_refGraph.frameSetSegment(tag.frameSetSegmentId).at(tag.frameSetId)->frames().at(
                    tag.frameId);
            // find 2D-3D correspondences
            std::vector<cv::DMatch> matches = matchFeatures(frame->features2D(), trainFrame->features2D());
            for (size_t j = 0; j < matches.size(); ++j)
            {
                cv::DMatch match = matches.at(j);
                allCorr2D3D.push_back(std::make_pair(frame->features2D().at(match.queryIdx),
                                                     trainFrame->features2D().at(match.trainIdx)->feature3D()));
            }
        }
        if (allCorr2D3D.size() < k_minCorrespondences2D3D)
        {
            return;
        }

        // find camera pose from P3P RANSAC
        Eigen::Matrix4d H;
        std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > inliers;
        radialpose::Camera cam_param;
        if (m_options.calibMode == "InRaSU")
            solvePnPRadialRansac(frame, allCorr2D3D, H, inliers);
        else
            solvePnPRansac(frame, allCorr2D3D, H, inliers, cam_param);

        int nInliers = inliers.size();

        if (nInliers < k_minCorrespondences2D3D)
        {
            return;
        }

        bestCorr2D3D.resize(inliers.size());
        bestCorr2D3D = inliers;
        bestInlierCount = nInliers;
        bestH = H;

        if (m_options.calibMode == "InRa" || m_options.calibMode == "InRaS")
        {
            best_cam_param = cam_param;
        }
    }

    if (bestInlierCount < k_minCorrespondences2D3D)
    {
        return;
    }

    if (m_verbose)
    {
        std::cout << "# INFO: [Cam " << frame->cameraId() <<  "] Found " << bestInlierCount
                  << " inlier 2D-3D correspondences from nearest image."
                  << std::endl;
    }

    if (m_options.calibMode == "InRa" || m_options.calibMode == "InRaS")
    {
        const CameraPtr& camera = m_cameraSystem.getCamera(frame->cameraId());
        const CameraPtr& camera_dummy = infrascal::CameraFactory::instance()->generateCamera(
                camera->modelType(), "camera_dummy",
                cv::Size(camera->imageWidth(), camera->imageHeight()));
        std::vector<double> cxcy(2);
        camera->getPrinciplePoint(cxcy);
        camera_dummy->setPrinciplePoint(cxcy);
        camera_dummy->setFocalLength({best_cam_param.focal, best_cam_param.focal});
        if (camera_dummy->modelType() == Camera::PINHOLE)
        {
            std::vector<double> parameterVec(camera_dummy->parameterCount());
            camera_dummy->writeParameters(parameterVec);
            parameterVec[0] = best_cam_param.dist_params[0];//k1
            parameterVec[1] = best_cam_param.dist_params[1];//k2
            camera_dummy->readParameters(parameterVec);
        }

        // intrinsics
        std::vector<double> intrinsicParams(camera_dummy->parameterCount());
        int camid = frame->cameraId();
        camera_dummy->writeParameters(intrinsicParams);
        double t3_offset = 0;

        // optimize intrinsics
        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 1000;
        options.num_threads = 8;

        Eigen::Matrix4d T_world_cam = bestH;

        for (size_t k = 0; k < bestCorr2D3D.size(); ++k) {
            Eigen::Vector3d P2 = bestCorr2D3D.at(k).second->point();
            Eigen::Vector3d P1 = transformPoint(T_world_cam, P2);
            const Point2DFeatureConstPtr &f1 = bestCorr2D3D.at(k).first;

            ceres::CostFunction* costFunction =
                    CostFunctionFactory::instance()->generateCostFunction(m_cameraSystem.getCamera(camid),
                                                                          P1,
                                                                          Eigen::Vector2d(f1->keypoint().pt.x, f1->keypoint().pt.y),
                                                                          Eigen::Vector2d(cxcy[0], cxcy[1]));

            ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);
            problem.AddResidualBlock(costFunction, lossFunction,
                                     intrinsicParams.data(), &t3_offset);

        }
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (bestInlierCount > m_inlierCounts_best.at(camid))
        {
            m_cameraSystem.getCamera(camid)->readParameters(intrinsicParams);
            m_inlierCounts_best.at(camid) = bestInlierCount;
        }

        T_world_cam(2,3) = T_world_cam(2,3) + t3_offset;
        bestH = T_world_cam;

    }

    if (m_options.calibMode == "InRaSU") {

        // BA on each frame

        // extrinsics
        Pose T_world_cam(bestH);

        // optimize extrinsics based on radial reprojection error
        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 1000;
        options.num_threads = 8;

        const CameraConstPtr &camera = m_cameraSystem.getCamera(frame->cameraId());
        std::vector<double> cxcy(2);
        camera->getPrinciplePoint(cxcy);

        for (size_t k = 0; k < bestCorr2D3D.size(); ++k) {
            Point2DFeaturePtr &feature2D = bestCorr2D3D.at(k).first;
            Point3DFeaturePtr &feature3D = bestCorr2D3D.at(k).second;

            ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
            ceres::CostFunction *costFunction
                    = CostFunctionFactory::instance()->radialPoseCostFunction2(
                            cxcy[0], cxcy[1],
                            feature3D->point(),
                            Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y));

            problem.AddResidualBlock(costFunction, lossFunction,
                                     T_world_cam.rotationData(),
                                     T_world_cam.translationData());
        }


        ceres::LocalParameterization *quaternionParameterization =
                new EigenQuaternionParameterization;

        problem.SetParameterization(T_world_cam.rotationData(), quaternionParameterization);


        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        T_world_cam.translation()(2) = 0;
        bestH = T_world_cam.toMatrix();
    }

    PosePtr pose = boost::allocate_shared<Pose>(Eigen::aligned_allocator<Pose>(), bestH);
    pose->timeStamp() = timestamp;

    frame->cameraPose() = pose;

    // store inlier 2D-3D correspondences
    for (size_t i = 0; i < bestCorr2D3D.size(); ++i)
    {
        Point2DFeaturePtr& p2D = bestCorr2D3D.at(i).first;
        Point3DFeaturePtr& p3D = bestCorr2D3D.at(i).second;

        boost::lock_guard<boost::mutex> lock(m_feature3DMapMutex);
        boost::unordered_map<Point3DFeature*, Point3DFeaturePtr>::iterator it = m_feature3DMap.find(p3D.get());

        Point3DFeaturePtr feature3D;
        if (it == m_feature3DMap.end())
        {
            feature3D = boost::make_shared<Point3DFeature>();
            feature3D->point() = p3D->point();
            feature3D->attributes() = p3D->attributes();

            m_feature3DMap.insert(std::make_pair(p3D.get(), feature3D));
        }
        else
        {
            feature3D = it->second;
        }

        feature3D->features2D().push_back(p2D);
        p2D->feature3D() = feature3D;
    }

    // prune features that are not associated to a scene point
    std::vector<Point2DFeaturePtr>::iterator it = frame->features2D().begin();
    while (it != frame->features2D().end())
    {
        if ((*it)->feature3D().get() == 0)
        {
            frame->features2D().erase(it);
        }
        else
        {
            ++it;
        }
    }

    if (m_verbose)
    {
        std::cout << "# INFO: [Cam " << frame->cameraId() <<  "] Estimated camera pose" << std::endl;
        std::cout << bestH << std::endl;
        std::cout << "           time: " << timeInSeconds() - tsStart << " s" << std::endl;

        double minError, maxError, avgError;
        size_t featureCount;

        if (m_options.calibMode == "InRaSU"){
            frameRadialReprojectionError(frame, m_cameraSystem.getCamera(frame->cameraId()),
                                         minError, maxError, avgError, featureCount);
            std::cout << "   radial reproj: " << avgError << std::endl;
        } else{
            frameReprojectionError(frame, m_cameraSystem.getCamera(frame->cameraId()),
                                   minError, maxError, avgError, featureCount);
            std::cout << "          reproj: " << avgError << std::endl;
        }
        std::cout << "              ts: " << pose->timeStamp() << std::endl;
    }
}

const CameraSystem&
InfrastructureCalibration::cameraSystem() const
{
    return m_cameraSystem;
}

void
InfrastructureCalibration::prune(int flags, int poseType)
{
    std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_odo(m_cameraSystem.cameraCount());
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > H_odo_cam(m_cameraSystem.cameraCount());
    for (int i = 0; i < m_cameraSystem.cameraCount(); ++i)
    {
        T_cam_odo.at(i) = m_cameraSystem.getGlobalCameraPose(i);

        H_odo_cam.at(i) = T_cam_odo.at(i).toMatrix().inverse();
    }

    if (m_verbose)
    {
        double minError, maxError, avgError;
        size_t featureCount;
        reprojectionError(minError, maxError, avgError, featureCount);

        std::cout << "# INFO: Initial reprojection error: avg = " << avgError
                << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
    }

    // prune points that are too far away or behind a camera

    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        FrameSet& frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            FramePtr& frame = frameset.frames.at(j);

            if (!frame)
            {
                continue;
            }

            int cameraId = frame->cameraId();

            std::vector<Point2DFeaturePtr>& features2D = frame->features2D();

            Eigen::Matrix4d H_cam = Eigen::Matrix4d::Identity();
            if (poseType == CAMERA)
            {
                H_cam = frame->cameraPose()->toMatrix();
            }
            else
            {
                H_cam = H_odo_cam.at(cameraId) * frame->systemPose()->toMatrix().inverse();
            }

            for (size_t l = 0; l < features2D.size(); ++l)
            {
                Point2DFeaturePtr& pf = features2D.at(l);

                if (!pf->feature3D())
                {
                    continue;
                }

                Eigen::Vector3d P_cam = transformPoint(H_cam, pf->feature3D()->point());

                bool prune = false;

                if ((flags & PRUNE_BEHIND_CAMERA) &&
                    P_cam(2) < 0.0)
                {
                    prune = true;
                }

                if ((flags & PRUNE_FARAWAY) &&
                    P_cam.block<3,1>(0,0).norm() > k_maxPoint3DDistance)
                {
                    prune = true;
                }

                if ((flags & PRUNE_FARAWAY) &&
                        (P_cam.block<2,1>(0,0)/P_cam(2)).norm() > k_maxPoint3DDistance/10.0)
                {
                    prune = true;
                }


                if (flags & PRUNE_HIGH_REPROJ_ERR)
                {
                    double error = 0.0;

                    if (poseType == CAMERA)
                    {
                        error = m_cameraSystem.getCamera(cameraId)->reprojectionError(pf->feature3D()->point(),
                                                                                      frame->cameraPose()->rotation(),
                                                                                      frame->cameraPose()->translation(),
                                                                                      Eigen::Vector2d(pf->keypoint().pt.x, pf->keypoint().pt.y));
                    }
                    else
                    {
                        error = reprojectionError(m_cameraSystem.getCamera(cameraId),
                                                  pf->feature3D()->point(),
                                                  T_cam_odo.at(cameraId).rotation(),
                                                  T_cam_odo.at(cameraId).translation(),
                                                  frame->systemPose()->position(),
                                                  frame->systemPose()->attitude(),
                                                  Eigen::Vector2d(pf->keypoint().pt.x, pf->keypoint().pt.y));
                    }

                    if (error > k_maxReprojErr)
                    {
                        prune = true;
                    }
                }

                if (prune)
                {
                    // delete entire feature track
                    std::vector<Point2DFeatureWPtr> features2D = pf->feature3D()->features2D();

                    for (size_t m = 0; m < features2D.size(); ++m)
                    {
                        if (Point2DFeaturePtr feature2D = features2D.at(m).lock())
                        {
                            feature2D->feature3D() = Point3DFeaturePtr();
                        }
                    }
                }
            }
        }
    }

    if (m_verbose)
    {
        double minError, maxError, avgError;
        size_t featureCount;
        reprojectionError(minError, maxError, avgError, featureCount);

        std::cout << "# INFO: Reprojection error after pruning: avg = " << avgError
                << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
    }
    
}

void
InfrastructureCalibration::optimizeRadialRigpose() {
    // extrinsics
    std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref(m_cameras.size());
    for (size_t i = 0; i < m_cameras.size(); ++i) {
        T_cam_ref.at(i) = Pose(m_cameraSystem.getGlobalCameraPose(i));
    }


    // optimize extrinsics based on transformation error
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 1000;
    options.num_threads = 8;


    for (size_t i = 0; i < m_framesets.size(); ++i) {
        FrameSet &frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j) {
            FramePtr &frame = frameset.frames.at(j);
            int cameraId = frame->cameraId();
            Pose T_world_cam = *(frame->cameraPose());
            ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
            ceres::CostFunction *costFunction
                    = CostFunctionFactory::instance()->radialPoseCostFunction(T_world_cam.rotation(),
                                                                              T_world_cam.translation());

            problem.AddResidualBlock(costFunction, lossFunction,
                                     T_cam_ref.at(frame->cameraId()).rotationData(),
                                     T_cam_ref.at(frame->cameraId()).translationData(),
                                     frame->systemPose()->positionData(),
                                     frame->systemPose()->attitudeData());
        }
    }

    for (size_t i = 0; i < m_cameras.size(); ++i) {
        ceres::LocalParameterization *quaternionParameterization =
                new EigenQuaternionParameterization;

        problem.SetParameterization(T_cam_ref.at(i).rotationData(), quaternionParameterization);
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for (size_t i = 0; i < m_cameras.size(); ++i) {
        Eigen::Matrix4d T_ref_cam = T_cam_ref.at(i).toMatrix().inverse();
        T_ref_cam(2,3) = 0;
        m_cameraSystem.setGlobalCameraPose(i, T_ref_cam.inverse());
    }
}

void
InfrastructureCalibration::optimizeRadialRigposeBA()
{
    // extrinsics
    std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref(m_cameras.size());
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        T_cam_ref.at(i) = Pose(m_cameraSystem.getGlobalCameraPose(i));
    }

    std::vector<std::vector<double>> cxcys(m_cameras.size(), std::vector<double>(2));
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        m_cameraSystem.getCamera(i)->getPrinciplePoint(cxcys.at(i));
    }
    // optimize extrinsics based on radial reprojection error
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 1000;
    options.num_threads = 8;


    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        FrameSet& frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            FramePtr& frame = frameset.frames.at(j);
            const CameraConstPtr& camera = m_cameraSystem.getCamera(frame->cameraId());

            for (size_t k = 0; k < frame->features2D().size(); ++k) {
                Point2DFeaturePtr &feature2D = frame->features2D().at(k);
                if (feature2D->feature3D().get() == 0) {
                    continue;
                }
                bool optimize_cxy = true;
                if(optimize_cxy) {
                    ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
                    ceres::CostFunction *costFunction
                            = CostFunctionFactory::instance()->radialPoseCostFunction(
                                    feature2D->feature3D()->point(),
                                    Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y));

                    problem.AddResidualBlock(costFunction, lossFunction,
                                             T_cam_ref.at(frame->cameraId()).rotationData(),
                                             T_cam_ref.at(frame->cameraId()).translationData(),
                                             frame->systemPose()->positionData(),
                                             frame->systemPose()->attitudeData(),
                                             &(cxcys.at(frame->cameraId()).at(0)),
                                             &(cxcys.at(frame->cameraId()).at(1)));
                }
                else{
                    ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
                    ceres::CostFunction *costFunction
                            = CostFunctionFactory::instance()->radialPoseCostFunction(camera->imageWidth(), camera->imageHeight(),
                                                                                      feature2D->feature3D()->point(),
                                    Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y));

                    problem.AddResidualBlock(costFunction, lossFunction,
                                             T_cam_ref.at(frame->cameraId()).rotationData(),
                                             T_cam_ref.at(frame->cameraId()).translationData(),
                                             frame->systemPose()->positionData(),
                                             frame->systemPose()->attitudeData());
                }
            }
        }
    }

    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        ceres::LocalParameterization* quaternionParameterization =
                new EigenQuaternionParameterization;

        problem.SetParameterization(T_cam_ref.at(i).rotationData(), quaternionParameterization);
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        Eigen::Matrix4d T_ref_cam = T_cam_ref.at(i).toMatrix().inverse();
        T_ref_cam(2,3) = 0;
        m_cameraSystem.setGlobalCameraPose(i, T_ref_cam.inverse());
        m_cameraSystem.getCamera(i)->setPrinciplePoint(cxcys.at(i));
    }
}

void
InfrastructureCalibration::upgradeRadialCamera()
{
    //prepare feature2d indices
    std::vector<std::vector<Point2DFeaturePtr>> feature2dIndices(m_cameras.size());

    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        FrameSet& frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            FramePtr& frame = frameset.frames.at(j);
            const CameraConstPtr& camera = m_cameraSystem.getCamera(frame->cameraId());

            for (size_t k = 0; k < frame->features2D().size(); ++k) {
                Point2DFeaturePtr &feature2D = frame->features2D().at(k);
                if (feature2D->feature3D().get() == 0) {
                    continue;
                }
                feature2dIndices.at(frame->cameraId()).push_back(feature2D);
            }
        }
    }

    // intrinsics
    std::vector<std::vector<double> > intrinsicParams(m_cameraSystem.cameraCount());

    // extrinsics
    std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref(m_cameras.size());
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        T_cam_ref.at(i) = Pose(m_cameraSystem.getGlobalCameraPose(i));
    }

    for (int camid = 0; camid < m_cameras.size(); ++camid)
    {
        double p = 0.999; // probability that at least one set of random samples does not contain an outlier
        double v = 0.7; // probability of observing an outlier

        double u = 1.0 - v;
        int N = static_cast<int>(log(1.0 - p) / log(1.0 - u * u * u) + 0.5);

        Eigen::Matrix<double, 2, Eigen::Dynamic> x;
        Eigen::Matrix<double, 3, Eigen::Dynamic> X;
        x.resize(2, 5);
        X.resize(3, 5);
        std::vector<Point2DFeaturePtr> indices = feature2dIndices.at(camid);
        const CameraConstPtr& camera1 = m_cameraSystem.getCamera(camid);
        std::vector<double> cxcy(2);
        m_cameraSystem.getCamera(camid)->getPrinciplePoint(cxcy);

        // run RANSAC to find best H
        Eigen::Matrix4d T_ref_cam_best;
        radialpose::Camera cam_param_best;
        std::vector<Point2DFeaturePtr> inlierIds_best;
        for (int i = 0; i < N; ++i) {
            std::random_shuffle(indices.begin(), indices.end());

            for (int j = 0; j < 5; ++j) {

                //transform X to camera frame with cam_T_ref and ref_T_world
                const FramePtr& frame = indices.at(j)->frame().lock();
                Eigen::Matrix4d T_world_ref = frame->systemPose()->toMatrix().inverse();
                Eigen::Matrix4d T_ref_cam = T_cam_ref.at(camid).toMatrix().inverse();
                Eigen::Vector3d P_w = indices.at(j)->feature3D()->point();
                Eigen::Vector3d P_ref = T_world_ref.block<3,3>(0,0)*P_w;
                P_ref = P_ref + T_world_ref.block<3,1>(0,3);
                Eigen::Vector3d P_c = T_ref_cam.block<3,3>(0,0)*P_ref;
                P_c(0) = P_c(0) + T_ref_cam(0,3);
                P_c(1) = P_c(1) + T_ref_cam(1,3);

                X.block<3, 1>(0, j) = P_c;

                const cv::KeyPoint &kpt1 = indices.at(j)->keypoint();

                x.block<2, 1>(0, j) = Eigen::Vector2d(kpt1.pt.x - cxcy[0],
                                                      kpt1.pt.y - cxcy[1]);
            }

            radialpose::larsson_iccv19::Solver<2, 0, true> estimator_radtan;
            radialpose::kukelova_iccv13::Solver estimator_equi(2);
            estimator_radtan.use_radial_solver = false;
            estimator_radtan.normalize_world_coord = true;
            estimator_radtan.center_world_coord = false;
            estimator_radtan.check_chirality = false;
            estimator_radtan.check_reprojection_error = false;
            estimator_equi.use_radial_solver = false;
            estimator_equi.normalize_world_coord = true;
            estimator_equi.center_world_coord = false;
            estimator_equi.check_chirality = false;
            estimator_equi.check_reprojection_error = false;

            std::vector<radialpose::Camera> poses;

            if (camera1->modelType() == Camera::PINHOLE)
                int n_sols = estimator_radtan.estimate(x, X, &poses);
            else if (camera1->modelType() == Camera::KANNALA_BRANDT)
                int n_sols = estimator_equi.estimate(x, X, &poses);
            else
                int n_sols = estimator_radtan.estimate(x, X, &poses);

            for (int j = 0; j < poses.size(); j++) {
                std::vector<Point2DFeaturePtr> inliersIds;
                Eigen::Matrix4d T_ref_cam = T_cam_ref.at(camid).toMatrix().inverse();
                T_ref_cam(2,3)=poses[j].t(2);

                for (size_t k = 0; k < indices.size(); ++k) {
                    //transform X to camera frame with cam_T_ref and ref_T_world
                    const FramePtr& frame = indices.at(k)->frame().lock();
                    Eigen::Matrix4d T_world_ref = frame->systemPose()->toMatrix().inverse();
                    Eigen::Matrix4d T_world_cam = T_ref_cam * T_world_ref;

                    Eigen::Vector3d P2 = indices.at(k)->feature3D()->point();
                    Eigen::Vector3d P1 = transformPoint(T_world_cam, P2);
                    Eigen::Vector2d p1_pred(P1[0] / P1[2], P1[1] / P1[2]);
                    Eigen::Matrix<double, 2, Eigen::Dynamic> p1_pred_;
                    p1_pred_.resize(2, 1);
                    p1_pred_.block<2, 1>(0, 0) = p1_pred;

                    if (camera1->modelType() == Camera::PINHOLE)
                        radialpose::forward_rational_model(poses[j].dist_params, 2, 0, p1_pred_, &p1_pred_);
                    else if (camera1->modelType() == Camera::KANNALA_BRANDT)
                        radialpose::inverse_rational_model(poses[j].dist_params, 0, 2, p1_pred_, &p1_pred_);
                    else
                        radialpose::forward_rational_model(poses[j].dist_params, 2, 0, p1_pred_, &p1_pred_);

                    p1_pred = p1_pred_.block<2, 1>(0, 0);
                    p1_pred(0) = p1_pred(0) * poses[j].focal + cxcy[0];
                    p1_pred(1) = p1_pred(1) * poses[j].focal + cxcy[1];
                    const Point2DFeatureConstPtr &f1 = indices.at(k);

                    double err = hypot(f1->keypoint().pt.x - p1_pred(0),
                                       f1->keypoint().pt.y - p1_pred(1));
                    if (!isnormal(err) || err > k_reprojErrorThresh*5) {
                        continue;
                    }

                    inliersIds.push_back(indices.at(k));
                }
                if (inliersIds.size() > inlierIds_best.size()) {
                    cam_param_best = poses[j];
                    T_ref_cam_best = T_ref_cam;
                    inlierIds_best = inliersIds;

                }

            }
        }//ransac

        // remove outliers
//        for (size_t k = 0; k < indices.size(); ++k) {
//            if (std::find(inlierIds_best.begin(),inlierIds_best.end(), indices.at(k)) ==inlierIds_best.end()){
//                indices.at(k)->feature3D() = Point3DFeaturePtr();
//            }
//        }

        m_cameraSystem.setGlobalCameraPose(camid, T_ref_cam_best.inverse());

        const CameraPtr& camera = m_cameraSystem.getCamera(camid);
        camera->setFocalLength({cam_param_best.focal, cam_param_best.focal});
        if (camera->modelType() == Camera::PINHOLE)
        {
            std::vector<double> parameterVec(camera->parameterCount());
            camera->writeParameters(parameterVec);
            parameterVec[0] = cam_param_best.dist_params[0];//k1
            parameterVec[1] = cam_param_best.dist_params[1];//k2
            camera->readParameters(parameterVec);
        }
    }
//    boost::filesystem::path cameraSystemBeforeBAPath(m_options.outputDir);
//    cameraSystemBeforeBAPath /= "infrastr_beforeBA.xml";
//    m_cameraSystem.writeToXmlFile(cameraSystemBeforeBAPath.string());
//    boost::filesystem::path extrinsics_init(m_options.outputDir);
//    extrinsics_init /= "extrinsic_init.txt";
//    m_cameraSystem.writePosesToTextFile(extrinsics_init.string());
}

void
InfrastructureCalibration::upgradeOptiRadialCamera()
{
    //prepare feature2d indices
    std::vector<std::vector<Point2DFeaturePtr>> feature2dIndices(m_cameras.size());

    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        FrameSet& frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            FramePtr& frame = frameset.frames.at(j);
            const CameraConstPtr& camera = m_cameraSystem.getCamera(frame->cameraId());

            for (size_t k = 0; k < frame->features2D().size(); ++k) {
                Point2DFeaturePtr &feature2D = frame->features2D().at(k);
                if (feature2D->feature3D().get() == 0) {
                    continue;
                }
                feature2dIndices.at(frame->cameraId()).push_back(feature2D);
            }
        }
    }

    // intrinsics
    std::vector<std::vector<double> > intrinsicParams(m_cameraSystem.cameraCount());

    // extrinsics
    std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref(m_cameras.size());
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        T_cam_ref.at(i) = Pose(m_cameraSystem.getGlobalCameraPose(i));
    }

    for (int camid = 0; camid < m_cameras.size(); ++camid) {
        std::vector<Point2DFeaturePtr> indices = feature2dIndices.at(camid);
        const CameraConstPtr& camera = m_cameraSystem.getCamera(camid);
        camera->writeParameters(intrinsicParams[camid]);
        Eigen::Matrix4d T_ref_cam = T_cam_ref.at(camid).toMatrix().inverse();
        double t3_offset = 0;
        // optimize intrinsics
        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 1000;
        options.num_threads = 8;


        for (size_t k = 0; k < indices.size(); ++k) {
            //transform X to camera frame with cam_T_ref and ref_T_world
            const FramePtr& frame = indices.at(k)->frame().lock();
            Eigen::Matrix4d T_world_ref = frame->systemPose()->toMatrix().inverse();
            Eigen::Matrix4d T_world_cam = T_ref_cam * T_world_ref;

            Eigen::Vector3d P2 = indices.at(k)->feature3D()->point();
            Eigen::Vector3d P1 = transformPoint(T_world_cam, P2);
            const Point2DFeatureConstPtr &f1 = indices.at(k);

            ceres::CostFunction* costFunction =
                    CostFunctionFactory::instance()->generateCostFunction(m_cameraSystem.getCamera(camid),
                                                                          P1,
                                                                          Eigen::Vector2d(f1->keypoint().pt.x, f1->keypoint().pt.y),
                                                                          CAMERA_INTRINSICS | PRINCIPLE_TRANSLATION);

            ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);
            problem.AddResidualBlock(costFunction, lossFunction,
                                     intrinsicParams[camid].data(), &t3_offset);

        }
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        m_cameraSystem.getCamera(camid)->readParameters(intrinsicParams[camid]);

        T_ref_cam(2,3) = T_ref_cam(2,3) + t3_offset;
        m_cameraSystem.setGlobalCameraPose(camid, T_ref_cam.inverse());

    }
//    boost::filesystem::path cameraSystemBeforeBAPath(m_options.outputDir);
//    cameraSystemBeforeBAPath /= "infrastr_beforeBA_upgradeba.xml";
//    m_cameraSystem.writeToXmlFile(cameraSystemBeforeBAPath.string());
//    boost::filesystem::path extrinsics_init(m_options.outputDir);
//    extrinsics_init /= "extrinsic_init_upgradeba.txt";
//    m_cameraSystem.writePosesToTextFile(extrinsics_init.string());
}

void
InfrastructureCalibration::optimize(int flags)
{
    if(m_options.verbose)
    {
        if (flags & POINT_3D)
        {
            std::cout << "# INFO: Optimize camera poses, odometry and scene points." << std::endl;
        } else
        {
            std::cout << "# INFO: Optimize camera poses, odometry." << std::endl;
        }
        if (flags & CAMERA_INTRINSICS)
        {
            std::cout << "# INFO: Optimize camera intrinsics as well." << std::endl;
        }
    }

    // intrinsics
    /// @todo vec<vec<>> is slow! consider alternatives like boost::static_vector multiarray, or even an eigen matrix
    std::vector<std::vector<double> > intrinsicParams(m_cameraSystem.cameraCount());

    for (int i = 0; i < m_cameraSystem.cameraCount(); ++i)
    {
        m_cameraSystem.getCamera(i)->writeParameters(intrinsicParams[i]);
    }
    std::vector<double> intrinsicParams_ref;
    m_cameraSystem.getCamera(0)->writeParameters(intrinsicParams_ref);


    // extrinsics
    std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref(m_cameras.size());
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        T_cam_ref.at(i) = Pose(m_cameraSystem.getGlobalCameraPose(i));
    }
    Pose T_rig_ref = Pose(m_cameraSystem.getGlobalCameraPose(0));

    if (m_verbose) {
        double minError, maxError, avgError;
        size_t featureCount;
        reprojectionError(minError, maxError, avgError, featureCount);
        std::cout << "# INFO: Reprojection error: avg = " << avgError
                  << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
    }

    double tsStart = timeInSeconds();

    ceres::Problem problem;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 1000;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = false;

    boost::dynamic_bitset<> optimizeExtrinsics(m_cameraSystem.cameraCount());

    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        FrameSet& frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            FramePtr& frame = frameset.frames.at(j);
            int cameraId = frame->cameraId();

            for (size_t k = 0; k < frame->features2D().size(); ++k)
            {
                Point2DFeaturePtr& feature2D = frame->features2D().at(k);

                if (feature2D->feature3D().get() == 0)
                {
                    continue;
                }

                optimizeExtrinsics[cameraId] = 1;

                ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);
                switch (flags)
                {
                case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
                {
                    ceres::CostFunction* costFunction
                        = CostFunctionFactory::instance()->generateCostFunction(m_cameraSystem.getCamera(frame->cameraId()),
                                                                                Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y),
                                                                                flags);

                    problem.AddResidualBlock(costFunction, lossFunction,
                                             T_cam_ref.at(frame->cameraId()).rotationData(),
                                             T_cam_ref.at(frame->cameraId()).translationData(),
                                             frame->systemPose()->positionData(),
                                             frame->systemPose()->attitudeData(),
                                             feature2D->feature3D()->pointData());

                    break;
                }
                case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE:
                {
                    ceres::CostFunction* costFunction
                        = CostFunctionFactory::instance()->generateCostFunction(m_cameraSystem.getCamera(frame->cameraId()),
                                                                                feature2D->feature3D()->point(),
                                                                                Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y),
                                                                                flags);

                    problem.AddResidualBlock(costFunction, lossFunction,
                                             T_cam_ref.at(frame->cameraId()).rotationData(),
                                             T_cam_ref.at(frame->cameraId()).translationData(),
                                             frame->systemPose()->positionData(),
                                             frame->systemPose()->attitudeData());

                    break;
                }
                case CAMERA_INTRINSICS | CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
                {
                    ceres::CostFunction* costFunction
                        = CostFunctionFactory::instance()->generateCostFunction(m_cameraSystem.getCamera(frame->cameraId()),
                                                                                Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y),
                                                                                flags);

                    problem.AddResidualBlock(costFunction, lossFunction,
                                             intrinsicParams[frame->cameraId()].data(),
                                             T_cam_ref.at(frame->cameraId()).rotationData(),
                                             T_cam_ref.at(frame->cameraId()).translationData(),
                                             frame->systemPose()->positionData(),
                                             frame->systemPose()->attitudeData(),
                                             feature2D->feature3D()->pointData());

                    break;
                }
                case CAMERA_INTRINSICS | CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE:
                {
                    ceres::CostFunction* costFunction
                        = CostFunctionFactory::instance()->generateCostFunction(m_cameraSystem.getCamera(frame->cameraId()),
                                                                                feature2D->feature3D()->point(),
                                                                                Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y),
                                                                                flags);

                    problem.AddResidualBlock(costFunction, lossFunction,
                                             intrinsicParams[frame->cameraId()].data(),
                                             T_cam_ref.at(frame->cameraId()).rotationData(),
                                             T_cam_ref.at(frame->cameraId()).translationData(),
                                             frame->systemPose()->positionData(),
                                             frame->systemPose()->attitudeData());

                    break;
                }

                }
            }
        }
    }

    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        if (optimizeExtrinsics[i])
        {
            ceres::LocalParameterization* quaternionParameterization =
                    new EigenQuaternionParameterization;

            problem.SetParameterization(T_cam_ref.at(i).rotationData(), quaternionParameterization);
        }
        else
        {
            std::cout<<"# WARN: Camera id "<<i<<" has no frame associated, thus not constrained."<<std::endl;
        }
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (m_verbose)
    {
        std::cout << "# INFO: Optimization took "
                  << timeInSeconds() - tsStart << " s." << std::endl;
    }

    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        m_cameraSystem.setGlobalCameraPose(i, T_cam_ref.at(i).toMatrix());
    }

    if (flags & CAMERA_INTRINSICS)
    {
        for (int i = 0; i < m_cameraSystem.cameraCount(); ++i)
        {
            m_cameraSystem.getCamera(i)->readParameters(intrinsicParams[i]);
        }
    }


    if (m_verbose) {
        double minError, maxError, avgError;
        size_t featureCount;
        reprojectionError(minError, maxError, avgError, featureCount);
        std::cout << "# INFO: Reprojection error: avg = " << avgError
                  << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
    }
}


void
InfrastructureCalibration::optimizeIntrinsics()
{

    std::cout << "# INFO: Optimize camera intrinsics" <<std::endl;

    // intrinsics
    /// @todo vec<vec<>> is slow! consider alternatives like boost::static_vector multiarray, or even an eigen matrix
    std::vector<std::vector<double> > intrinsicParams(m_cameraSystem.cameraCount());

    for (int i = 0; i < m_cameraSystem.cameraCount(); ++i)
    {
        m_cameraSystem.getCamera(i)->writeParameters(intrinsicParams[i]);
    }

    // extrinsics
    std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref(m_cameras.size());
    for (size_t i = 0; i < m_cameras.size(); ++i)
    {
        T_cam_ref.at(i) = Pose(m_cameraSystem.getGlobalCameraPose(i));
    }

    if (m_verbose)
    {
        double minError, maxError, avgError;
        size_t featureCount;
        reprojectionError(minError, maxError, avgError, featureCount);

        std::cout << "# INFO: Initial reprojection error: avg = " << avgError
                  << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
    }

    double tsStart = timeInSeconds();

    ceres::Problem problem;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 1000;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = false;


    boost::dynamic_bitset<> optimizeExtrinsics(m_cameraSystem.cameraCount());

    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        FrameSet& frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            FramePtr& frame = frameset.frames.at(j);
            int cameraId = frame->cameraId();
            Eigen::Matrix4d T_world_ref = frame->systemPose()->toMatrix().inverse();
            Eigen::Matrix4d T_world_cam = T_cam_ref.at(cameraId).toMatrix().inverse() * T_world_ref;
            PosePtr pose = boost::make_shared<Pose>(T_world_cam);
            pose->timeStamp() = frame->cameraPose()->timeStamp();

            frame->cameraPose() = pose;
            for (size_t k = 0; k < frame->features2D().size(); ++k)
            {
                Point2DFeaturePtr& feature2D = frame->features2D().at(k);

                if (feature2D->feature3D().get() == 0)
                {
                    continue;
                }


                ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);
                ceres::CostFunction* costFunction
                        = CostFunctionFactory::instance()->generateCostFunction(m_cameraSystem.getCamera(frame->cameraId()),
                                                                                feature2D->feature3D()->point(),
                                                                                Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y),
                                                                                CAMERA_INTRINSICS | CAMERA_POSE);

                problem.AddResidualBlock(costFunction, lossFunction,
                                         intrinsicParams[frame->cameraId()].data(),
                                         frame->cameraPose()->rotationData(),
                                         frame->cameraPose()->translationData());
            }

            ceres::LocalParameterization* quaternionParameterization =
                    new EigenQuaternionParameterization;
            problem.SetParameterization(frame->cameraPose()->rotationData(), quaternionParameterization);

        }
    }
    for (int i = 0; i < m_cameraSystem.cameraCount(); ++i) {
        std::cout << "intrinsics for camera before" << i << std::endl;
        for (double s : intrinsicParams[i])
            std::cout << s << std::endl;
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (m_verbose)
    {
        std::cout << "# INFO: Optimization took "
                  << timeInSeconds() - tsStart << " s." << std::endl;
    }


    for (int i = 0; i < m_cameraSystem.cameraCount(); ++i)
    {
        m_cameraSystem.getCamera(i)->readParameters(intrinsicParams[i]);
        std::cout<<"intrinsics for camera "<<i<<std::endl;
        for(double s : intrinsicParams[i])
            std::cout<<s<<std::endl;
    }



    if (m_verbose)
    {
        double minError, maxError, avgError;
        size_t featureCount;
        reprojectionError(minError, maxError, avgError, featureCount);

        std::cout << "# INFO: Final reprojection error: avg = " << avgError
                  << " px | max = " << maxError << " px | count = " << featureCount << std::endl;
    }
}


cv::Mat
InfrastructureCalibration::buildDescriptorMat(const std::vector<Point2DFeaturePtr>& features,
                                              std::vector<size_t>& indices,
                                              bool hasScenePoint) const
{
    for (size_t i = 0; i < features.size(); ++i)
    {
        if (hasScenePoint && !features.at(i)->feature3D())
        {
            continue;
        }

        indices.push_back(i);
    }

    cv::Mat dtor(indices.size(), features.at(0)->descriptor().cols, features.at(0)->descriptor().type());

    for (size_t i = 0; i < indices.size(); ++i)
    {
         features.at(indices.at(i))->descriptor().copyTo(dtor.row(i));
    }

    return dtor;
}

std::vector<cv::DMatch>
InfrastructureCalibration::matchFeatures(const std::vector<Point2DFeaturePtr>& queryFeatures,
                                         const std::vector<Point2DFeaturePtr>& trainFeatures) const
{
    std::vector<size_t> queryIndices, trainIndices;
    cv::Mat queryDtor = buildDescriptorMat(queryFeatures, queryIndices, false);
    cv::Mat trainDtor = buildDescriptorMat(trainFeatures, trainIndices, true);

    if (queryDtor.cols != trainDtor.cols)
    {
        std::cout << "# WARNING: Descriptor lengths do not match." << std::endl;
        return std::vector<cv::DMatch>();
    }

    if (queryDtor.type() != trainDtor.type())
    {
        std::cout << "# WARNING: Descriptor types do not match." << std::endl;
        return std::vector<cv::DMatch>();
    }

    cv::Ptr<SurfGPU> surf = SurfGPU::instance();

    std::vector<std::vector<cv::DMatch> > candidateFwdMatches;
    surf->knnMatch(queryDtor, trainDtor, candidateFwdMatches, 2);

    std::vector<std::vector<cv::DMatch> > candidateRevMatches;
    surf->knnMatch(trainDtor, queryDtor, candidateRevMatches, 2);

    std::vector<std::vector<cv::DMatch> > fwdMatches(candidateFwdMatches.size());
    for (size_t i = 0; i < candidateFwdMatches.size(); ++i)
    {
        std::vector<cv::DMatch>& match = candidateFwdMatches.at(i);

        if (match.size() < 2)
        {
            continue;
        }

        float distanceRatio = match.at(0).distance / match.at(1).distance;

        if (distanceRatio < k_maxDistanceRatio)
        {
            fwdMatches.at(i).push_back(match.at(0));
        }
    }

    std::vector<std::vector<cv::DMatch> > revMatches(candidateRevMatches.size());
    for (size_t i = 0; i < candidateRevMatches.size(); ++i)
    {
        std::vector<cv::DMatch>& match = candidateRevMatches.at(i);

        if (match.size() < 2)
        {
            continue;
        }

        float distanceRatio = match.at(0).distance / match.at(1).distance;

        if (distanceRatio < k_maxDistanceRatio)
        {
            revMatches.at(i).push_back(match.at(0));
        }
    }

    // cross-check
    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < fwdMatches.size(); ++i)
    {
        if (fwdMatches.at(i).empty())
        {
            continue;
        }

        cv::DMatch& fwdMatch = fwdMatches.at(i).at(0);

        if (revMatches.at(fwdMatch.trainIdx).empty())
        {
            continue;
        }

        cv::DMatch& revMatch = revMatches.at(fwdMatch.trainIdx).at(0);

        if (fwdMatch.queryIdx == revMatch.trainIdx &&
            fwdMatch.trainIdx == revMatch.queryIdx)
        {
            cv::DMatch match;
            match.queryIdx = queryIndices.at(fwdMatch.queryIdx);
            match.trainIdx = trainIndices.at(revMatch.queryIdx);

            matches.push_back(match);
        }
    }

    return matches;
}


void
InfrastructureCalibration::solveP3PRansac(const FrameConstPtr& frame1,
                                          const std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > corr2D3Ds,
                                          Eigen::Matrix4d& H,
                                          std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> >& inliers) const
{
    inliers.clear();

    double p = 0.99; // probability that at least one set of random samples does not contain an outlier
    double v = 0.6; // probability of observing an outlier

    double u = 1.0 - v;
    int N = static_cast<int>(log(1.0 - p) / log(1.0 - u * u * u) + 0.5);

    std::vector<size_t> indices;
    for (size_t i = 0; i < corr2D3Ds.size(); ++i)
    {
        indices.push_back(i);
    }

    const CameraConstPtr& camera1 = m_cameraSystem.getCamera(frame1->cameraId());

    // run RANSAC to find best H
    Eigen::Matrix4d H_best;
    std::vector<size_t> inlierIds_best;
    for (int i = 0; i < N; ++i)
    {
        std::random_shuffle(indices.begin(), indices.end());

        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > rays(3);
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > worldPoints(3);
        for (int j = 0; j < 3; ++j)
        {
            const std::pair<Point2DFeaturePtr, Point3DFeaturePtr> corr2D3D = corr2D3Ds.at(indices.at(j));
            worldPoints.at(j) = corr2D3D.second->point();

            const cv::KeyPoint& kpt1 = corr2D3D.first->keypoint();

            Eigen::Vector3d ray;
            camera1->liftProjective(Eigen::Vector2d(kpt1.pt.x, kpt1.pt.y), ray);

            rays.at(j) = ray.normalized();
        }

        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > solutions;
        if (!solveP3P(rays, worldPoints, solutions))
        {
            continue;
        }

        for (size_t j = 0; j < solutions.size(); ++j)
        {
            Eigen::Matrix4d H_inv = solutions.at(j).inverse();

            std::vector<size_t> inliersIds;
            for (size_t k = 0; k < corr2D3Ds.size(); ++k)
            {
                const std::pair<Point2DFeaturePtr, Point3DFeaturePtr> corr2D3D = corr2D3Ds.at(k);

                Eigen::Vector3d P2 = corr2D3D.second->point();

                Eigen::Vector3d P1 = transformPoint(H_inv, P2);
                Eigen::Vector2d p1_pred;
                camera1->spaceToPlane(P1, p1_pred);

                const Point2DFeatureConstPtr& f1 = corr2D3D.first;

                double err = hypot(f1->keypoint().pt.x - p1_pred(0),
                                   f1->keypoint().pt.y - p1_pred(1));
                if (err > k_reprojErrorThresh)
                {
                    continue;
                }

                inliersIds.push_back(k);
            }

            if (inliersIds.size() > inlierIds_best.size())
            {
                H_best = H_inv;
                inlierIds_best = inliersIds;
            }
        }
    }

    for (size_t i = 0; i < inlierIds_best.size(); ++i)
    {
        inliers.push_back(corr2D3Ds.at(inlierIds_best.at(i)));
    }

    H = H_best;
}

void
InfrastructureCalibration::solvePnPRansac(const FrameConstPtr& frame1,
                                          const std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > corr2D3Ds,
                                          Eigen::Matrix4d& H,
                                          std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> >& inliers,
                                          radialpose::Camera& cam_param)
{

    inliers.clear();

    double p = 0.99; // probability that at least one set of random samples does not contain an outlier
    double v = 0.6; // probability of observing an outlier

    double u = 1.0 - v;
    int N = static_cast<int>(log(1.0 - p) / log(1.0 - u * u * u) + 0.5);

    std::vector<size_t> indices;
    for (size_t i = 0; i < corr2D3Ds.size(); ++i)
    {
        indices.push_back(i);
    }

    const CameraConstPtr& camera1 = m_cameraSystem.getCamera(frame1->cameraId());
    std::vector<double> cxcy(2);
    camera1->getPrinciplePoint(cxcy);

    Eigen::Matrix<double, 2, Eigen::Dynamic> x;
    Eigen::Matrix<double, 3, Eigen::Dynamic> X;
    x.resize(2, 5);
    X.resize(3, 5);

    // run RANSAC to find best H
    Eigen::Matrix4d H_best;
    radialpose::Camera cam_param_best;
    std::vector<size_t> inlierIds_best;
    Eigen::Matrix4d H_best_radial;
    std::vector<size_t> inlierIds_best_radial;
    for (int i = 0; i < N; ++i)
    {
        std::random_shuffle(indices.begin(), indices.end());

        for (int j = 0; j < 5; ++j)
        {
            const std::pair<Point2DFeaturePtr, Point3DFeaturePtr> corr2D3D = corr2D3Ds.at(indices.at(j));
            X.block<3,1>(0, j)  = corr2D3D.second->point();

            const cv::KeyPoint& kpt1 = corr2D3D.first->keypoint();
            x.block<2,1>(0, j) = Eigen::Vector2d(kpt1.pt.x-cxcy[0], kpt1.pt.y-cxcy[1]);
        }

        radialpose::larsson_iccv19::Solver<2, 0, true> estimator_radtan;
        radialpose::kukelova_iccv13::Solver estimator_equi(2);
        std::vector<radialpose::Camera> poses;

        if (camera1->modelType() == Camera::PINHOLE)
            int n_sols = estimator_radtan.estimate(x, X, &poses);
        else if (camera1->modelType() == Camera::KANNALA_BRANDT)
            int n_sols = estimator_equi.estimate(x, X, &poses);
        else
            int n_sols = estimator_radtan.estimate(x, X, &poses);

        for (int j = 0; j < poses.size(); j++){
            Eigen::Matrix4d H_inv;// C_T_W
            H_inv.block<3,3>(0,0) = poses[j].R;
            H_inv.block<3,1>(0,3) = poses[j].t;
            H_inv.block<1,4>(3,0) << 0,0,0,1;

            std::vector<size_t> inliersIds;
            for (size_t k = 0; k < corr2D3Ds.size(); ++k)
            {
                const std::pair<Point2DFeaturePtr, Point3DFeaturePtr> corr2D3D = corr2D3Ds.at(k);

                Eigen::Vector3d P2 = corr2D3D.second->point();

                Eigen::Vector3d P1 = transformPoint(H_inv, P2);
                Eigen::Vector2d p1_pred(P1[0]/P1[2], P1[1]/P1[2]);
                Eigen::Matrix<double, 2, Eigen::Dynamic> p1_pred_;
                p1_pred_.resize(2,1);
                p1_pred_.block<2,1>(0,0) = p1_pred;

                if (camera1->modelType() == Camera::PINHOLE)
                    radialpose::forward_rational_model(poses[j].dist_params, 2, 0, p1_pred_, &p1_pred_);
                else if (camera1->modelType() == Camera::KANNALA_BRANDT)
                    radialpose::inverse_rational_model(poses[j].dist_params, 0, 2, p1_pred_, &p1_pred_);
                else
                    radialpose::forward_rational_model(poses[j].dist_params, 2, 0, p1_pred_, &p1_pred_);

                p1_pred = p1_pred_.block<2,1>(0,0);
                p1_pred(0) = p1_pred(0)*poses[j].focal + cxcy[0];
                p1_pred(1) = p1_pred(1)*poses[j].focal + cxcy[1];
                const Point2DFeatureConstPtr& f1 = corr2D3D.first;

                double err = hypot(f1->keypoint().pt.x - p1_pred(0),
                                   f1->keypoint().pt.y - p1_pred(1));
                if (!isnormal(err) || err > k_reprojErrorThresh)
                {
                    continue;
                }

                inliersIds.push_back(k);
            }
            if (inliersIds.size() > inlierIds_best.size())
            {
                cam_param_best = poses[j];
                H_best = H_inv;
                inlierIds_best = inliersIds;

            }

        }
    }


    for (size_t i = 0; i < inlierIds_best.size(); ++i)
    {
        inliers.push_back(corr2D3Ds.at(inlierIds_best.at(i)));
    }

    H = H_best;
    cam_param = cam_param_best;

}

void
InfrastructureCalibration::solvePnPRadialRansac(const FrameConstPtr& frame1,
                                                const std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> > corr2D3Ds,
                                                Eigen::Matrix4d& H,
                                                std::vector<std::pair<Point2DFeaturePtr, Point3DFeaturePtr> >& inliers) const
{

    inliers.clear();

    double p = 0.99; // probability that at least one set of random samples does not contain an outlier
    double v = 0.6; // probability of observing an outlier

    double u = 1.0 - v;
    int N = static_cast<int>(log(1.0 - p) / log(1.0 - u * u * u) + 0.5);

    std::vector<size_t> indices;
    for (size_t i = 0; i < corr2D3Ds.size(); ++i)
    {
        indices.push_back(i);
    }

    const CameraConstPtr& camera1 = m_cameraSystem.getCamera(frame1->cameraId());
    std::vector<double> cxcy(2);
    camera1->getPrinciplePoint(cxcy);

    Eigen::Matrix<double, 2, Eigen::Dynamic> x;
    Eigen::Matrix<double, 3, Eigen::Dynamic> X;
    x.resize(2, 5);
    X.resize(3, 5);

    // run RANSAC to find best H
    Eigen::Matrix4d H_best_radial;
    std::vector<size_t> inlierIds_best_radial;
    for (int i = 0; i < N; ++i)
    {
        std::random_shuffle(indices.begin(), indices.end());

        for (int j = 0; j < 5; ++j)
        {
            const std::pair<Point2DFeaturePtr, Point3DFeaturePtr> corr2D3D = corr2D3Ds.at(indices.at(j));
            X.block<3,1>(0, j)  = corr2D3D.second->point();

            const cv::KeyPoint& kpt1 = corr2D3D.first->keypoint();
            x.block<2,1>(0, j) = Eigen::Vector2d(kpt1.pt.x-cxcy[0], kpt1.pt.y-cxcy[1]);
        }

        typedef radialpose::kukelova_iccv13::Radial1DSolver SOLVER_RADIAL;

        std::vector<radialpose::Camera> poses_radial;

        SOLVER_RADIAL estimator_radial;
        int n_sols_radial = estimator_radial.estimate(x, X, &poses_radial);
        poses_radial.insert(poses_radial.end(), poses_radial.begin(), poses_radial.end());
        for(int j = 0; j < n_sols_radial; ++j){
            poses_radial.at(n_sols_radial + j).R *= -1.0;
            poses_radial.at(n_sols_radial + j).t *= -1.0;
            poses_radial.at(n_sols_radial + j).R.row(2) *= -1.0;
            poses_radial.at(n_sols_radial + j).t.row(2) *= -1.0;
            poses_radial.at(j).t(2) = 0;
            poses_radial.at(n_sols_radial + j).t(2) = 0;
        }

        for (int j = 0; j < poses_radial.size(); j++){
            Eigen::Matrix4d H_inv;// C_T_W
            H_inv.block<3,3>(0,0) = poses_radial[j].R;
            H_inv.block<3,1>(0,3) = poses_radial[j].t;
            H_inv.block<1,4>(3,0) << 0,0,0,1;

            std::vector<size_t> inliersIds;
            for (size_t k = 0; k < corr2D3Ds.size(); ++k)
            {
                const std::pair<Point2DFeaturePtr, Point3DFeaturePtr> corr2D3D = corr2D3Ds.at(k);

                Eigen::Vector3d P2 = corr2D3D.second->point();

                Eigen::Vector3d P1 = transformPoint(H_inv, P2);
                Eigen::Vector2d p1_pred(P1[0]/P1[2], P1[1]/P1[2]);
                double p1_pred_norm = p1_pred.norm();
                p1_pred(0) = p1_pred(0)/p1_pred_norm;
                p1_pred(1) = p1_pred(1)/p1_pred_norm;
                const Point2DFeatureConstPtr& f1 = corr2D3D.first;
                Eigen::Vector2d p1_pixel(f1->keypoint().pt.x - cxcy[0],
                                         f1->keypoint().pt.y - cxcy[1]);
                if(p1_pixel(0) * P1[0] + p1_pixel(1) * P1[1] < 0)
                    continue;
                Eigen::Vector2d p1_pred_proj;
                p1_pred_proj(0) = p1_pred(0)*p1_pred(0)*p1_pixel(0)
                                  + p1_pred(0)*p1_pred(1)*p1_pixel(1);
                p1_pred_proj(1) = p1_pred(1)*p1_pred(1)*p1_pixel(1)
                                  + p1_pred(0)*p1_pred(1)*p1_pixel(0);
                double err = hypot(p1_pixel(0) - p1_pred_proj(0),
                                   p1_pixel(1) - p1_pred_proj(1));
                if (!isnormal(err) || err > k_reprojErrorThresh)
                {
                    continue;
                }

                inliersIds.push_back(k);
            }
            if (inliersIds.size() > inlierIds_best_radial.size())
            {
                H_best_radial = H_inv;
                inlierIds_best_radial = inliersIds;

            }
        }
    }

    for (size_t i = 0; i < inlierIds_best_radial.size(); ++i)
    {
        inliers.push_back(corr2D3Ds.at(inlierIds_best_radial.at(i)));
    }

    H = H_best_radial;
}

double
InfrastructureCalibration::reprojectionError(const CameraConstPtr& camera,
                                             const Eigen::Vector3d& P,
                                             const Eigen::Quaterniond& cam_ref_q,
                                             const Eigen::Vector3d& cam_ref_t,
                                             const Eigen::Vector3d& ref_p,
                                             const Eigen::Vector3d& ref_att,
                                             const Eigen::Vector2d& observed_p) const
{
    Eigen::Quaterniond q_z_inv(cos(ref_att(0) / 2.0), 0.0, 0.0, -sin(ref_att(0) / 2.0));
    Eigen::Quaterniond q_y_inv(cos(ref_att(1) / 2.0), 0.0, -sin(ref_att(1) / 2.0), 0.0);
    Eigen::Quaterniond q_x_inv(cos(ref_att(2) / 2.0), -sin(ref_att(2) / 2.0), 0.0, 0.0);

    Eigen::Quaterniond q_world_ref = q_x_inv * q_y_inv * q_z_inv;
    Eigen::Quaterniond q_cam = cam_ref_q.conjugate() * q_world_ref;

    Eigen::Vector3d t_cam = - q_cam.toRotationMatrix() * ref_p - cam_ref_q.conjugate().toRotationMatrix() * cam_ref_t;

    return camera->reprojectionError(P, q_cam, t_cam, observed_p);
}

void
InfrastructureCalibration::frameReprojectionError(const FramePtr& frame,
                                                  const CameraConstPtr& camera,
                                                  const Pose& T_cam_ref,
                                                  double& minError, double& maxError, double& avgError,
                                                  size_t& featureCount) const
{
    minError = std::numeric_limits<double>::max();
    maxError = std::numeric_limits<double>::min();

    size_t count = 0;
    double totalError = 0.0;

    const std::vector<Point2DFeaturePtr>& features2D = frame->features2D();

    for (size_t i = 0; i < features2D.size(); ++i)
    {
        const Point2DFeatureConstPtr& feature2D = features2D.at(i);
        const Point3DFeatureConstPtr& feature3D = feature2D->feature3D();

        if (feature3D.get() == 0)
        {
            continue;
        }

        double error
            = reprojectionError(camera, feature3D->point(),
                                T_cam_ref.rotation(),
                                T_cam_ref.translation(),
                                frame->systemPose()->position(),
                                frame->systemPose()->attitude(),
                                Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y));

        if (minError > error)
        {
            minError = error;
        }
        if (maxError < error)
        {
            maxError = error;
        }
        totalError += error;
        ++count;
    }

    if (count == 0)
    {
        avgError = 0.0;
        minError = 0.0;
        maxError = 0.0;
        featureCount = count;

        return;
    }

    avgError = totalError / count;
    featureCount = count;
}

void
InfrastructureCalibration::frameReprojectionError(const FramePtr& frame,
                                                  const CameraConstPtr& camera,
                                                  double& minError, double& maxError, double& avgError,
                                                  size_t& featureCount) const
{
    minError = std::numeric_limits<double>::max();
    maxError = std::numeric_limits<double>::min();

    size_t count = 0;
    double totalError = 0.0;

    const std::vector<Point2DFeaturePtr>& features2D = frame->features2D();

    for (size_t i = 0; i < features2D.size(); ++i)
    {
        const Point2DFeatureConstPtr& feature2D = features2D.at(i);
        const Point3DFeatureConstPtr& feature3D = feature2D->feature3D();

        if (feature3D.get() == 0)
        {
            continue;
        }

        double error = camera->reprojectionError(feature3D->point(),
                                                 frame->cameraPose()->rotation(),
                                                 frame->cameraPose()->translation(),
                                                 Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y));

        if (minError > error)
        {
            minError = error;
        }
        if (maxError < error)
        {
            maxError = error;
        }
        totalError += error;
        ++count;
    }

    if (count == 0)
    {
        avgError = 0.0;
        minError = 0.0;
        maxError = 0.0;
        featureCount = count;

        return;
    }

    avgError = totalError / count;
    featureCount = count;
}

void
InfrastructureCalibration::reprojectionError(double& minError, double& maxError,
                                             double& avgError, size_t& featureCount) const
{
    minError = std::numeric_limits<double>::max();
    maxError = std::numeric_limits<double>::min();

    size_t count = 0;
    double totalError = 0.0;

    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        const FrameSet& frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            const FramePtr& frame = frameset.frames.at(j);

            Pose T_cam_ref(m_cameraSystem.getGlobalCameraPose(frame->cameraId()));

            double frameMinError;
            double frameMaxError;
            double frameAvgError;
            size_t frameFeatureCount;

            frameReprojectionError(frame,
                                   m_cameraSystem.getCamera(frame->cameraId()),
                                   T_cam_ref,
                                   frameMinError, frameMaxError, frameAvgError, frameFeatureCount);

            if (minError > frameMinError)
            {
                minError = frameMinError;
            }
            if (maxError < frameMaxError)
            {
                maxError = frameMaxError;
            }
            totalError += frameAvgError * frameFeatureCount;
            count += frameFeatureCount;
        }
    }

    if (count == 0)
    {
        avgError = 0.0;
        minError = 0.0;
        maxError = 0.0;
        featureCount = 0;

        return;
    }

    avgError = totalError / count;
    featureCount = count;
}


double
InfrastructureCalibration::radialReprojectionError(const CameraConstPtr& camera,
                                             const Eigen::Vector3d& P,
                                             const Eigen::Quaterniond& cam_ref_q,
                                             const Eigen::Vector3d& cam_ref_t,
                                             const Eigen::Vector3d& ref_p,
                                             const Eigen::Vector3d& ref_att,
                                             const Eigen::Vector2d& observed_p) const
{
    Eigen::Quaterniond q_z_inv(cos(ref_att(0) / 2.0), 0.0, 0.0, -sin(ref_att(0) / 2.0));
    Eigen::Quaterniond q_y_inv(cos(ref_att(1) / 2.0), 0.0, -sin(ref_att(1) / 2.0), 0.0);
    Eigen::Quaterniond q_x_inv(cos(ref_att(2) / 2.0), -sin(ref_att(2) / 2.0), 0.0, 0.0);

    Eigen::Quaterniond q_world_ref = q_x_inv * q_y_inv * q_z_inv;
    Eigen::Quaterniond q_cam = cam_ref_q.conjugate() * q_world_ref;

    Eigen::Vector3d t_cam = - q_cam.toRotationMatrix() * ref_p - cam_ref_q.conjugate().toRotationMatrix() * cam_ref_t;

    return camera->radialReprojectionError(P, q_cam, t_cam, observed_p);
}

void
InfrastructureCalibration::frameRadialReprojectionError(const FramePtr& frame,
                                                  const CameraConstPtr& camera,
                                                  const Pose& T_cam_ref,
                                                  double& minError, double& maxError, double& avgError,
                                                  size_t& featureCount) const
{
    minError = std::numeric_limits<double>::max();
    maxError = std::numeric_limits<double>::min();

    size_t count = 0;
    double totalError = 0.0;

    const std::vector<Point2DFeaturePtr>& features2D = frame->features2D();

    for (size_t i = 0; i < features2D.size(); ++i)
    {
        const Point2DFeatureConstPtr& feature2D = features2D.at(i);
        const Point3DFeatureConstPtr& feature3D = feature2D->feature3D();

        if (feature3D.get() == 0)
        {
            continue;
        }

        double error
                = radialReprojectionError(camera, feature3D->point(),
                                    T_cam_ref.rotation(),
                                    T_cam_ref.translation(),
                                    frame->systemPose()->position(),
                                    frame->systemPose()->attitude(),
                                    Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y));

        if (minError > error)
        {
            minError = error;
        }
        if (maxError < error)
        {
            maxError = error;
        }
        totalError += error;
        ++count;
    }

    if (count == 0)
    {
        avgError = 0.0;
        minError = 0.0;
        maxError = 0.0;
        featureCount = count;

        return;
    }

    avgError = totalError / count;
    featureCount = count;
}

void
InfrastructureCalibration::frameRadialReprojectionError(const FramePtr& frame,
                                                  const CameraConstPtr& camera,
                                                  double& minError, double& maxError, double& avgError,
                                                  size_t& featureCount) const
{
    minError = std::numeric_limits<double>::max();
    maxError = std::numeric_limits<double>::min();

    size_t count = 0;
    double totalError = 0.0;

    const std::vector<Point2DFeaturePtr>& features2D = frame->features2D();

    for (size_t i = 0; i < features2D.size(); ++i)
    {
        const Point2DFeatureConstPtr& feature2D = features2D.at(i);
        const Point3DFeatureConstPtr& feature3D = feature2D->feature3D();

        if (feature3D.get() == 0)
        {
            continue;
        }

        double error = camera->radialReprojectionError(feature3D->point(),
                                                 frame->cameraPose()->rotation(),
                                                 frame->cameraPose()->translation(),
                                                 Eigen::Vector2d(feature2D->keypoint().pt.x, feature2D->keypoint().pt.y));

        if (minError > error)
        {
            minError = error;
        }
        if (maxError < error)
        {
            maxError = error;
        }
        totalError += error;
        ++count;
    }

    if (count == 0)
    {
        avgError = 0.0;
        minError = 0.0;
        maxError = 0.0;
        featureCount = count;

        return;
    }

    avgError = totalError / count;
    featureCount = count;
}



void
InfrastructureCalibration::radialReprojectionError(double& minError, double& maxError,
                                             double& avgError, size_t& featureCount) const
{
    minError = std::numeric_limits<double>::max();
    maxError = std::numeric_limits<double>::min();

    size_t count = 0;
    double totalError = 0.0;

    for (size_t i = 0; i < m_framesets.size(); ++i)
    {
        const FrameSet& frameset = m_framesets.at(i);

        for (size_t j = 0; j < frameset.frames.size(); ++j)
        {
            const FramePtr& frame = frameset.frames.at(j);

            Pose T_cam_ref(m_cameraSystem.getGlobalCameraPose(frame->cameraId()));

            double frameMinError;
            double frameMaxError;
            double frameAvgError;
            size_t frameFeatureCount;

            frameRadialReprojectionError(frame,
                                   m_cameraSystem.getCamera(frame->cameraId()),
                                   T_cam_ref,
                                   frameMinError, frameMaxError, frameAvgError, frameFeatureCount);

            if (minError > frameMinError)
            {
                minError = frameMinError;
            }
            if (maxError < frameMaxError)
            {
                maxError = frameMaxError;
            }
            totalError += frameAvgError * frameFeatureCount;
            count += frameFeatureCount;
        }
    }

    if (count == 0)
    {
        avgError = 0.0;
        minError = 0.0;
        maxError = 0.0;
        featureCount = 0;

        return;
    }

    avgError = totalError / count;
    featureCount = count;
}

OdometryPtr
InfrastructureCalibration::solveRigPoses(Eigen::MatrixXd cam_T_ref_stack,
                                         Eigen::MatrixXd cam_T_world_stack,
                                         FrameSet& frameset) const
{
    int numberOverlap = cam_T_ref_stack.rows()/2;
    // solve for world_T_ref in cam_T_world * world_T_ref == cam_T_ref
    //svd for rotation
    Eigen::Matrix3d ATC = (cam_T_ref_stack.transpose() * cam_T_world_stack).block<3,3>(0,0);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ATC, Eigen::ComputeFullU | Eigen::ComputeFullV );
    Eigen::Matrix3d V = svd.matrixV(), U = svd.matrixU();
    Eigen::Matrix3d sigma = Eigen::Matrix3d::Identity();
    sigma(2,2) = (V*U.transpose()).determinant();
    Eigen::Matrix3d R = V * sigma * U.transpose();

    // solve for translation
    Eigen::MatrixXd A_trans = cam_T_world_stack.block(0,0, numberOverlap * 2, 3);
    Eigen::MatrixXd B_trans = cam_T_ref_stack.block(0,3, numberOverlap * 2, 1) -
                              cam_T_world_stack.block(0,3, numberOverlap * 2, 1);
    Eigen::Vector3d world_t_ref = (A_trans.transpose()*A_trans).inverse()*A_trans.transpose()*B_trans;


    //store Qi for frameset i
    OdometryPtr odometry = boost::make_shared<Odometry>();
    odometry->timeStamp() = frameset.frames.at(0)->cameraPose()->timeStamp();

    odometry->position() = world_t_ref;

    Eigen::Quaterniond qAvg = Eigen::Quaterniond(R);

    double roll, pitch, yaw;
    mat2RPY(qAvg.toRotationMatrix(), roll, pitch, yaw);

    odometry->attitude() = Eigen::Vector3d(yaw, pitch, roll);
    return odometry;
}

void
InfrastructureCalibration::setRigPoses(std::vector<Pose, Eigen::aligned_allocator<Pose> > T_cam_ref){

    //set rig global poses
    for (size_t j = 0; j < m_framesets.size(); ++j)
    {
        int numberOverlap = 0;
        int cameraIdx;

        Eigen::MatrixXd cam_T_ref_stack(4,4);
        Eigen::MatrixXd cam_T_world_stack(4,4);

        for (size_t frameid = 0; frameid <  m_framesets.at(j).frames.size(); ++frameid) {
            cameraIdx =  m_framesets.at(j).frames.at(frameid)->cameraId();
            cam_T_world_stack.conservativeResize(numberOverlap * 2 + 2, 4);
            cam_T_ref_stack.conservativeResize(numberOverlap * 2 + 2, 4);
            cam_T_ref_stack.block<2, 4>(numberOverlap * 2, 0) =
                    T_cam_ref.at(cameraIdx).toMatrix().inverse().block<2, 4>(0, 0);
            cam_T_world_stack.block<2, 4>(numberOverlap * 2, 0) =
                    m_framesets.at(j).frames.at(frameid)->cameraPose()->toMatrix().block<2, 4>(0, 0);
            numberOverlap++;
        }

        // solve for world_T_ref in cam_T_world * world_T_ref == cam_T_ref
        OdometryPtr odometry = solveRigPoses(cam_T_ref_stack, cam_T_world_stack, m_framesets.at(j));

        for (size_t k = 0; k <  m_framesets.at(j).frames.size(); ++k)
        {
            m_framesets.at(j).frames.at(k)->systemPose() = odometry;
        }
    }
}

void
InfrastructureCalibration::printCameraExtrinsics(const std::string& header) const
{
    std::cout<<header<<std::endl;
    std::cout << std::fixed << std::setprecision(10);
    for (int i = 0; i < m_cameras.size(); ++i)
    {
        const Eigen::Matrix4d& globalPose = m_cameraSystem.getLocalCameraPose(i);

        if (m_cameras.at(i)->cameraName().empty())
        {
            std::cout << "camera_" << i;
        }
        else
        {
            std::cout << m_cameras.at(i)->cameraName();
        }
        std::cout << std::endl;

        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                std::cout << globalPose(j,k);

                if (k < 3)
                {
                    std::cout << " ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
}

