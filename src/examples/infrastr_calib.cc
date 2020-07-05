#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <iomanip>
#include <iostream>
#include <Eigen/Eigen>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <thread>
#include <limits>
#include "../gpl/gpl.h"

#ifdef HAVE_OPENCV3
#include <opencv2/imgproc.hpp>
#ifdef HAVE_OPENCV4
#include <opencv2/imgproc/imgproc_c.h>
#endif // HAVE_OPENCV4
#else
#include <opencv2/imgproc/imgproc.hpp>
#endif // HAVE_OPENCV3

#include <numeric>
#include "infrascal/camera_models/CameraFactory.h"
#include "infrascal/infrastr_calib/InfrastructureCalibration.h"

int
main(int argc, char** argv)
{
    using namespace infrascal;
    namespace fs = ::boost::filesystem;
    
    //Eigen::initParallel();
    std::string calibDir;
    int cameraCount;
    std::string cameraModel;
    std::vector<int> cameraIds;
    std::string outputDir;
    bool preprocessImages;
    bool optimizePoints;
    std::string mapDataDir;
    std::string databaseDataDir;
    std::string inputDir;
    std::string vocDir;
    bool verbose;
    int startFrame;
    int endFrame;
    bool saveFrames;
    bool loadFrames;
    int frameStep;
    std::string calibMode;
    //================= Handling Program options ==================
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("calib,c", boost::program_options::value<std::string>(&calibDir)->default_value("calib"), "Directory containing camera intrinsic input calibration files.")
        ("camera-count", boost::program_options::value<int>(&cameraCount)->default_value(1), "Number of cameras in rig.")
        ("camera-model", boost::program_options::value<std::string>(&cameraModel)->default_value("pinhole-radtan"), "Model name of camera. pinhole-radtan/pinhole-equi")
        ("camera-ids", boost::program_options::value<std::vector<int> >(&cameraIds)->multitoken(), "Ids of cameras to be calibrated in rig.")
        ("output,o", boost::program_options::value<std::string>(&outputDir)->default_value("calibration_data"), "Directory to write calibration result to.")
        ("preprocess", boost::program_options::bool_switch(&preprocessImages)->default_value(false), "Preprocess images.")
        ("optimize-points", boost::program_options::bool_switch(&optimizePoints)->default_value(false), "Optimize 3d scene points in BA step.")
        ("map", boost::program_options::value<std::string>(&mapDataDir)->default_value("map"), "Location of folder which contains map reconstruction data.")
        ("database", boost::program_options::value<std::string>(&databaseDataDir)->default_value("database.db"), "File of Colmap database which contains map data.")
        ("input", boost::program_options::value<std::string>(&inputDir)->default_value("input"), "Location of the folder containing all input data.")
        ("vocab", boost::program_options::value<std::string>(&vocDir)->default_value("vocab"), "file path of vovabulary")
        ("verbose,v", boost::program_options::bool_switch(&verbose)->default_value(false), "Verbose output")
        ("start-frame", boost::program_options::value<int>(&startFrame)->default_value(0), "Start frame of calibration.")
        ("end-frame", boost::program_options::value<int>(&endFrame)->default_value(100000), "End frame of calibration.")
        ("save", boost::program_options::bool_switch(&saveFrames)->default_value(false), "Save frames")
        ("load", boost::program_options::bool_switch(&loadFrames)->default_value(false), "Load frames")
        ("frame-step", boost::program_options::value<int>(&frameStep)->default_value(1), "Frame stride for input frames.")
        ("calib-mode", boost::program_options::value<std::string>(&calibMode)->default_value("InRaSU"), "Calibration mode, choosing from InRa, InRaS, InRaSU, In, InRI")

            ;
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);


    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }
    std::cout << "# INFO: Calibration mode " << calibMode << std::endl;

    // Check if directory containing camera calibration files exists
    if (!boost::filesystem::exists(calibDir))
    {
        if(calibMode=="In" || calibMode =="InRI")
        {
            std::cout << "# ERROR: No calibration file provided." <<std::endl;
            exit(0);
        }
        if (verbose)
        {
            std::cout << "# INFO: Using radial pose absolute estimator with unknown intrinsics." << std::endl;
        }
    }
    {
        std::cout << "# INFO: Initializing... " << std::endl << std::flush;
    }
    //========================= Handling Input =======================

    //===========================Initialize calibration==========================

    // read camera params
    std::vector<infrascal::CameraPtr> cameras(cameraCount);
    if (cameraIds.empty() || cameraIds.size()!=cameraCount)
    {
        cameraIds.resize(cameraCount);
        std::iota(cameraIds.begin(), cameraIds.end(), 0);
    }
    Camera::ModelType cameraModelType;
    if(cameraModel == "pinhole-radtan") cameraModelType = Camera::PINHOLE;
    else if(cameraModel == "pinhole-equi") cameraModelType = Camera::KANNALA_BRANDT;
    else
    {
        std::cout << "# ERROR: Please select correct camera model." <<std::endl;
        exit(0);
    }

    if (loadFrames || calibMode=="In" || calibMode =="InRI")
    {
        for (int i = 0; i < cameraCount; ++i)
        {
            infrascal::CameraPtr camera;

            boost::filesystem::path calibFilePath(calibDir);

            std::ostringstream oss;
            oss << "camera_" << cameraIds[i] << "_calib.yaml";
            calibFilePath /= oss.str();

            camera = infrascal::CameraFactory::instance()->generateCameraFromYamlFile(calibFilePath.string());
            if (camera.get() == 0)
            {
                std::cout << "# ERROR: Unable to read calibration file: " << calibFilePath.string() << std::endl;

                return 0;
            }

            // read camera mask
            {
            boost::filesystem::path maskFilePath(calibDir);

            std::ostringstream oss;
            oss << "camera_" << cameraIds[i] << "_mask.png";
            maskFilePath /= oss.str();

            cv::Mat mask = cv::imread(maskFilePath.string());
            if (!mask.empty())
            {
                cv::Mat grey;
                cv::cvtColor(mask, grey, CV_RGB2GRAY, 1);
                camera->mask() = grey;
                std::cout << "# INFO: Found camera mask for camera " << camera->cameraName() << std::endl;
            }
            }
        cameras.at(i) = camera;
        }
    }

    //========================= Get all files  =========================
    typedef std::map<int64_t, std::string>  ImageMap;

    std::vector< ImageMap > inputImages(cameraCount);
    {
        if (verbose)
        {
            printf("Get images from input directory\n");
        }
        fs::path inputFilePath(inputDir);

        //images
        for(int i = 0; i < cameraCount; i++)
        {
            std::ifstream file(inputDir+"/cam"+std::to_string(cameraIds[i])+"_data.txt");
            if (!file.is_open())
            {
                printf("Cannot open %s", (inputDir+"/cam"+std::to_string(cameraIds[i])+"_data.txt").c_str());
                return 1;
            }
            // read line by line
            std::string line;
            while(std::getline(file, line))
            {
                std::stringstream str(line);

                if(str.str()[0]=='#') continue;
                // type of event
                unsigned long long timestamp = 0;
                std::string imgfilename;

                str >> timestamp >> imgfilename;
                inputImages[i][timestamp] = inputDir + "/cam"+std::to_string(cameraIds[i]) + "/" + imgfilename;
            }
            cv::Mat imgTemp = cv::imread(inputImages[i].begin()->second);
            if (loadFrames)
                continue;
            if (calibMode=="InRa" || calibMode =="InRaS" || calibMode =="InRaSU") {
                cameras.at(i) = infrascal::CameraFactory::instance()->generateCamera(
                        cameraModelType, "camera_" + std::to_string(cameraIds[i]),
                        cv::Size(imgTemp.cols, imgTemp.rows));
            }

        }
    }
    if(endFrame > inputImages[0].size())
        endFrame = inputImages[0].size();
    {
        std::cout << "# INFO: Number of frames: " << (endFrame - startFrame) / frameStep << std::endl;
    }
    //========================= Start Threads =========================
    InfrastructureCalibration::Options options;
    options.outputDir = outputDir;
    options.verbose = verbose;
    options.optimizeIntrinsics = !(calibMode == "In");
    options.optimizePoints = optimizePoints;
    options.calibMode = calibMode;

    InfrastructureCalibration infrasCalib(cameras, options);
    if(infrasCalib.loadMap(mapDataDir, databaseDataDir, vocDir)){
        std::cout << "# INFO: Initialization finished!" << std::endl;
    }
    else{
        std::cout << "# INFO: Initialization failed!" << std::endl;
        exit(0);
    }


    std::vector<ImageMap::iterator> camIterator(cameraCount);
    for (int c=0; c < cameraCount; c++) {
        camIterator[c] = inputImages[c].begin();
        std::advance (camIterator[c],startFrame);
    }
    double tsStart = timeInSeconds();
    int count_frames = startFrame;
    const int number_of_frames_used = endFrame;
    if(!loadFrames) {
        while (camIterator[0] != inputImages[0].end()) {
            std::vector<cv::Mat> imageset;
            uint64_t timestamp_imageset = camIterator[0]->first;
            for (int c = 0; c < cameraCount; c++) {
                imageset.push_back(cv::imread(camIterator[c]->second));
                std::advance (camIterator[c],frameStep);

            }
            count_frames = count_frames+frameStep;
            infrasCalib.addFrameSet(imageset, timestamp_imageset, preprocessImages);
            if (count_frames >= number_of_frames_used)
                break;
        }
    }
    boost::filesystem::path outFilePath(outputDir);
    if (!boost::filesystem::is_directory(outFilePath))
    {
        boost::filesystem::create_directory(outFilePath);
    }
    outFilePath /= "frames.sg";
    if(loadFrames){
        infrasCalib.loadFrameSets(outFilePath.string());
    }
    if (saveFrames){
        infrasCalib.saveFrameSets(outFilePath.string());
    }
    {
        std::cout << "# INFO: Number of framesets as input:" << count_frames - startFrame << std::endl;
        std::cout << "# INFO: Pose estimation took "
                  << timeInSeconds() - tsStart << " s." << std::endl;
    }
    infrasCalib.run();
    {
        std::cout << "# INFO: Calibration took "
                  << timeInSeconds() - tsStart << " s." << std::endl;
    }
    if (saveFrames||loadFrames){
        infrasCalib.writeToColmap(outputDir);
    }

    return 0;
}
