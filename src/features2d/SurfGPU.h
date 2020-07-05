#ifndef SURFGPU_H
#define SURFGPU_H
#include <boost/thread.hpp>


#ifdef HAVE_CUDA

    //////////////////
    // CUDA + OPENCV3
    //////////////////
#include <opencv2/cudafeatures2d.hpp>
#else  // HAVE_CUDA
    //////////////////
    // OPENCV3
    //////////////////
#include <opencv2/features2d.hpp>
#endif // HAVE_CUDA

#ifndef HAVE_OPENCV3
#include <opencv2/legacy/legacy.hpp>
#endif // HAVE_OPENCV3

namespace infrascal
{


class SurfGPU
{

#ifdef HAVE_CUDA

    //////////////////
    // CUDA + OPENCV3
    //////////////////
    typedef cv::cuda::GpuMat                               MatType;
    typedef cv::cuda::DescriptorMatcher                    MatcherType;
#else // HAVE_CUDA
    
    //////////////////
    // OPENCV3
    //////////////////
    typedef cv::Mat                                        MatType;
    typedef cv::DescriptorMatcher                          MatcherType;
#endif // HAVE_CUDA

public:
    SurfGPU();

    ~SurfGPU();

    static cv::Ptr<SurfGPU> instance();

    void knnMatch(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors,
                  std::vector<std::vector<cv::DMatch> >& matches, int k,
                  const cv::Mat& mask = cv::Mat(), bool compactResult = false);
    void radiusMatch(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors,
                     std::vector<std::vector<cv::DMatch> >& matches, float maxDistance,
                     const cv::Mat& mask = cv::Mat(), bool compactResult = false);

private:
    static cv::Ptr<SurfGPU> m_instance;
    static boost::mutex m_instanceMutex;

    cv::Ptr<MatcherType> m_matcher;
};

}

#endif
