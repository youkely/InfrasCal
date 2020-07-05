#include "SurfGPU.h"

#include <iostream>

namespace infrascal
{

cv::Ptr<SurfGPU> SurfGPU::m_instance;
boost::mutex SurfGPU::m_instanceMutex;

SurfGPU::SurfGPU()
 :
#ifdef    HAVE_CUDA

    // opencv3 + CUDA
    m_matcher(cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2))

#else  // HAVE_CUDA

    // opencv3
    m_matcher(cv::DescriptorMatcher::create("BruteForce"))
#endif // HAVE_CUDA
{

}

SurfGPU::~SurfGPU()
{

}

cv::Ptr<SurfGPU>
SurfGPU::instance()
{
    boost::lock_guard<boost::mutex> lock(m_instanceMutex);

    if (m_instance.empty())
    {
        m_instance = cv::Ptr<SurfGPU>(new SurfGPU());
    }

    return m_instance;
}


void
SurfGPU::knnMatch(const cv::Mat& queryDescriptors,
                  const cv::Mat& trainDescriptors,
                  std::vector<std::vector<cv::DMatch> >& matches,
                  int k,
                  const cv::Mat& mask,
                  bool compactResult)
{
    boost::lock_guard<boost::mutex> lock(m_instanceMutex);

    if (queryDescriptors.empty() || trainDescriptors.empty())
    {
        matches.clear();
        return;
    }

    matches.reserve(queryDescriptors.rows);
#ifdef HAVE_CUDA
    MatType qDtorsGPU(queryDescriptors);
    MatType tDtorsGPU(trainDescriptors);

    MatType maskGPU;
    if (!mask.empty())
    {
        maskGPU.upload(mask);
    }

    m_matcher->knnMatch(qDtorsGPU, tDtorsGPU, matches, k, maskGPU, compactResult);
#else
    m_matcher->knnMatch(queryDescriptors, trainDescriptors, matches, k,mask,compactResult);
#endif
}

void
SurfGPU::radiusMatch(const cv::Mat& queryDescriptors,
                     const cv::Mat& trainDescriptors,
                     std::vector<std::vector<cv::DMatch> >& matches,
                     float maxDistance,
                     const cv::Mat& mask,
                     bool compactResult)
{
    boost::lock_guard<boost::mutex> lock(m_instanceMutex);

    if (queryDescriptors.empty() || trainDescriptors.empty())
    {
        matches.clear();
        return;
    }

    matches.reserve(queryDescriptors.rows);
#ifdef HAVE_CUDA
    MatType qDtorsGPU(queryDescriptors);
    MatType tDtorsGPU(trainDescriptors);

    MatType maskGPU;
    if (!mask.empty())
    {
        maskGPU.upload(mask);
    }

    m_matcher->radiusMatch(qDtorsGPU, tDtorsGPU, matches, maxDistance, maskGPU, compactResult);
#else
    m_matcher->radiusMatch(queryDescriptors, trainDescriptors, matches, maxDistance, mask, compactResult);
#endif // HAVE_CUDA
}


}
