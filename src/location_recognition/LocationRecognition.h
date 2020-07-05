#ifndef LOCATIONRECOGNITION_H
#define LOCATIONRECOGNITION_H

#include "infrascal/sparse_graph/SparseGraph.h"
#include "../dbow2/DBoW2/DBoW2.h"
#include "../dbow2/DUtils/DUtils.h"
#include "../dbow2/DUtilsCV/DUtilsCV.h"
#include "../dbow2/DVision/DVision.h"

namespace infrascal
{

class LocationRecognition
{
public:
    LocationRecognition();

    void setup(const SparseGraph& graph, const std::string& vocFilename = "surf64.yml.gz");

    void knnMatch(const FrameConstPtr& frame, int k, std::vector<FrameTag>& matches) const;
    void knnMatch_maploc(const FrameConstPtr& frame, int k, std::vector<FrameTag>& matches) const;
    void knnMatch(const FrameConstPtr& frame, int k, std::vector<FramePtr>& matches) const;

private:
    std::vector<std::vector<float> > frameToBOW(const FrameConstPtr& frame) const;

    Sift128Database m_db;

    std::vector<FrameTag> m_frameTags;
    std::vector<FramePtr> m_frames;
    boost::unordered_map<const Frame*,FrameTag> m_frameMap;
};

}

#endif
