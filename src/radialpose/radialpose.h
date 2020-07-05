#pragma once
#include <Eigen/Dense>
#include "misc/distortion.h"
#include "misc/refinement.h"
#include "solvers/pose_estimator.h"
#include "solvers/larsson_iccv19.h"
#include "solvers/larsson_iccv17.h"
#include "solvers/kukelova_iccv13.h"
#include "solvers/oskarsson_arxiv18.h"
#include "solvers/bujnak_accv10.h"
//#include "solvers/byrod_cvpr09.h"

/* 
 Notes:
  - In VS you may need to set /bigobj to be able to compile
*/


extern template class radialpose::PoseEstimator<radialpose::bujnak_accv10::NonPlanarSolver>;
extern template class radialpose::PoseEstimator<radialpose::larsson_iccv17::NonPlanarSolver>;
extern template class radialpose::PoseEstimator<radialpose::kukelova_iccv13::Radial1DSolver>;
extern template class radialpose::PoseEstimator<radialpose::kukelova_iccv13::Solver>;

extern template class radialpose::larsson_iccv19::Solver<1, 0, true>;
extern template class radialpose::larsson_iccv19::Solver<2, 0, true>;
extern template class radialpose::larsson_iccv19::Solver<3, 0, true>;
extern template class radialpose::larsson_iccv19::Solver<3, 3, true>;
extern template class radialpose::larsson_iccv19::Solver<1, 0, false>;

extern template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<1, 0, true>>;
extern template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<2, 0, true>>;
extern template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<3, 0, true>>;
extern template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<3, 3, true>>;
extern template class radialpose::PoseEstimator<radialpose::larsson_iccv19::Solver<1, 0, false>>;

