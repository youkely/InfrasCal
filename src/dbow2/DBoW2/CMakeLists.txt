if(OpenCV_FOUND)
include_directories(
  ../DUtils
  ../DVision
)

infrascal_library(infrascal_dbow2 SHARED
  BowVector.cpp
  FBrief.cpp
  FeatureVector.cpp
  FOrb.cpp
  FSurf64.cpp
  FSift128.cpp
  QueryResults.cpp
  ScoringObject.cpp
)

infrascal_link_libraries(infrascal_dbow2
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  infrascal_dutils
  infrascal_dutilscv
  infrascal_dvision
)

infrascal_install(infrascal_dbow2)
endif()
