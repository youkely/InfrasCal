if(OpenCV_FOUND)
include_directories(
  ${CERES_INCLUDE_DIRECTORIES}
)

set_source_files(GPL_SRC_FILES
  CubicSpline.cc
  EigenQuaternionParameterization.cc
  gpl.cc
)

infrascal_library(infrascal_gpl SHARED ${GPL_SRC_FILES})
infrascal_link_libraries(infrascal_gpl
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  ${OpenCV_LIBS}
  ceres
  ${CERES_LIBRARIES}
)

infrascal_install(infrascal_gpl)
endif(OpenCV_FOUND)
