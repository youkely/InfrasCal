if(OpenCV_FOUND AND HAVE_OPENCV_XFEATURES2D_NONFREE)
include_directories(
  ${CERES_INCLUDE_DIRECTORIES}
  ../dbow2/DBoW2
  ../dbow2/DUtils
  ../dbow2/DUtils
  ../dbow2/DUtilsCV
  ../dbow2/DVision
  ../SiftGPU/src/SiftGPU/
  ${OpenGL_INCLUDE_DIR}
)

infrascal_library(infrascal_infrastr_calib SHARED
  InfrastructureCalibration.cc
)


infrascal_link_libraries(infrascal_infrastr_calib
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${OpenCV_LIBS}
  infrascal_camera_systems
  infrascal_features2d
  infrascal_gpl
  infrascal_pose_estimation
  infrascal_location_recognition
  infrascal_sparse_graph
  radialpose
  ceres
  sift_gpu
)



#infrascal_install(infrascal_infrastr_calib)
endif(OpenCV_FOUND AND HAVE_OPENCV_XFEATURES2D_NONFREE)
