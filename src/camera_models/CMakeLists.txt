include_directories(
  ${CERES_INCLUDE_DIRECTORIES}
)

if(GLOG_FOUND AND OpenCV_FOUND)
infrascal_library(infrascal_camera_models SHARED
  Camera.cc
  CameraFactory.cc
  CataCamera.cc
  CostFunctionFactory.cc
  EquidistantCamera.cc
  PinholeCamera.cc
  ScaramuzzaCamera.cc
)

infrascal_link_libraries(infrascal_camera_models
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  ${OpenCV_LIBS}
  infrascal_gpl
  ceres
  ${GLOG_LIBRARIES}
)

infrascal_install(infrascal_camera_models)

infrascal_test(CataCamera)
infrascal_link_libraries(CataCamera_test infrascal_camera_models)

infrascal_test(EquidistantCamera)
infrascal_link_libraries(EquidistantCamera_test infrascal_camera_models)

infrascal_test(PinholeCamera)
infrascal_link_libraries(PinholeCamera_test infrascal_camera_models)

endif(GLOG_FOUND AND OpenCV_FOUND)
