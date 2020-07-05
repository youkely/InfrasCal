
if(OpenCV_FOUND)
include_directories(
  ${CERES_INCLUDE_DIRECTORIES}
)

infrascal_library(infrascal_camera_systems SHARED
  CameraSystem.cc
)

infrascal_link_libraries(infrascal_camera_systems
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  infrascal_camera_models
  infrascal_pugixml
)

infrascal_install(infrascal_camera_systems)
endif()
