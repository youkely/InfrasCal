if(OpenCV_FOUND)
add_subdirectory(DBoW2)
add_subdirectory(DUtils)
add_subdirectory(DUtilsCV)
add_subdirectory(DVision)

include_directories(
  DBoW2
  DUtils
  DUtilsCV
  DVision
)

infrascal_executable(dbow2_demo
  demo.cpp
)

infrascal_link_libraries(dbow2_demo
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  infrascal_dbow2
  infrascal_dutils
  infrascal_dutilscv
  infrascal_dvision
)
endif()
