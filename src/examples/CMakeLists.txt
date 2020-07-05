
if(OpenCV_FOUND AND HAVE_OPENCV_XFEATURES2D_NONFREE)

infrascal_executable(infrastr_calib
  infrastr_calib.cc
)

infrascal_link_libraries(infrastr_calib
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  infrascal_infrastr_calib
)

endif(OpenCV_FOUND AND HAVE_OPENCV_XFEATURES2D_NONFREE)

if(OpenCV_FOUND)
include_directories(
  ../dbow2/DBoW2
  ../dbow2/DUtils
  ../dbow2/DUtils
  ../dbow2/DUtilsCV
  ../dbow2/DVision
)

infrascal_executable(train_voctree
  train_voctree.cc
)

infrascal_link_libraries(train_voctree
  infrascal_dbow2
)
endif()
