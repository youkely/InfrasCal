if(OpenCV_FOUND)
include_directories(
  ../dbow2/DBoW2
  ../dbow2/DUtils
  ../dbow2/DUtilsCV
  ../dbow2/DVision
)

infrascal_library(infrascal_location_recognition SHARED
  LocationRecognition.cc
)

infrascal_link_libraries(infrascal_location_recognition
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  infrascal_dbow2
  infrascal_dutilscv
  infrascal_dutils
  infrascal_dvision
  infrascal_sparse_graph
)

infrascal_install(infrascal_location_recognition)
endif(OpenCV_FOUND)