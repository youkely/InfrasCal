if(OpenCV_FOUND)
infrascal_library(infrascal_sparse_graph SHARED
  Odometry.cc
  Pose.cc
  SparseGraph.cc
  SparseGraphUtils.cc
  Transform.cc
)

find_package (SQLite3)
include_directories(${SQLite3_INCLUDE_DIR})

infrascal_link_libraries(infrascal_sparse_graph
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${OpenCV_LIBS}
  ${SQLite3_LIBRARY}
)

infrascal_install(infrascal_sparse_graph)
endif(OpenCV_FOUND)
