if(OpenCV_FOUND AND HAVE_OPENCV_XFEATURES2D_NONFREE)
infrascal_library(infrascal_features2d SHARED
  SurfGPU.cc
)


infrascal_link_libraries(infrascal_features2d
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_SYSTEM_LIBRARY_RELEASE}
  ${CUDA_CUDART_LIBRARY_OPTIONAL}
  ${Boost_LIBRARIES}
)

infrascal_install(infrascal_features2d)

endif(OpenCV_FOUND AND HAVE_OPENCV_XFEATURES2D_NONFREE)
