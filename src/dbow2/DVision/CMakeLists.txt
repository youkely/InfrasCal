include_directories(
  ../DUtils
  ../DUtilsCV
)

if(OpenCV_FOUND)
    
  if(${OpenCV_VERSION} VERSION_LESS 3.0.0)
    # outdated components that don't work in opencv3
    set(DVision_COMPONENTS
      SurfSet.cpp
      Matches.cpp
    )
  endif()
endif()

infrascal_library(infrascal_dvision SHARED
  BRIEF.cpp
  BundleCamera.cpp
  FSolver.cpp
  HSolver.cpp
  ImageFunctions.cpp
  ORB.cpp
  PatchFile.cpp
  PixelPointFile.cpp
  PLYFile.cpp
  PMVSCamera.cpp
  ${DVision_COMPONENTS}
)

infrascal_link_libraries(infrascal_dvision
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  ${OpenCV_LIBS}
  infrascal_dutils
)

infrascal_install(infrascal_dvision)
