if(OpenCV_FOUND)
infrascal_library(infrascal_dutilscv SHARED
  Drawing.cpp
  Geometry.cpp
  GUI.cpp
  IO.cpp
  Mat.cpp
  Transformations.cpp
  Types.cpp
)

infrascal_link_libraries(infrascal_dutilscv
  ${INFRASCAL_PLATFORM_UNIX_LIBRARIES}
  ${OpenCV_LIBS}
)

infrascal_install(infrascal_dutilscv)
endif()
