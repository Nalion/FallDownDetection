if(IOS)
  ocv_module_disable(superres)
endif()

if(EVSS_OPENCV_REDUCE)
ocv_module_disable(superres)
endif()



set(the_description "Super Resolution")
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 -Wundef -Wshadow)
if(ENABLE_DYNAMIC_CUDA)
  add_definitions(-DDYNAMIC_CUDA_SUPPORT)
  ocv_define_module(superres EXCLUDE_CUDA opencv_imgproc opencv_video OPTIONAL opencv_highgui opencv_ocl)
else()
  ocv_define_module(superres opencv_imgproc opencv_video OPTIONAL opencv_gpu opencv_highgui opencv_ocl ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
endif()
