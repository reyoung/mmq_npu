cmake_minimum_required(VERSION 3.16.0)
project(mmq_npu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "Ascend910B2" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(TORCH_NPU_PATH "/opt/micromamba/lib/python3.10/site-packages/torch_npu" CACHE PATH "torch_npu package path")
set(TORCH_PATH "/opt/micromamba/lib/python3.10/site-packages/torch" CACHE PATH "torch package path")

find_package(Torch REQUIRED
    PATHS ${TORCH_PATH}/share/cmake/Torch)

find_package(tikicpulib REQUIRED
    PATHS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/cmake)
set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

add_subdirectory(3rd/pybind11)

set(KernelSources csrc/add_kernel.cpp)

# always compile CPU kernels for intelli sense
add_library(kernels_cpu SHARED ${KernelSources})
target_link_libraries(kernels_cpu PUBLIC tikicpulib::${SOC_VERSION})
ascendc_library(kernels_npu STATIC ${KernelSources})


pybind11_add_module(mmq_npu_ csrc/pybind.cpp csrc/add_cpu.cpp csrc/acl_system_info.cpp)
target_link_libraries(mmq_npu_ PRIVATE ${TORCH_LIBRARIES} kernels_npu
    ${TORCH_NPU_PATH}/lib/libtorch_npu.so)
target_include_directories(mmq_npu_ PRIVATE ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include)
target_include_directories(mmq_npu_ PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_npu)

install(TARGETS mmq_npu_ DESTINATION mmq_npu)
