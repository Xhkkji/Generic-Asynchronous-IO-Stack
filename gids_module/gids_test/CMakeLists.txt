# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(BAMTest)

find_package(CUDA REQUIRED)

# 设置CUDA编译选项
set(CUDA_NVCC_FLAGS 
    -std=c++14
    -O3
    -Xcompiler -fPIC
    -gencode arch=compute_70,code=sm_70
)

# 添加可执行文件
cuda_add_executable(test_bam test_simple.cpp)

# 链接CUDA库
target_link_libraries(test_bam 
    ${CUDA_LIBRARIES}
)