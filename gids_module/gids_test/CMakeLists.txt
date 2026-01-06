# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(gids_test)

find_package(CUDA REQUIRED)

include_directories(
    /home/xhk/hyperion/GIDS/gids_module/include  # bam_nvme.h应该在这里
    /home/xhk/hyperion/GIDS/bam/include
    /home/xhk/hyperion/GIDS/bam/include/freestanding/include/
    ${CUDA_INCLUDE_DIRS}
)



# 设置CUDA编译选项
set(CUDA_NVCC_FLAGS 
    -std=c++14
    -O3
    -Xcompiler -fPIC
    -gencode arch=compute_70,code=sm_70
)

# 添加kernel实现文件
cuda_add_executable(gids_test 
    gids_test.cu                    # 你的测试文件
    gids_kernel.cu  # kernel实现文件
)

# 链接CUDA库
target_link_libraries(gids_test 
    ${CUDA_LIBRARIES}
)

# target_include_directories(BAM_Feature_Store PRIVATE /home/xhk/hyperion/GIDS/bam/include)
# target_include_directories(BAM_Feature_Store PRIVATE /home/xhk/hyperion/GIDS/bam/include/freestanding/include/)