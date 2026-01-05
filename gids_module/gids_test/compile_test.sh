nvcc -std=c++14 -O3 -Xcompiler -fPIC \
    test_simple.cpp \
    -I/path/to/your/headers \
    -L/usr/local/cuda/lib64 -lcudart \
    -o test_bam