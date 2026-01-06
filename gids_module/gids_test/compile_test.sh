# compile_test.sh
echo "编译CPU backing测试..."
nvcc gids_test.cu gids_kernel.cu -o test_cpu_backing
echo "运行测试..."
./test_cpu_backing