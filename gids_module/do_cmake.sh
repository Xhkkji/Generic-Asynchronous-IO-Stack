# 所有对bam系统代码的修改在此处重新编译即可
cd build
cmake .. && make -j
cd BAM_Feature_Store
pip install .
cd ..
# cd ../..
# cd evaluation