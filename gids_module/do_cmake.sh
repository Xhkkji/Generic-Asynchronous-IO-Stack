# 所有对bam系统代码的修改在此处重新编译即可
cd build
cmake .. && make -j
cd BAM_Feature_Store
pip install .
cd ..
# cd ../..
# cd evaluation

# 卸载AGILE驱动并加载BaM驱动
# echo "0000:68:00.0" | sudo tee /sys/bus/pci/devices/0000:68:00.0/driver/unbind
# cd build/module
# sudo make load