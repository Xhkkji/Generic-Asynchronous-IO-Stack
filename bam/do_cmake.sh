git submodule update --init --recursive
mkdir -p build; cd build
cmake ..
make libnvm                         # builds library
make benchmarks                     # builds benchmark program
cd build/module
make
