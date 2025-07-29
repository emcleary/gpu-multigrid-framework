#!/bin/env bash
# rm -rfv build main
cmake -B build -DCMAKE_BUILD_TYPE="Release" -DWITH_OPENMP=ON
# cmake -B build -DCMAKE_BUILD_TYPE="Release" -DWITH_OPENMP=OFF
cd build
make -j6
cd -
