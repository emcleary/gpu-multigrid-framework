#!/bin/env bash
# rm -rfv build main
cmake -B build -DCMAKE_BUILD_TYPE="Release"
cd build
make -j6
cd -
