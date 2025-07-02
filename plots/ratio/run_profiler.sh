#!/bin/env bash

# nsys profile --stats=true ./build/bin/performance/lhs  $((1<<22)) 100
# # nsys profile --stats=true ./build/bin/examples/testing_iterator  $((1<<22)) 100
# # view out in
# # nsys-ui


# compute profiler
# ncu --page details -f -o testing_iterator_testing.profout  ./build/bin/examples/testing_iterator  $((1<<22)) 100
# ncu --section MemoryWorkloadAnalysis --page details -f -o testing_lhs_naive.profout  ./build/bin/performance/lhs  $((1<<22)) 256
/usr/local/cuda-12.9/bin/ncu --page details -f -o testing_cpu_gpu_ratio.profout ../../build/bin/performance/cpu_gpu_ratio 22 1 10
