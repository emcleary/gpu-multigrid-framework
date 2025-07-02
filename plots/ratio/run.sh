#!/bin/env bash

solver=1
n_levels=22
n_burnin=5
n_iter=10

rm -v results.txt
for n_gpu in `seq 0 $n_levels`; do
    ../../build/bin/performance/cpu_gpu_ratio $n_levels $solver $n_gpu $n_burnin $n_iter | grep "data" >> results.txt
    tail -n1 results.txt
done
python3 plots.py
