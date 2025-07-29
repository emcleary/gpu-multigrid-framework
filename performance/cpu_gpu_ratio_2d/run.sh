#!/bin/env bash

levels=$1
solver=1 # linear
burnin=10
iterations=400

for cpu in `seq 1 8`; do
    filename="data_levels_${levels}_cpu_${cpu}.csv"
    rm -vf $filename
    echo $filename

    for i in `seq 0 $levels`; do
	echo $i "gpu levels"
	for j in `seq 0 7`; do
	    echo "../../build/bin/performance/cpu_gpu_ratio_2d" $levels $solver $i $cpu $burnin $iterations
	    ../../build/bin/performance/cpu_gpu_ratio_2d $levels $solver $i $cpu $burnin $iterations | grep data >> $filename
	    tail -n1 $filename
	    sleep 3
	done
    done
done
