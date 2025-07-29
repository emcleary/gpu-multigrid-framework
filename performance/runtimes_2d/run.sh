#!/bin/env bash

elf=../../build/bin/performance/runtimes_2d

n_levels_min=2
n_levels_max=13

solver=1 # linear

n_cpu=4

seconds=3

filename="data.csv"
rm -vf $filename

for i in `seq $n_levels_min $n_levels_max`; do
    $elf $i $solver | grep GPU >> $filename
    tail -n1 $filename
    sleep $seconds
    $elf $i $solver 1 | grep CPU >> $filename
    tail -n1 $filename
    sleep $seconds
    $elf $i $solver $n_cpu | grep CPU >> $filename
    tail -n1 $filename
    sleep $seconds
done
