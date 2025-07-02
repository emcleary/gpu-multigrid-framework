#!/bin/env bash
executable=../../build/bin/examples/heat_equation_1d_2nd_order

n_levels_min=2
n_levels_max=15

linear=1
nonlinear_full=2
nonlinear_error=3

on_cpu=0
on_gpu=1

rm -vf results* levels.txt


for i in `seq $n_levels_min $n_levels_max`; do
    echo $i
    echo $i >> levels.txt
    $executable $i $linear $on_cpu | grep "|e|" >> results_linear_cpu.txt
    $executable $i $linear $on_gpu | grep "|e|" >> results_linear_gpu.txt
    $executable $i $nonlinear_full $on_cpu | grep "|e|" >> results_nonlinear_full_cpu.txt
    $executable $i $nonlinear_full $on_gpu | grep "|e|" >> results_nonlinear_full_gpu.txt
done
