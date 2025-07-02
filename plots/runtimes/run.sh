#!/bin/env bash

rm -v *.txt *.csv

elf2=../../build/bin/examples/heat_equation_1d_2nd_order
elf_nonlinear=../../build/bin/examples/nonlinear_1d_2nd_order


n_levels_min=2
n_levels_max=26

linear=1
nonlinear_full=2

on_cpu=0
on_gpu=1

# rm -vf results* levels.txt

for i in `seq $n_levels_min $n_levels_max`; do
    echo $i
    echo $i >> levels.txt

    echo "results_linear_2nd_order_cpu.txt"
    $elf2 $i $linear $on_cpu | grep "Average time" >> results_linear_2nd_order_cpu.txt
    echo "results_linear_2nd_order_gpu.txt"
    $elf2 $i $linear $on_gpu | grep "Average time" >> results_linear_2nd_order_gpu.txt

    # echo "results_nonlinear_full_cpu.txt"
    # $elf_nonlinear $i $nonlinear_full $on_cpu | grep "Average time" >> results_nonlinear_full_cpu.txt
    # echo "results_nonlinear_full_gpu.txt"
    # $elf_nonlinear $i $nonlinear_full $on_gpu | grep "Average time" >> results_nonlinear_full_gpu.txt

done
