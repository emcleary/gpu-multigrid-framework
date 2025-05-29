#!/bin/env bash

elf2=../../build/bin/examples/heat_equation_1d_2nd_order
elf4=../../build/bin/examples/heat_equation_1d_4th_order
elf_nonlinear=../../build/bin/examples/nonlinear_1d_2nd_order


# n_levels_min=2
# n_levels_max=24

n_levels_min=25
n_levels_max=26

linear=1
nonlinear_full=2
nonlinear_error=3

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
    echo "results_linear_4th_order_cpu.txt"
    $elf4 $i $linear $on_cpu | grep "Average time" >> results_linear_4th_order_cpu.txt
    echo "results_linear_4th_order_gpu.txt"
    $elf4 $i $linear $on_gpu | grep "Average time" >> results_linear_4th_order_gpu.txt

    echo "results_nonlinear_full_cpu.txt"
    $elf_nonlinear $i $nonlinear_full $on_cpu | grep "Average time" >> results_nonlinear_full_cpu.txt
    echo "results_nonlinear_full_gpu.txt"
    $elf_nonlinear $i $nonlinear_full $on_gpu | grep "Average time" >> results_nonlinear_full_gpu.txt
    echo "results_nonlinear_error_cpu.txt"
    $elf_nonlinear $i $nonlinear_error $on_cpu | grep "Average time" >> results_nonlinear_error_cpu.txt
    echo "results_nonlinear_error_gpu.txt"
    $elf_nonlinear $i $nonlinear_error $on_gpu | grep "Average time" >> results_nonlinear_error_gpu.txt

done
