#!/bin/env bash
elf2=../../build/bin/examples/heat_equation_1d_2nd_order

n_levels_min=2
n_levels_max=22

linear=1
nonlinear=2

on_gpu=1

rm -vf results* levels.txt


for i in `seq $n_levels_min $n_levels_max`; do
    echo $i
    echo $i >> levels.txt
    $elf2 $i $linear $on_gpu | grep "|e|" >> results_linear_2nd_order.txt
    $elf2 $i $nonlinear $on_gpu | grep "|e|" >> results_nonlinear_2nd_order.txt
done
