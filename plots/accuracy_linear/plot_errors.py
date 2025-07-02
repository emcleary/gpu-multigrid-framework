import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')
from accuracy import AccuracyPlot


def load_data(filename):
    assert(filename.startswith('results_'))
    v = []
    with open(filename, 'r') as file:
        while line := file.readline():
            e = float(line.rstrip('\n').split(' ')[-1])
            v.append(e)
    return tuple(v)

def load_levels():
    levels = []
    with open('levels.txt', 'r') as file:
        while line := file.readline():
            levels.append(int(line.rstrip('\n')))
    return tuple(levels)

if __name__ == '__main__':
    levels = load_levels()

    data2 = load_data('results_linear_2nd_order.txt')
    
    ap2 = AccuracyPlot(levels, data2)
    ap2.add_reference(2, 1)
    # ap2.title = 'Linear Example'
    ap2.plot('results_2nd_order_linear.png')


    data2 = load_data('results_linear_2nd_order.txt')
    ap2.add_reference(2, 1)
    # ap2.title = 'Nonlinear Solver'
    ap2.plot('results_2nd_order_nonlinear.png')
