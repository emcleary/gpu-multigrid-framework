import sys
import matplotlib.pyplot as plt
sys.path.append('..')
from runtimes import RuntimesPlot
from speedup import SpeedupPlot

def load_data(filename):
    v = []
    with open(filename, 'r') as file:
        while line := file.readline():
            t = float(line.split(' ')[2])
            v.append(t)
    return tuple(v)

def load_levels():
    levels = []
    with open('levels.txt', 'r') as file:
        while line := file.readline():
            levels.append(int(line.rstrip('\n')))
    return tuple(levels)

def make_plots(levels, file_cpu, file_gpu, title, name_runtime, name_speedup):
    data = [
        ("CPU", load_data(file_cpu)),
        ("GPU", load_data(file_gpu)),
    ]

    rt = RuntimesPlot(levels, data)
    rt.title = title
    rt.plot(name_runtime)

    sp = SpeedupPlot(levels, data[0][1], data[1][1])
    sp.title = title
    sp.plot(name_speedup)


if __name__ == '__main__':
    levels = load_levels()

    make_plots(levels,
               'results_linear_2nd_order_cpu.txt',
               'results_linear_2nd_order_gpu.txt',
               'Linear 2nd Order Example',
               'results_runtimes_linear_2nd_order.png',
               'results_speedup_linear_2nd_order.png')

    make_plots(levels,
               'results_linear_4th_order_cpu.txt',
               'results_linear_4th_order_gpu.txt',
               'Linear 4th Order Example',
               'results_runtimes_linear_4th_order.png',
               'results_speedup_linear_4th_order.png')

    make_plots(levels,
               'results_nonlinear_full_cpu.txt',
               'results_nonlinear_full_gpu.txt',
               'Nonlinear Full Example',
               'results_runtimes_nonlinear_full.png',
               'results_speedup_nonlinear_full.png')

    make_plots(levels,
               'results_nonlinear_error_cpu.txt',
               'results_nonlinear_error_gpu.txt',
               'Nonlinear Error Example',
               'results_runtimes_nonlinear_error.png',
               'results_speedup_nonlinear_error.png')
    
