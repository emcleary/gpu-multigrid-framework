import sys
import numpy as np
import matplotlib.pyplot as plt

class Data:
    def __init__(self, gpu_levels):
        self._n_gpu = gpu_levels
        self._value = 0
        self._count = 0

    def add(self, time):
        self._value += time
        self._count += 1

    @property
    def value(self):
        if self._count == 0:
            return -1
        return self._value / self._count

def load_file(filename):
    data = {} # key=gpu_levels, value=Data obj
    with open(filename, 'r') as file:
        for line in file.readlines():
            _, gpu, _, time = line.rstrip('\n').split(',')
            if gpu not in data:
                data[gpu] = Data(gpu)
            data[gpu].add(float(time))

    d = []
    for k, v in data.items():
        d.append((int(k), v.value))
    d.sort(key=lambda x: x[0])
    return list(zip(*d))
    

def plot(gpu, times):
    plt.plot(gpu, times)
    plt.show()

if __name__ == '__main__':
    # filenames = sys.argv[1:]
    # for filename in filenames:
    #     gpu, times = load_file(filename)
    #     plt.plot(gpu, times, label=filename)
    # plt.legend()
    # plt.show()

    filenames = [
        'data_levels_10_cpu_1.csv',
        'data_levels_10_cpu_2.csv',
        'data_levels_10_cpu_4.csv',
        'data_levels_10_cpu_8.csv',
    ]

    cpu = [1, 2, 4, 8]

    for threads, filename in zip(cpu, filenames):
        gpu, times = load_file(filename)
        plt.plot(gpu, times, label=f'{threads} CPU threads')

    plt.xlabel('# Finest Mesh Levels On GPU')
    plt.ylabel('runtime / V-cycle [ms]')
    plt.legend()
    plt.title('1025x1025 Grid')

    plt.savefig('cpu_gpu_ratio_2d.png', bbox_inches='tight')
