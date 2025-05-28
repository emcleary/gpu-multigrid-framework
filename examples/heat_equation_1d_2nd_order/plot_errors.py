import sys
import numpy as np
import matplotlib.pyplot as plt

def plot(levels, data):
    fig, ax = plt.subplots()
    for label, errors in data:
        ax.semilogy(levels, errors, label=label)
    plt.legend()
    plt.show()


def load_data(filename):
    assert(f.startswith('results_'))
    k = ' '.join(f.split('.')[0].split('_')[1:])
    v = []
    with open(f, 'r') as file:
        while line := file.readline():
            e = float(line.rstrip('\n').split(' ')[-1])
            v.append(e)
    return k, tuple(v)

def load_levels():
    levels = []
    with open('levels.txt', 'r') as file:
        while line := file.readline():
            levels.append(int(line.rstrip('\n')))
    return tuple(levels)

if __name__ == '__main__':
    levels = load_levels()
    
    filenames = sys.argv[1:]
    data = list()
    for f in filenames:
        k, v = load_data(f)
        data.append((k, v))

    plot(levels, data)
