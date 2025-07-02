import sys
import matplotlib.pyplot as plt
sys.path.append('..')

def load_data(filename):
    v = []
    with open(filename, 'r') as file:
        while line := file.readline():
            if line.startswith('data'):
                _, gpu, cpu, duration = list(filter(lambda x: x, line.rstrip('\n').split(',')))
            v.append((float(gpu), float(cpu), float(duration)))
    return tuple(v)

if __name__ == '__main__':
    data = load_data("results.txt")

    n_gpu = [d[0] for d in data]
    duration = [d[2] for d in data]
    plt.bar(n_gpu, duration)
    plt.xlabel('Number of Levels Calculated on the GPU')
    plt.ylabel('Average V-Cycle Iteration Time [ms]')
    plt.title(f'{len(n_gpu)-1} Level V-Cycle')
    plt.savefig('results_ratio.png', bbox_inches='tight')
