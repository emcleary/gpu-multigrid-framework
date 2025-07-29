import sys
import matplotlib.pyplot as plt

def load_data(filename):
    data = {'GPU': [], 'CPU serial': [], 'CPU parallel': []}
    v = []
    with open(filename, 'r') as file:
        while line := file.readline():
            key, levels, runtime = line.rstrip('\n').split(',')
            gridpoints = (1 << int(levels)) ** 2
            data[key].append((gridpoints, float(runtime)))

    for key in data:
        data[key].sort()
    
    return data

def plot_runtimes(data):
    for k, d in data.items():
        gridpoints, runtimes = zip(*d)
        plt.loglog(gridpoints, runtimes, label=k)
    plt.legend()
    plt.xlabel('Number of Grid Cells')
    plt.ylabel('Avg. Runtime / V-Cycle [ms]')
    plt.savefig('results_runtimes.png', bbox_inches='tight')
    plt.close()

def plot_speedup(data):
    gp, r_gpu = zip(*data['GPU'])
    gp_cpu_s, r_cpu_s = zip(*data['CPU serial'])
    gp_cpu_p, r_cpu_p = zip(*data['CPU parallel'])

    # confirm all gridpoint data are identical
    assert len(gp) == len(gp_cpu_s)
    assert len(gp) == len(gp_cpu_p)
    for a, b in zip(gp, gp_cpu_s):
        assert(a == b)
    for a, b in zip(gp, gp_cpu_p):
        assert(a == b)

    gpu_speedup = [ref/gpu for gpu, ref in zip(r_gpu, r_cpu_s)]
    cpu_speedup = [ref/cpu for cpu, ref in zip(r_cpu_p, r_cpu_s)]

    plt.semilogx(gp, gpu_speedup, label='CUDA')
    plt.semilogx(gp, cpu_speedup, label='OpenMP')
    plt.legend()
    plt.xlabel('Number of Grid Cells')
    plt.ylabel('Speedup (Serial / Parallel)')
    plt.savefig('results_speedup.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    filename = sys.argv[1]
    data = load_data(filename)
    plot_runtimes(data)
    plot_speedup(data)
