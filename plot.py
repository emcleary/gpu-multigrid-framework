import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(infile):
    x = []
    u = []
    v = []

    print('Loading', infile)
    with open(infile, 'r') as file:
        while line := file.readline():
            if line.startswith('#'):
                continue
            xi, ui, vi = line.rstrip('\n').strip().split(',')
            try:
                x.append(float(xi))
                u.append(float(ui))
                v.append(float(vi))
            except:
                print(f"xi={xi}, ui={ui}, vi={vi}")
                exit()

    return np.asarray(x), np.asarray(u), np.asarray(v)

def plot(x, u, v):
    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(x, u, label='analytical solution', lw=4)
    ax[0].plot(x, v, label='numerical solution', lw=2)
    ax[0].legend()

    ax[1].plot(x, u - v, label='analytical solution')

    ax[0].set_ylabel('v(x)')
    ax[1].set_ylabel('e(x)')
    ax[1].set_xlabel('x')

    plt.show()
    # plt.savefig('plot.png', bbox_inches='tight')

if __name__ == '__main__':
    if len(sys.argv) >= 1:
        infiles = sys.argv[1:]
    else:
        infiles = ['results.csv']

    for infile in infiles:
        x, u, v = load_data(infile)
        plot(x, u, v)
