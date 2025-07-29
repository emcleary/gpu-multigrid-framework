import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(infile):
    x = []
    y = []
    u = []
    v = []

    print('Loading', infile)
    with open(infile, 'r') as file:
        while line := file.readline():
            if line.startswith('#'):
                continue
            xi, yi, ui, vi = line.rstrip('\n').strip().split(',')
            try:
                x.append(float(xi))
                y.append(float(yi))
                u.append(float(ui))
                v.append(float(vi))
            except:
                print(f"xi={xi}, yi={yi}, ui={ui}, vi={vi}")
                exit()

    n = int(np.sqrt(len(x)))
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)
    x = x.reshape((n, n))
    y = y.reshape((n, n))
    u = u.reshape((n, n))
    v = v.reshape((n, n))

    # vavg = np.mean(v.ravel())
    # v -= vavg
    
    return x, y, u, v

def plot(x, y, u, v):
    vmin = min(np.min(v), np.min(u))
    vmax = max(np.max(v), np.max(u))
    levels = np.linspace(vmin, vmax, 21)
    
    fig, ax = plt.subplots(3, 1, sharex=True)

    im = ax[0].contourf(x, y, u, levels=levels) # analytical
    fig.colorbar(im, ax=ax[0])
    im = ax[1].contourf(x, y, v, levels=levels) # numerical
    fig.colorbar(im, ax=ax[1])
    im = ax[2].contourf(x, y, u - v) # error
    fig.colorbar(im, ax=ax[2])

    # ax[0].set_ylabel('v(x)')
    # ax[1].set_ylabel('e(x)')
    # ax[1].set_xlabel('x')

    plt.show()
    # plt.savefig('plot.png', bbox_inches='tight')

if __name__ == '__main__':
    if len(sys.argv) >= 1:
        infiles = sys.argv[1:]
    else:
        infiles = ['results.csv']

    for infile in infiles:
        x, y, u, v = load_data(infile)
        plot(x, y, u, v)
