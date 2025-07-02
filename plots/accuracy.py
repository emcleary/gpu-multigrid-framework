import sys
import numpy as np
import matplotlib.pyplot as plt


class AccuracyPlot:
    def __init__(self, levels, data):
        self._levels = levels
        self._grid = [1<<lvl for lvl in levels]
        self._data = data
        self._ref = {}
        self._xlabel = 'Number of Grid Cells'
        self._ylabel = '$||e||_2$'
        self._title = None

    @property
    def ylabel(self):
        return self._ylabel

    @ylabel.setter
    def ylabel(self, label):
        self._ylabel = label

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    def add_reference(self, order, start):
        data = [start]
        scale = 1 / 2 ** order
        for _ in self._grid[1:]:
            data.append(data[-1] * scale)
        self._ref[order] = data

    def plot(self, outfile=None):
        fig, ax = plt.subplots()
        ax.semilogy(self._levels, self._data, label='numerical solution')

        if self._ref:
            for k, v in self._ref.items():
                if k == 1:
                    label = '1st order accuracy'
                elif k == 2:
                    label = '2nd order accuracy'
                elif k <= 20:
                    label = f'{k}th order accuracy'
                else:
                    print('Not setup for order of accuracy greater than 20')
                    sys.exit()
                ax.semilogy(self._levels, v, '--', label=label)
        plt.legend()

        ax.set_xlabel(self._xlabel)
        ax.set_ylabel(self._ylabel)

        ax.xaxis.set_ticklabels([f"$2^{{{i}}}$" for i in self._levels][::2])
        ax.set_title(self._title)
        
        if outfile:
            plt.savefig(outfile, bbox_inches='tight')
        else:
            plt.show()
