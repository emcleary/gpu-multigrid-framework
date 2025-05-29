import sys
import matplotlib.pyplot as plt


class Plot:
    def __init__(self):
        self._xlabel = None
        self._ylabel = None
        self._title = None
        self._fig = None
        self._ax = None

    def plot(self, name=None):
        self._fig, self._ax = plt.subplots()

        self._plot_data()

        if self._xlabel:
            self._ax.set_xlabel(self._xlabel)

        if self._ylabel:
            self._ax.set_ylabel(self._ylabel)

        if self._title:
            self._ax.set_title(self._title)

        if name:
            plt.savefig(name, bbox_inches='tight')
        else:
            plt.show()

    def _plot_data(self):
        raise NotImplementedError

    @property
    def xlabel(self):
        return self._xlabel

    @xlabel.setter
    def xlabel(self, label):
        self._xlabel = label

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
    def title(self, label):
        self._title = label

class SpeedupPlot(Plot):
    def __init__(self, levels, cpu, gpu):
        self._levels = levels
        self._grid = [1<<lvl for lvl in levels]
        self._data = [c / g for c, g in zip(cpu, gpu)]
        self._xlabel = 'Number of Grid Cells'
        self._ylabel = 'Speedup (CPU / GPU)'

    def _plot_data(self):
        self._ax.semilogx(self._grid, self._data)
