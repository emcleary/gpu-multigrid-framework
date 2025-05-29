# Heat Equation

This example solves a steady, 1 dimensional heat equation with a
source term,

$$ -u_{xx} = f(x) $$

where $x \in \left[0, 1\right]$ and $u(0) = u(1) = 0$. The
differential equation is discretized with a fourth order central
difference scheme plus a non-central, fourth order scheme near the
boundaries.  The following plot shows the accuracy of its results
using the linear solver.

<p align="center">
<img src="../../plots/accuracy_linear/results_2nd_order.png" height="200">
</p>

Inaccuracies at the highest resolutions are likely due to the
additional schemes near the boundaries. This has not been
investigated.