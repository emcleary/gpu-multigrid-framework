# Iterative Solvers

The iterative solver or "relaxing" steps of a cycle are the main
bottleneck in GPU performance, hence their explicit testing
here. Unfortunately, the use of shared memory, despite the wide
stencil of the 4th order schemes implmemented here, does not increase
performance.