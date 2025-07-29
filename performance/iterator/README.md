# Iterative Solvers

The iterative solver or "relaxing" steps of a cycle are the main
bottleneck in GPU performance, hence their explicit testing
here. Unfortunately, the use of shared memory for an asynchronous
Red-Black Gauss-Seidel iterative solver does not improve runtime.

# Future Work

At least for second order accurate schemes, it is common practice in
when parallelizing with OpenMP to split the solution into 2 arrays, 1
for red points and 1 for black points, and finally joining them after
all iterations are completed. It might be worth doing the same for
shared memory.