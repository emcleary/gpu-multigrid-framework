# CPU - GPU Tradeoff

According the [NVIDIA's blog
post](https://developer.nvidia.com/blog/high-performance-geometric-multi-grid-gpu-acceleration/)
on multigrid solvers, switching from GPU to CPU at coarse levels of
the cycles yield an improvement in F-cycles but not in V-cycles. While
I have not implemented F-cycles, I implemented these tests with
V-cycles and my results so far support this claim.