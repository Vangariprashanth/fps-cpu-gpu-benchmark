# fps-cpu-gpu-benchmark project

## Farthest Point Sampling — CPU & CUDA

Farthest Point Sampling (FPS) from scratch on CPU and GPU, with a shared-memory
reduction kernel and AoS-vs-SoA benchmark. FPS is a common
downsampling step in 3D point-cloud pipelines. It repeatedly selects the point
farthest from everything chosen so far so that the selected points are evenly distributed (covers the whole point cloud)

## Results

N = 100,000 points, k = 1024, FP64, end-to-end.
**NVIDIA RTX 3060** (12 GB) / **Intel Xeon E5-2680 v4**.

| Implementation             | Time    | Notes                                          |
| -------------------------- | ------- | ---------------------------------------------- |
| CPU (single-threaded, -O2) | ~273 ms | optimized baseline                             |
| Naive GPU                  | ~337 ms | **slower than CPU** — per-round transfer bound |
| Shared-memory GPU          | ~40 ms  | **6.8× over CPU, 8.4× over naive GPU**         |

## Algorithm

Brute-force FPS is O(N·k²). Each round recomputes every point's distance to the
whole selected set. Keeping a running "nearest distance to any selected point"
array and folding in only the newly selected point each round reduces this to
**O(N·k)**.

## Files

- `08_fps_cpu_baseline.cpp` — final CPU version, templated over AoS/SoA layouts
- `10_fps_gpu_naive_benchmark.cu` — naive GPU (update on GPU, argmax on host)
- `11_fps_gpu_shared_memory_benchmark.cu` — final GPU (shared-memory argmax reduction)
- `01`–`07`, `09` — incremental build-up (layouts, templates, timing, CPU FPS, naive GPU)

## Build & run

### CPU:

compile:

```
g++ -O2 -march=native 08_fps_cpu_baseline.cpp -o fps_cpu
```

run:

```
./fps_cpu
```

### GPU:

compile:

```
nvcc -O2 11_fps_gpu_shared_memory_benchmark.cu -o fps_gpu
```

run:

```
./fps_gpu
```

## Benchmarking Notes

- All implementations use seed index **5** and produce identical selected indices (verified across CPU, naive GPU, and shared-memory GPU).
- Timing uses warm-up runs, min-of-_N_ sampling to reduce OS noise, and a dead-code-elimination guard to prevent the compiler from removing the measured work.
- Both memory layouts execute the **same templated algorithm** (via compile-time polymorphism), ensuring that memory layout is the only experimental variable.

## Future Work

- Replace hardcoded parameters (`N`, `K`, and seed) with command-line arguments.
- Perform a roofline analysis using **Nsight Compute** to validate the memory-bound behavior against measured memory bandwidth.
- Implement a kd-tree-accelerated FPS algorithm and compare its performance with the brute-force **O(N·K)** implementation.
