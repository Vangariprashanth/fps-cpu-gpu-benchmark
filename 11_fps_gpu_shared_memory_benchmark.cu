/**
 * This code implements the optimized version of gpu version of the FPS algorithm.
 * It benchmarks the code using TimeIt function.
 * Optimized implementation :
 * Shared memory reduction : Instead of host - device transfer of distance array and computing the max distance on the cpu,
 * Shared memory performs block level maximum and stores it in the blockMaxDist. Instead of computing maximum on N sized array we can compute the maximum on n blocks
 * **/

#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
using namespace std;

// update the distance array with the minimum distance of all the selected points so far
__global__ void updateKernel(const double *xs, const double *ys, const double *zs,
                             double *dist, size_t N,
                             double tx, double ty, double tz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    double dx = xs[i] - tx;
    double dy = ys[i] - ty;
    double dz = zs[i] - tz;
    dist[i] = fmin(dist[i], dx * dx + dy * dy + dz * dz);
}

// perform block level reduction- get the maximum and store it in the index 0 of the blockMaxDist
__global__ void argMaxKernel(const double *dist, size_t N,
                             double *blockMaxDist, int *blockMaxIdx)
{
    __shared__ double sDist[256];
    __shared__ int sIdx[256];

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    // load value and aswell as index (index of the point cloud array)
    sDist[t] = (i < N) ? dist[i] : 0.0;
    sIdx[t] = (i < N) ? (int)i : -1;
    __syncthreads();

    // tree reduction toward slot 0.
    // store both value and index
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (t < stride)
        {
            if (sDist[t + stride] > sDist[t])
            {
                sDist[t] = sDist[t + stride];
                sIdx[t] = sIdx[t + stride];
            }
        }
        __syncthreads();
    }

    // thread 0 writes this block's maximum
    if (t == 0)
    {
        blockMaxDist[blockIdx.x] = sDist[0];
        blockMaxIdx[blockIdx.x] = sIdx[0];
    }
}

void fps(size_t N, size_t K,
         const vector<double> &xs, const vector<double> &ys, const vector<double> &zs,
         double *d_xs, double *d_ys, double *d_zs, double *d_dist,
         double *d_blockMaxDist, int *d_blockMaxIdx,
         vector<double> &h_blockMaxDist, vector<int> &h_blockMaxIdx,
         vector<double> &hostDist,
         vector<size_t> &selectedIndices,
         int threadsPerBlock, int blocks)
{
    size_t lastSelected = 5; // seed
    selectedIndices.clear();
    selectedIndices.push_back(lastSelected);

    // dist starts at "infinity"
    fill(hostDist.begin(), hostDist.end(), numeric_limits<double>::max());
    cudaMemcpy(d_dist, hostDist.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    for (size_t r = 1; r < K; ++r)
    {
        double tx = xs[lastSelected];
        double ty = ys[lastSelected];
        double tz = zs[lastSelected];

        updateKernel<<<blocks, threadsPerBlock>>>(d_xs, d_ys, d_zs, d_dist, N, tx, ty, tz);
        argMaxKernel<<<blocks, threadsPerBlock>>>(d_dist, N, d_blockMaxDist, d_blockMaxIdx);

        // copy the maximum of each block.
        cudaMemcpy(h_blockMaxDist.data(), d_blockMaxDist, blocks * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_blockMaxIdx.data(), d_blockMaxIdx, blocks * sizeof(int), cudaMemcpyDeviceToHost);

        // Get the maximum of all the blocks (each element already stores the maximum of that block)
        double farthestDistance = numeric_limits<double>::lowest();
        size_t farthestIndex = 0;
        for (int b = 0; b < blocks; ++b)
        {
            if (h_blockMaxDist[b] > farthestDistance)
            {
                farthestDistance = h_blockMaxDist[b];     // maximum value
                farthestIndex = (size_t)h_blockMaxIdx[b]; // maximum index
            }
        }

        selectedIndices.push_back(farthestIndex);
        lastSelected = farthestIndex;
    }
}

template <typename F>
double timeItMin(F &&f, int reps = 5)
{
    f(); // warm-up
    double best = 1e18;
    for (int r = 0; r < reps; ++r)
    {
        auto t1 = chrono::high_resolution_clock::now();
        f();
        cudaDeviceSynchronize();
        auto t2 = chrono::high_resolution_clock::now();
        best = min(best, chrono::duration_cast<chrono::microseconds>(t2 - t1).count() * 1.0);
    }
    return best;
}

int main()
{
    const size_t N = 100000;
    const size_t K = 1024;

    // SoA host data: point i = (i, 2i, 3i)
    vector<double> xs(N), ys(N), zs(N);
    for (size_t i = 0; i < N; i++)
    {
        xs[i] = double(i);
        ys[i] = double(i) * 2;
        zs[i] = double(i) * 3;
    }

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    double *d_xs, *d_ys, *d_zs, *d_dist;
    cudaMalloc(&d_xs, N * sizeof(double));
    cudaMalloc(&d_ys, N * sizeof(double));
    cudaMalloc(&d_zs, N * sizeof(double));
    cudaMalloc(&d_dist, N * sizeof(double));
    cudaMemcpy(d_xs, xs.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, ys.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zs, zs.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // block max dist stores the maximum of each block
    double *d_blockMaxDist;
    int *d_blockMaxIdx;
    cudaMalloc(&d_blockMaxDist, blocks * sizeof(double));
    cudaMalloc(&d_blockMaxIdx, blocks * sizeof(int));

    vector<double> h_blockMaxDist(blocks);
    vector<int> h_blockMaxIdx(blocks);
    vector<double> hostDist(N);
    vector<size_t> selectedIndices;

    double gpuTime = timeItMin([&]()
                               { fps(N, K, xs, ys, zs, d_xs, d_ys, d_zs, d_dist,
                                     d_blockMaxDist, d_blockMaxIdx, h_blockMaxDist, h_blockMaxIdx,
                                     hostDist, selectedIndices, threadsPerBlock, blocks); });

    cout << "Point Cloud size: " << N << endl;
    cout << "Number of samples: " << K << endl;
    cout << "Shared-memory GPU FPS: " << gpuTime << " us\n";

    // freeing the pointers
    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_zs);
    cudaFree(d_dist);
    cudaFree(d_blockMaxDist);
    cudaFree(d_blockMaxIdx);
    return 0;
}
