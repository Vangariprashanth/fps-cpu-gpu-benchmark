/**
 * This code implements the naive gpu version of the FPS algorithm.
 * It benchmarks the code using TimeIt function.
 * Naive implementation:
 *  kernel is launcehd K times to calculate the distance between the last selected sample and each point in the point cloud.
 *  The distance array is updated.
 *  This updated distance is copied back to the host and the maximum is computed on the host side.
 * This version is slower than the cpu version becuase of K number of host-device transfers.
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

    // compute the global thread id. Each thread represents each point in the point cloud to compute the distance between the point and the last selected point
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
        return;

    // distance between source point and the target point. Taking the eucledian distance.
    // skipping the square root because we will take the maximum anyway
    double dx = xs[i] - tx;
    double dy = ys[i] - ty;
    double dz = zs[i] - tz;
    dist[i] = fmin(dist[i], (dx * dx) + (dy * dy) + (dz * dz));
}

void fps(
    const size_t N,
    const size_t K,
    const vector<double> &xs,
    const vector<double> &ys,
    const vector<double> &zs,
    double *d_xs,
    double *d_ys,
    double *d_zs,
    double *d_dist,
    vector<size_t> &selectedIndices,
    vector<double> &hostDist)
{
    size_t lastSelected = 5; // seed
    selectedIndices.clear();
    selectedIndices.push_back(lastSelected);

    // Initialize the host distance to infinity again because we are repeatedly calling this method in the TimeIt function.
    fill(hostDist.begin(), hostDist.end(), std::numeric_limits<double>::max());
    cudaMemcpy(d_dist, hostDist.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // repeat sample number of times-1. because the first is already selected using random seed.
    for (size_t r = 1; r < K; ++r)
    {

        // x,y,z coordinates of the last selected point
        double tx = xs[lastSelected];
        double ty = ys[lastSelected];
        double tz = zs[lastSelected];

        // call the kernel and update the distance array
        updateKernel<<<blocks, threadsPerBlock>>>(d_xs, d_ys, d_zs, d_dist, N, tx, ty, tz);

        cudaMemcpy(hostDist.data(), d_dist,
                   N * sizeof(double), cudaMemcpyDeviceToHost); // copy back the distance array back to the host to get the maximum

        double farthestDistance{};
        size_t farthestIndex = 0;

        // get the max distance(farthest distance of all the points) and its index
        for (size_t i = 0; i < N; ++i)
        {
            if (hostDist[i] > farthestDistance)
            {
                farthestDistance = hostDist[i];
                farthestIndex = i;
            }
        }

        // push the maximum index as the new farthest point
        selectedIndices.push_back(farthestIndex);
        lastSelected = farthestIndex;
    }
}

template <typename F>
double timeItMin(F &&f, int reps = 5)
{
    f();                // warm-up
    double best = 1e18; // initialize the minimum time taken (best) to infinity
    for (int r = 0; r < reps; ++r)
    {
        auto t1 = std::chrono::high_resolution_clock::now(); // start time
        f();
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();                                                 // end time
        best = std::min(best, std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * 1.0); // get the minimum time
    }
    return best;
}

int main()
{
    const size_t N = 1'00'000;
    const size_t K = 1024;

    vector<double> xs(N), ys(N), zs(N);

    // synthetic point cloud generation

    for (size_t i = 0; i < N; i++)
    {
        xs[i] = double(i);
        ys[i] = double(i) * 2;
        zs[i] = double(i) * 3;
    }

    // device memory allocation and initialization

    double *d_xs, *d_ys, *d_zs, *d_dist;
    cudaMalloc(&d_xs, N * sizeof(double));
    cudaMalloc(&d_ys, N * sizeof(double));
    cudaMalloc(&d_zs, N * sizeof(double));
    cudaMalloc(&d_dist, N * sizeof(double));
    cudaMemcpy(d_xs, xs.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, ys.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zs, zs.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // Nearest Distance out of all the selected points is set to infinity and copied to the device
    vector<double> hostDist(N, std::numeric_limits<double>::max());
    cudaMemcpy(d_dist, hostDist.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    vector<size_t> selectedIndices;

    // Call the fps function and time it
    double gpuTime = timeItMin([&]()
                               { return fps(N, K, xs, ys, zs, d_xs, d_ys, d_zs, d_dist, selectedIndices, hostDist); });

    cout << "Point Cloud size: " << N << endl;
    cout << "Number of samples: " << K << endl;
    cout << "Naive GPU FPS time: " << gpuTime << " us\n";

    // cudaFree the four device pointers
    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_zs);
    cudaFree(d_dist);

    return 0;
}
