#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <cuda_runtime.h>
using namespace std;

__global__ void updateKernel(const double *xs, const double *ys, const double *zs,
                             double *dist, size_t N,
                             double tx, double ty, double tz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        double dx = xs[i] - tx;
        double dy = ys[i] - ty;
        double dz = zs[i] - tz;
        dist[i] = fmin(dist[i], (dx * dx) + (dy * dy) + (dz * dz));
    }
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
    selectedIndices.clear(); // clears the selectedIndices because it is ran multiple times by TimeIt function
    selectedIndices.push_back(lastSelected);

    fill(hostDist.begin(), hostDist.end(), std::numeric_limits<double>::max());      // initializing the host distance to "infinity"
    cudaMemcpy(d_dist, hostDist.data(), N * sizeof(double), cudaMemcpyHostToDevice); // copying the host distance array to the device array

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (size_t r = 1; r < K; ++r)
    {

        double tx = xs[lastSelected];
        double ty = ys[lastSelected];
        double tz = zs[lastSelected];
        updateKernel<<<blocks, threadsPerBlock>>>(d_xs, d_ys, d_zs, d_dist, N, tx, ty, tz);

        cudaMemcpy(hostDist.data(), d_dist,
                   N * sizeof(double), cudaMemcpyDeviceToHost); // Get the distance array from the device and copy it to the
                                                                // host to find the max distance which is our farthest point
        double farthestDistance{};
        size_t farthestIndex = 0;

        // iterate over the distance array and get the max distance and its index to get the farthest point and the index
        for (size_t i = 0; i < N; ++i)
        {
            if (hostDist[i] > farthestDistance)
            {
                farthestDistance = hostDist[i];
                farthestIndex = i;
            }
        }

        selectedIndices.push_back(farthestIndex); // push the farthest piont
        lastSelected = farthestIndex;
    }
}

int main()
{
    const size_t N = 1'00'000; // point cloud size
    const size_t K = 1024;     // number of samples

    // Generating synthetic point cloud with point(i,2i,3i) format
    vector<double> xs(N), ys(N), zs(N);
    for (size_t i = 0; i < N; i++)
    {
        xs[i] = double(i);
        ys[i] = double(i) * 2;
        zs[i] = double(i) * 3;
    }

    // allocating device memory for x,y,z coordinates and distance array

    double *d_xs, *d_ys, *d_zs, *d_dist;
    cudaMalloc(&d_xs, N * sizeof(double));
    cudaMalloc(&d_ys, N * sizeof(double));
    cudaMalloc(&d_zs, N * sizeof(double));
    cudaMalloc(&d_dist, N * sizeof(double));
    cudaMemcpy(d_xs, xs.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, ys.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zs, zs.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // initialize dist to "infinity"
    vector<double> hostDist(N, std::numeric_limits<double>::max());
    cudaMemcpy(d_dist, hostDist.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    vector<size_t> selectedIndices;

    fps(N, K, xs, ys, zs, d_xs, d_ys, d_zs, d_dist, selectedIndices, hostDist);

    // printing selected indices
    for (auto index : selectedIndices)
    {
        cout << index << " ";
    }

    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_zs);
    cudaFree(d_dist);

    return 0;
}