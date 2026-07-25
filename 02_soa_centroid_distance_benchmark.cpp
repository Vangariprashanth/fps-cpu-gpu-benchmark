/*
 * This code is implemented as a part of iterative development towards fps-cpu-gpu-benchmark project.
It implements two simple point cloud workloads on cpu.
    1. Centroid computation
 *  2. Sum of Euclidean distances from a reference point
 */
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
using namespace std;

// Structure to hold contigous x,y,z coordinates of the point cloud
struct PointCloudSoA
{
    vector<double> x, y, z;
};

// computes centroid of the point cloud by summing the individual coordinates and dividing by N
PointCloudSoA computeCentroid(const PointCloudSoA &pointcloud)
{
    if (pointcloud.x.empty())
        throw std::runtime_error("point cloud is empty");

    double sum_x{}, sum_y{}, sum_z{};
    size_t size = pointcloud.x.size();
    for (size_t i = 0; i < size; i++)
    {
        sum_x += pointcloud.x[i];
        sum_y += pointcloud.y[i];
        sum_z += pointcloud.z[i];
    }
    return PointCloudSoA{{sum_x / size}, {sum_y / size}, {sum_z / size}};
}

// calcuate the sum of the distances from the given point to other points.
// This method serves no other purpose than just to iterate over the point cloud and some computations
double sumDistancesTo(const PointCloudSoA &cloud, double tx, double ty, double tz)
{
    if (cloud.x.empty())
        throw std::runtime_error("point cloud is empty");
    double sum{};

    for (size_t i = 0; i < cloud.x.size(); ++i)
    {
        double dx = tx - cloud.x[i];
        double dy = ty - cloud.y[i];
        double dz = tz - cloud.z[i];
        sum += sqrt((dx * dx) + (dy * dy) + (dz * dz));
    }
    return sum;
}

int main()
{
    const size_t N = 1'000'000;

    PointCloudSoA pointCloud;
    pointCloud.x.reserve(N);
    pointCloud.y.reserve(N);
    pointCloud.z.reserve(N);

    // Generating synthetic point cloud in the (i,2i,3i) format
    for (size_t i = 0; i < N; i++)
    {
        pointCloud.x.push_back(double(i));
        pointCloud.y.push_back(double(i) * 2.0f);
        pointCloud.z.push_back(double(i) * 3.0f);
    }

    // measuring the time taken for both the methods -
    auto t1 = std::chrono::high_resolution_clock::now();
    auto centroid = computeCentroid(pointCloud);
    auto sum = sumDistancesTo(pointCloud, 0.0, 0.0, 0.0);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    cout << "Centroid:" << " (x: " << centroid.x[0] << ", y: " << centroid.y[0] << ", z: " << centroid.z[0] << ")" << endl;

    cout << "Sum of the distance from point: " << "(x: " << 0.0 << ", y: " << 0.0 << ", z:" << 0.0 << ") is: " << sum << endl;

    cout << "Time taken for centroid and sum distances: " << duration.count() << " us\n";

    return 0;
}