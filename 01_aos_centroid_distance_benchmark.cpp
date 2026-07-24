// This code demonstrates Array of Structure memory layout of a point cloud
//  and measures the time taken to compute the basic operations such as centroid and sum of the distances from a given point.
//  This is a first attempt to build towards fps algorithm in aos and soa memory layout for cpu benchmarking.

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

// Represents a 3D point in cartesian coordinates
struct Point
{
    double x{}, y{}, z{};
};

// Computes the centroid of a given point cloud.
//  Input: vector of point structure
//  output: point structure
//  Time complexity: O(n)

Point computeCentroid(const vector<Point> &pointCloud)
{
    size_t size = pointCloud.size();
    if (pointCloud.empty())
    {
        throw std::runtime_error("computeCentroid: empty point cloud");
    }
    double x_sum{}, y_sum{}, z_sum{};
    for (const auto &point : pointCloud)
    {
        x_sum += point.x;
        y_sum += point.y;
        z_sum += point.z;
    }

    return Point{
        x_sum / size, y_sum / size, z_sum / size};
}

// Computes the sum of all the distances from a given point
//  Input: vector of point structure and a point
//  output: sum of the distances
//  Time complexity: O(n)

double sumDistancesTo(const std::vector<Point> &pointCloud, Point target)
{
    double sum{};
    for (const auto &point : pointCloud)
    {
        double dx = target.x - point.x;
        double dy = target.y - point.y;
        double dz = target.z - point.z;
        sum += sqrt(dx * dx + dy * dy + dz * dz);
    }
    return sum;
}

int main()
{

    const size_t N = 1'000'000;
    vector<Point> pointCloud;
    pointCloud.reserve(N);

    // Generate a synthetic point cloud in point(i,2i,3i) format of size N
    for (size_t i = 0; i < N; i++)
    {
        pointCloud.push_back({double(i), double(i) * 2, double(i) * 3});
    }

    // start time before calling the two methods
    auto t1 = std::chrono::high_resolution_clock::now();
    auto centroid = computeCentroid(pointCloud);
    auto point = Point{1.f, 2., 3.f};
    auto sumDistances = sumDistancesTo(pointCloud, point);
    auto t2 = std::chrono::high_resolution_clock::now();

    // total time taken (t2-t1)
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    cout << "Centroid:" << " (x: " << centroid.x << ", y: " << centroid.y << ", z: " << centroid.z << ")" << endl;

    cout << "Sum of the distance from point: " << "(x: " << point.x << ", y: " << point.y << ", z:" << point.z << ") is: " << sumDistances << endl;

    cout << "Time taken for centroid and sum distances: " << duration.count() << " us\n";
}