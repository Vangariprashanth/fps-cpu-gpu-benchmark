/**
 * This code computes centroid and sum of the distances from a given point in the point cloud.
 * These methods are templated for both the soa and aos memory layout types.
 **/
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// Structure to hold a point in cartesian coordiantes
struct Point
{
    double x{}, y{}, z{};
};

// Structure representing Array of struture(point)
struct AOSCloud
{
    vector<Point> pointCloud;
    size_t size() const
    {
        return pointCloud.size();
    }
    double getX(size_t i) const
    {
        return pointCloud[i].x;
    }
    double getY(size_t i) const
    {
        return pointCloud[i].y;
    }
    double getZ(size_t i) const
    {
        return pointCloud[i].z;
    }
};

// structure representing structure of contigous coordinates
struct SOACloud
{
    vector<double> xs, ys, zs;
    size_t size() const
    {
        return xs.size();
    }
    double getX(size_t i) const
    {
        return xs[i];
    }
    double getY(size_t i) const
    {
        return ys[i];
    }
    double getZ(size_t i) const
    {
        return zs[i];
    }
};

// templated centroid method for both layouts
template <typename Cloud>
Point computeCentroid(const Cloud &c)
{
    size_t size = c.size();
    if (size == 0)
        throw std::runtime_error("point cloud is empty");

    double sum_x{}, sum_y{}, sum_z{};
    for (size_t i = 0; i < size; i++)
    {
        sum_x += c.getX(i);
        sum_y += c.getY(i);
        sum_z += c.getZ(i);
    }
    return Point{sum_x / size, sum_y / size, sum_z / size};
}

// templated method for both soa and aos layouts
template <typename Cloud>
double sumDistancesTo(const Cloud &c, Point target)
{
    size_t size = c.size();
    if (size == 0)
        throw std::runtime_error("point cloud is empty");

    double sum{};
    for (size_t i = 0; i < size; i++)
    {
        double dx = target.x - c.getX(i);
        double dy = target.y - c.getY(i);
        double dz = target.z - c.getZ(i);
        sum += sqrt(dx * dx + dy * dy + dz * dz);
    }
    return sum;
}

int main()
{
    const size_t N = 1'000'000;
    vector<Point> pointcloud;
    pointcloud.reserve(N);

    SOACloud soaCloud;
    AOSCloud aosCloud;

    // synthetic point cloud generation and initializing both the layouts
    for (size_t i = 0; i < N; i++)
    {
        double x = double(i);
        double y = double(i) * 2.0;
        double z = double(i) * 3.0;
        aosCloud.pointCloud.push_back(Point{x, y, z});
        soaCloud.xs.push_back(x);
        soaCloud.ys.push_back(y);
        soaCloud.zs.push_back(z);
    }

    // calling centroid method on aos layout
    auto aosCentroidPoint = computeCentroid(aosCloud);
    auto aosSumDistances = sumDistancesTo(aosCloud, Point{0.0, 0., 0.});

    cout << "---AOS----" << endl;
    cout << "Centroid Point: " << aosCentroidPoint.x << " " << aosCentroidPoint.y << " " << aosCentroidPoint.z << endl;
    cout << "Sum Distances: " << aosSumDistances << endl;

    auto soaCentroidPoint = computeCentroid(soaCloud);
    auto soaSumDistances = sumDistancesTo(soaCloud, Point{0.0, 0., 0.});

    cout << "---SOA----" << endl;
    cout << "Centroid Point: " << soaCentroidPoint.x << " " << soaCentroidPoint.y << " " << soaCentroidPoint.z << endl;
    cout << "Sum Distances: " << soaSumDistances << endl;

    return 0;
}