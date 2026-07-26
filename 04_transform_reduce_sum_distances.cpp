/**
 * This code implements the same point cloud methods but using transform and reduce methods
 *
 * **/
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
using namespace std;

struct Point
{
    double x{}, y{}, z{};
};

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

template <typename Cloud>
double sumDistancesTo(const Cloud &c, const Point target)
{
    size_t size = c.size();
    if (size == 0)
        throw runtime_error("point cloud size is empty");
    vector<size_t> idx(size);
    iota(idx.begin(), idx.end(), 0);

    return transform_reduce(
        idx.begin(), idx.end(), 0.0, plus<>{},
        [&](size_t i)
        {
            double dx = target.x - c.getX(i);
            double dy = target.y - c.getY(i);
            double dz = target.z - c.getZ(i);

            return std::sqrt(dx * dx + dy * dy + dz * dz);
        });
}

template <typename T>
void doNotOptimizeAway(const T &value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

template <typename F>
double timeIt(F &&f, int reps = 20)
{
    f(); // warm up
    auto best = 1e18;
    for (int i = 0; i < reps; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto result = f();
        auto t2 = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000.0;
        best = std::min(best, us);
        doNotOptimizeAway(result);
    }
    return best;
}

int main()
{
    const size_t N = 1'000'000;
    vector<Point> pointcloud;
    pointcloud.reserve(N);

    SOACloud soaCloud;
    AOSCloud aosCloud;

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

    auto soa_centroid_time = timeIt([&]()
                                    { return computeCentroid(soaCloud); });

    auto aos_centoid_time = timeIt([&]()
                                   { return computeCentroid(aosCloud); });

    auto aos_sum_distances_time = timeIt([&]()
                                         { return sumDistancesTo(aosCloud, Point{0.0, 0., 0.}); });

    auto soa_sum_distances_time = timeIt([&]()
                                         { return sumDistancesTo(soaCloud, Point{0.0, 0., 0.}); });

    // auto aosCentroidPoint = computeCentroid(aosCloud);
    auto aosSumDistances = sumDistancesTo(aosCloud, Point{0.0, 0., 0.});
    // cout << aosCentroidPoint.x << " " << aosCentroidPoint.y << " " << aosCentroidPoint.z << endl;
    cout << "---AOS----" << endl;
    cout << "Sum Distances: " << aosSumDistances << endl;

    // auto soaCentroidPoint = computeCentroid(soaCloud);
    auto soaSumDistances = sumDistancesTo(soaCloud, Point{0.0, 0., 0.});

    cout << "---SOA----" << endl;
    // cout << "Centroid Point: " << soaCentroidPoint.x << " " << soaCentroidPoint.y << " " << soaCentroidPoint.z << endl;
    cout << "Sum Distances: " << soaSumDistances << endl;
}