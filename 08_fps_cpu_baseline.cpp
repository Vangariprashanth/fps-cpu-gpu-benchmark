/**
 * This code implements a baseline cpu version of FPS algorithm.
 * FPS method is templated so that both SOA and AOS layouts are executed using the same algorithm.
 * Achieving zero cost abstraction and isolating only the memory layout and keeping the rest of the algorithm same for both.
 * **/
#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <cmath>
using namespace std;

struct Point
{
    double x, y, z;
};

struct SOACloud
{
    vector<double> xs, ys, zs;

    double getX(size_t i) const
    {
        return xs[i];
    };
    double getY(size_t i) const
    {
        return ys[i];
    };

    double getZ(size_t i) const
    {
        return zs[i];
    }
};
struct AOSCloud
{
    vector<Point> pointCloud;
    double getX(size_t i) const
    {
        return pointCloud[i].x;
    };
    double getY(size_t i) const
    {
        return pointCloud[i].y;
    };

    double getZ(size_t i) const
    {
        return pointCloud[i].z;
    }
};

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

template <typename Cloud>
vector<size_t> farthestPointSampling(const Cloud &c, const size_t N, const size_t K)
{

    size_t selectedIndex = 5;
    vector<size_t> selectedIndices;

    selectedIndices.push_back(selectedIndex);
    auto nearestDistanceToSelected = vector<double>(N, std::numeric_limits<double>::max());

    double tx = c.getX(selectedIndex);
    double ty = c.getY(selectedIndex);
    double tz = c.getZ(selectedIndex);

    for (size_t currentPoint = 0; currentPoint < N; ++currentPoint)
    {
        double sx = c.getX(currentPoint);
        double sy = c.getY(currentPoint);
        double sz = c.getZ(currentPoint);
        nearestDistanceToSelected[currentPoint] = (sx - tx) * (sx - tx) + (sy - ty) * (sy - ty) + (sz - tz) * (sz - tz);
    }

    for (size_t r = 1; r < K; ++r)
    {
        size_t nextSelectedIndex = 0;
        double farthestDistanceSoFar{};
        for (size_t currentPoint = 0; currentPoint < N; ++currentPoint)
        {
            if (nearestDistanceToSelected[currentPoint] > farthestDistanceSoFar)
            {
                farthestDistanceSoFar = nearestDistanceToSelected[currentPoint];
                nextSelectedIndex = currentPoint;
            }
        }
        selectedIndices.push_back(nextSelectedIndex);
        double tx = c.getX(nextSelectedIndex);
        double ty = c.getY(nextSelectedIndex);
        double tz = c.getZ(nextSelectedIndex);
        for (size_t currentPoint = 0; currentPoint < N; ++currentPoint)
        {
            double sx = c.getX(currentPoint);
            double sy = c.getY(currentPoint);
            double sz = c.getZ(currentPoint);
            nearestDistanceToSelected[currentPoint] = min(nearestDistanceToSelected[currentPoint], (sx - tx) * (sx - tx) + (sy - ty) * (sy - ty) + (sz - tz) * (sz - tz));
        }
    }
    return selectedIndices;
}

int main()
{
    const size_t N = 1'00'000;
    const size_t K = 1024;

    AOSCloud aosCloud;
    SOACloud soaCloud;

    aosCloud.pointCloud.reserve(N);
    soaCloud.xs.reserve(N);
    soaCloud.ys.reserve(N);
    soaCloud.zs.reserve(N);

    for (size_t i = 0; i < N; i++)
    {
        aosCloud.pointCloud.push_back({double(i), double(i) * 2, double(i) * 3});
        soaCloud.xs.push_back(double(i));
        soaCloud.ys.push_back(double(2 * i));
        soaCloud.zs.push_back(double(3 * i));
    }

    auto soaResult = farthestPointSampling<SOACloud>(soaCloud, N, K);
    auto aosResult = farthestPointSampling<AOSCloud>(aosCloud, N, K);
    for (size_t i = 0; i < K; i++)
    {
        if (soaResult[i] != aosResult[i])
            cout << "Error at index: " << i << endl;
    }

    auto soaTime = timeIt([&]()
                          { return farthestPointSampling<SOACloud>(soaCloud, N, K); });

    auto aosTime = timeIt([&]()
                          { return farthestPointSampling<AOSCloud>(aosCloud, N, K); });

    cout << "FPS algorithm for SOA layout: " << soaTime << " us" << endl;
    cout << "FPS algorithm for Aos layout: " << aosTime << " us" << endl;

    // for (auto index : soaResult)
    // {
    //     cout << index << " ";
    // }
    // cout << endl;

    return 0;
}