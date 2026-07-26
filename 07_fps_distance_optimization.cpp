/**
 * This code implements optimized farthest point sampling using distance array.
 * Distance array stores the nearest distance out of all the selected indicies.
 * Input: Point Cloud - vector<double> (N)
 * output: K selected indicies - vector<int> (K)
 * Time complexity: - O(k*N) -> O(N*k^2)
 * **/
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;

int main()
{
    const int N = 12;
    const int K = 4;
    const int int_max = std::numeric_limits<int>::max();
    auto pointCloud = vector<int>(N);
    auto distanceToNearestSelected = vector<int>(N, int_max);
    iota(pointCloud.begin(), pointCloud.end(), 1);
    vector<int> selectedIndices;

    // initialization
    int lastSelectedIndex = 10;
    selectedIndices.push_back(lastSelectedIndex);

    // establising the distance array - this array tells us the nearest distance to this point from all the selected indices

    for (int i = 0; i < N; ++i)
    {
        distanceToNearestSelected[i] = abs(pointCloud[i] - pointCloud[lastSelectedIndex]);
    }

    for (int r = 1; r < K; r++)
    {
        int nextFarthestDistance = 0;
        int nextFarthestIndex = 0;
        // find the new farthest point using the whole point cloud and the selected indices so far
        for (int currentPointIndex = 0; currentPointIndex < N; ++currentPointIndex)
        {
            // find the next farthest point by taking the maximum and then get the index of it.
            if (distanceToNearestSelected[currentPointIndex] > nextFarthestDistance)
            {
                nextFarthestDistance = distanceToNearestSelected[currentPointIndex];
                nextFarthestIndex = currentPointIndex;
            }
        }
        selectedIndices.push_back(nextFarthestIndex);
        for (int currentPointIndex = 0; currentPointIndex < N; ++currentPointIndex)
        {
            distanceToNearestSelected[currentPointIndex] = min(distanceToNearestSelected[currentPointIndex], abs(pointCloud[nextFarthestIndex] - pointCloud[currentPointIndex]));
        }
    }

    cout << "Point Cloud Size: " << N << endl;
    cout << "Point Cloud: " << endl;
    for (auto point : pointCloud)
        cout << point << " ";

    cout << endl;
    cout << "K (sample size):" << K << endl;
    cout << "Selected Indices: " << endl;
    for (int i = 0; i < K; i++)
    {
        cout << selectedIndices[i] << " ";
    }
    cout << endl;

    return 0;
}