/**
 * This code implements Farthest Point Sampling using brute force method.
 * This algorithm selects evenly spread out points in a point cloud.
 * Input: Point Cloud - vector<double> (N)
 * output: K selected indicies - vector<int> (K)
 * Time complexity: - O(k*N*k) -> O(N*k^2)
 * **/

#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>
using namespace std;

int main()
{
    const int N = 12;
    const int K = 4;
    auto pointCloud = vector<int>(N);
    iota(pointCloud.begin(), pointCloud.end(), 1);
    vector<int> selectedIndices;

    // initialization
    int lastSelectedIndex = 10;
    selectedIndices.push_back(lastSelectedIndex);

    for (int r = 1; r < K; r++)
    {
        int farthestPointDistanceSoFar = 0;
        int farthestPointDistanceSoFarIndex = 0;
        // find the new farthest point using the whole point cloud and the selected indices so far
        for (int currentPointIndex = 0; currentPointIndex < N; ++currentPointIndex)
        {

            //  nearest distance index from the current point compared to all the selected indices
            int nearestDistanceToSelectedIndices = 1e3;
            for (auto selectedIndex : selectedIndices)
            {
                int distance = abs(pointCloud[currentPointIndex] - pointCloud[selectedIndex]);
                if (distance < nearestDistanceToSelectedIndices)
                {
                    nearestDistanceToSelectedIndices = distance;
                }

            } // we got the nearestPoint to the currentPoint

            if (nearestDistanceToSelectedIndices > farthestPointDistanceSoFar)
            {
                farthestPointDistanceSoFar = nearestDistanceToSelectedIndices;
                farthestPointDistanceSoFarIndex = currentPointIndex;
            }
        }
        selectedIndices.push_back(farthestPointDistanceSoFarIndex);
    }

    cout << "Point Cloud Size: " << N << endl;
    cout << "Point Cloud: " << endl;
    for (auto point : pointCloud)
        cout << point << " ";

    cout << endl;
    cout << "K (sample size):" << K << endl;
    cout << "Selected Indices: " << endl;
    for (auto selectedIndex : selectedIndices)
        cout << selectedIndex << " ";
    return 0;
}