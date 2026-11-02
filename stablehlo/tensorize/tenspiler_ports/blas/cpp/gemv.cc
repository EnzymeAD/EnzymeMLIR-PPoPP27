#include <vector>
using namespace std;

// True GEMV: y[i] = sum_j A[i][j] * x[j]
vector<int> gemv(vector<vector<int>> A, vector<int> x) {
    vector<int> y;
    int m = A.size();
    int n = x.size();
    for (int row = 0; row < m; row++) {
        int curr = 0;
        for (int col = 0; col < n; col++) {
            curr += A[row][col] * x[col];
        }
        y.push_back(curr);
    }
    return y;
}
