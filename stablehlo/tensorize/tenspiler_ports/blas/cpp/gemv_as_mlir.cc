#include <vector>
using namespace std;

// Tensorize blas/gemv.mlir as written (loads x[row], not x[col]):
// y[i] = sum_j A[i][j] * x[i]  ==  x[i] * sum_j A[i][j]
vector<int> gemv_as_mlir(vector<vector<int>> A, vector<int> x) {
    vector<int> y;
    int m = A.size();
    int n = A[0].size();
    for (int row = 0; row < m; row++) {
        int curr = 0;
        for (int col = 0; col < n; col++) {
            curr += A[row][col] * x[row];
        }
        y.push_back(curr);
    }
    return y;
}
