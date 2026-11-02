#include <vector>
using namespace std;

// Tensorize blas/ger.mlir: A[i][j] += x[i] * y[j]; returns the updated matrix
vector<vector<int>> ger(vector<vector<int>> A, vector<int> x, vector<int> y) {
    vector<vector<int>> out;
    for (int i = 0; i < A.size(); i++) {
        vector<int> row_vec;
        for (int j = 0; j < A[0].size(); j++) {
            row_vec.push_back(A[i][j] + x[i] * y[j]);
        }
        out.push_back(row_vec);
    }
    return out;
}
