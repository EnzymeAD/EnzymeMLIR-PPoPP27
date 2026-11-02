#include <vector>
using namespace std;

// Tensorize dsp/matmul.mlir: C[i][j] = sum_k A[i][k] * B[k][j] (C = arg2, zero-initialised)
vector<vector<int>> matmul(vector<vector<int>> A, vector<vector<int>> B) {
    vector<vector<int>> out;
    int m = A.size();
    int n = B[0].size();
    int p = B.size();
    for (int i = 0; i < m; i++) {
        vector<int> row_vec;
        for (int j = 0; j < n; j++) {
            int curr = 0;
            for (int k = 0; k < p; k++) {
                curr += A[i][k] * B[k][j];
            }
            row_vec.push_back(curr);
        }
        out.push_back(row_vec);
    }
    return out;
}
