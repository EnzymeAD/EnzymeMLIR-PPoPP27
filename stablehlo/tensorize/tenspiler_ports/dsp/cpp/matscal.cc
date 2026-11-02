#include <vector>
using namespace std;

// Tensorize dsp/matscal.mlir: A[i][j] = A[i][j] * s (in place)
vector<vector<int>> matscal(vector<vector<int>> A, int s) {
    vector<vector<int>> out;
    for (int i = 0; i < A.size(); i++) {
        vector<int> row_vec;
        for (int j = 0; j < A[0].size(); j++) {
            row_vec.push_back(A[i][j] * s);
        }
        out.push_back(row_vec);
    }
    return out;
}
