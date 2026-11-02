#include <vector>
using namespace std;

// Tensorize dsp/matinit.mlir (func is named @matadd in the MLIR): A[i][j] = c (in place; A only gives the shape)
vector<vector<int>> matinit(vector<vector<int>> A, int c) {
    vector<vector<int>> out;
    for (int i = 0; i < A.size(); i++) {
        vector<int> row_vec;
        for (int j = 0; j < A[0].size(); j++) {
            row_vec.push_back(c);
        }
        out.push_back(row_vec);
    }
    return out;
}
