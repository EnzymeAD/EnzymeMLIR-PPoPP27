#include <vector>
using namespace std;

// Tensorize dsp/matadd.mlir: A[i][j] = A[i][j] + B[i][j] (in place)
vector<vector<int>> matadd(vector<vector<int>> A, vector<vector<int>> B) {
    vector<vector<int>> out;
    for (int i = 0; i < A.size(); i++) {
        vector<int> row_vec;
        for (int j = 0; j < A[0].size(); j++) {
            row_vec.push_back(A[i][j] + B[i][j]);
        }
        out.push_back(row_vec);
    }
    return out;
}
