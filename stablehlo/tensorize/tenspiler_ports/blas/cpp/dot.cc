#include <vector>
using namespace std;

// Tensorize blas/dot.mlir: out = sum_i a[i] * b[i]
int dot(vector<int> a, vector<int> b) {
    int sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
