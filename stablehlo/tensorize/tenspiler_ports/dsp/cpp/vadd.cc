#include <vector>
using namespace std;

// Tensorize dsp/vadd.mlir: a[i] = a[i] + b[i] (in place on arg0)
vector<int> vadd(vector<int> a, vector<int> b) {
    vector<int> out;
    for (int i = 0; i < a.size(); i++) {
        out.push_back(a[i] + b[i]);
    }
    return out;
}
