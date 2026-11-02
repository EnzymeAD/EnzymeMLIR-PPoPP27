#include <vector>
using namespace std;

// Tensorize dsp/vsub.mlir: out[i] = a[i] - b[i] (separate output arg2)
vector<int> vsub(vector<int> a, vector<int> b) {
    vector<int> out;
    for (int i = 0; i < a.size(); i++) {
        out.push_back(a[i] - b[i]);
    }
    return out;
}
