#include <vector>
using namespace std;

// Tensorize dsp/w_vec.mlir: out[i] = s * a[i] + b[i] (args: a, b, s, out)
vector<int> w_vec(vector<int> a, vector<int> b, int s) {
    vector<int> out;
    for (int i = 0; i < a.size(); i++) {
        out.push_back(s * a[i] + b[i]);
    }
    return out;
}
