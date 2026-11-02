#include <vector>
using namespace std;

// Tensorize dsp/voffset.mlir: a[i] = a[i] + c (in place)
vector<int> voffset(vector<int> a, int c) {
    vector<int> out;
    for (int i = 0; i < a.size(); i++) {
        out.push_back(a[i] + c);
    }
    return out;
}
