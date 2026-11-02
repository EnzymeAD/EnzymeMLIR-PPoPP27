#include <vector>
using namespace std;

// Tensorize dsp/vscal.mlir: a[i] = a[i] * s (in place)
vector<int> vscal(vector<int> a, int s) {
    vector<int> out;
    for (int i = 0; i < a.size(); i++) {
        out.push_back(a[i] * s);
    }
    return out;
}
