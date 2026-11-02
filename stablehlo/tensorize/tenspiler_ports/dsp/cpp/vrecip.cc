#include <vector>
using namespace std;

// Tensorize dsp/vrecip.mlir: a[i] = 1.0 / a[i] (in place)
vector<int> vrecip(vector<int> a) {
    vector<int> out;
    for (int i = 0; i < a.size(); i++) {
        out.push_back(1 / a[i]);
    }
    return out;
}
