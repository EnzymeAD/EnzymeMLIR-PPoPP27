#include <vector>
using namespace std;

// Tensorize dsp/vneg.mlir: a[i] = a[i] * -1.0 (in place)
vector<int> vneg(vector<int> a) {
    vector<int> out;
    for (int i = 0; i < a.size(); i++) {
        out.push_back(a[i] * -1);
    }
    return out;
}
