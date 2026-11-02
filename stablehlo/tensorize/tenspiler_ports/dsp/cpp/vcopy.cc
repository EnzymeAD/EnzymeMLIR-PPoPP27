#include <vector>
using namespace std;

// Tensorize dsp/vcopy.mlir: out[i] = a[i] (arg0 copied into arg1)
vector<int> vcopy(vector<int> a) {
    vector<int> out;
    for (int i = 0; i < a.size(); i++) {
        out.push_back(a[i]);
    }
    return out;
}
