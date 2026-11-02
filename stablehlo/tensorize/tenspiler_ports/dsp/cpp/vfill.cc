#include <vector>
using namespace std;

// Tensorize dsp/vfill.mlir: a[i] = c (in place; a only gives the length)
vector<int> vfill(vector<int> a, int c) {
    vector<int> out;
    for (int i = 0; i < a.size(); i++) {
        out.push_back(c);
    }
    return out;
}
