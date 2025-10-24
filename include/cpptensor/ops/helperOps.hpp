#pragma once

#include <vector>
#include <stdexcept>


//| A.shape | B.shape | Output                                                                   | Notes                                        |
//| ------- | ------- | ------------------------------------------------------------------------ | -------------------------------------------- |
//| [4,3]   | [3]     | [4,3]                                                                    | Standard broadcast                           |
//| [1,3]   | [4,3]   | [4,3]                                                                    | A broadcast along 0                          |
//| [4,1,6] | [7,1,5] | [7,4,6,5]? No, [7,1,6,5]? Actually: [7,4,6,5]? Wait; align trailing dims | Correctly [7,4,6,5]? (align trailing 3 dims) |
//| [5,4,3] | [3]     | [5,4,3]                                                                  | B broadcast along leading dims               |
//| [3,2]   | [4,3,2] | [4,3,2]                                                                  | A padded to [1,3,2]                          |


// Helper: compute broadcasted output shape for two input shapes
static std::vector<size_t> computeBroadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t n = std::max(na, nb);
    std::vector<size_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        // Align from the end; if one shape is shorter, treat missing dims as 1
        size_t dimA = (i < n-na ? 1 : a[i-(n-na)]);
        size_t dimB = (i < n-nb ? 1 : b[i-(n-nb)]);
        if (dimA != dimB && dimA != 1 && dimB != 1) {
            throw std::runtime_error("Shapes not broadcastable");
        }
        out[i] = (dimA > dimB ? dimA : dimB);
    }
    return out;
}