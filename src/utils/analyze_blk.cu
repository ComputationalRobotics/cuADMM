/*

    utils/analyze_blk.cu

    Analyze the blk vector to determine hyperparameters for the ADMM algorithm.

*/

#include <set>
#include <unordered_map>
#include <iomanip>

#include "cuadmm/memory.h"
#include "cuadmm/utils.h"
#include "cuadmm/matrix_sizes.h"

// Analyze the blk vector to determine the size and number of PSD matrices
void analyze_blk(
    char* cpu_blk_types,
    HostDenseVector<int>& blk, 
    std::vector<int>& psd_blk_sizes,
    std::vector<int>& psd_blk_nums
) { 
    // first pass: get PSD matrix sizes 
    std::set<int> size_set;
    for (int i = 0; i < blk.size; i++) {
        if (cpu_blk_types[i] == 's') {
            size_set.insert(blk.vals[i]);
        }
    }
    psd_blk_sizes = std::vector<int>(size_set.begin(), size_set.end());

    // determine the size of the small and large matrices
    std::cout << "\nAnalysis of the blk vector:" << std::endl;
    
    // second pass: get PSD matrix numbers
    psd_blk_nums = std::vector<int>(psd_blk_sizes.size(), 0);
    for (int i = 0; i < blk.size; i++) {
        for (int j = 0; j < psd_blk_sizes.size(); j++) {
            if (cpu_blk_types[i] == 's' && blk.vals[i] == psd_blk_sizes[j]) {
                psd_blk_nums[j] = psd_blk_nums[j] + 1;
            }
        }
    }

    // print unconstrained variables
    for (int i = 0; i < blk.size; i++) {
        if (cpu_blk_types[i] == 'u') {
            std::cout << "     " << std::setw(4) << 1 << " u. block of size " << std::setw(3) << blk.vals[i];
            std::cout << std::endl;
        }
    }

    // print the PSD matrices of the map
    for (int i = 0; i < psd_blk_sizes.size(); i++) {
        std::cout << "     " << std::setw(4) << psd_blk_nums[i] << " matrices of size " << std::setw(4) << psd_blk_sizes[i];
        if (MatrixSizes::get_size_category(psd_blk_sizes[i]) == MatrixSizeCategory::LARGE) {
            std::cout << " (large)";
        } else if (MatrixSizes::get_size_category(psd_blk_sizes[i]) == MatrixSizeCategory::MEDIUM) {
            std::cout << " (medium)";
        } else {
            std::cout << " (small)";
        }
        std::cout << std::endl;
    }

    return;
}