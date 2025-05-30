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

// Analyze the blk vector to determine the following hyperparameters:
// - LARGE: the size of the large blocks (moment matrices)
// - SMALL: the size of the small blocks (localizing matrices)
// - mom_mat_num: the number of moment matrices
// - loc_mat_num: the number of localizing matrices
void analyze_blk_duo(
    HostDenseVector<int>& blk, 
    int* LARGE, int* SMALL, int* mom_mat_num, int* loc_mat_num
) { 
    // first pass: get matrix sizes 
    std::set<int> size_set;
    for (int i = 0; i < blk.size; i++) {
        size_set.insert(blk.vals[i]);
    }

    // determine the size of the moment and localizing matrices
    int min_size = *size_set.begin();
    int max_size = *size_set.rbegin();
    int mat_type_num = (int) size_set.size();
    std::cout << "\nAnalysis of the blk vector:" << std::endl;
    std::cout << "moment matrix size: " << max_size << std::endl;
    std::cout << "localizing matrix size: " << min_size << std::endl;
    if (mat_type_num != 2) {
        std::cerr << "SDP solver only supports two matrix sizes! ";
        std::cerr << "You matrix type number is: " << mat_type_num << std::endl;
    }
    assert(mat_type_num == 2);
    *LARGE = max_size;
    *SMALL = min_size;
    
    // second pass: get matrix numbers
    *mom_mat_num = 0;
    *loc_mat_num = 0;
    for (int i = 0; i < blk.size; i++) {
        if (blk.vals[i] == max_size) {
            *mom_mat_num += 1;
        } else {
            *loc_mat_num += 1;
        }
    }
    std::cout << "number of moment matrices: " << *mom_mat_num << std::endl;
    std::cout << "number of localizing matrices: " << *loc_mat_num << std::endl << std::endl;
    return;
}

// Analyze the blk vector to determine the size and number of PSD matrices
void analyze_blk(
    char* cpu_blk_types,
    HostDenseVector<int>& blk, 
    std::vector<int>& blk_sizes,
    std::vector<int>& blk_nums
) { 
    // first pass: get matrix sizes 
    std::set<int> size_set;
    for (int i = 0; i < blk.size; i++) {
        if (cpu_blk_types[i] == 's') {
            size_set.insert(blk.vals[i]);
        }
    }
    blk_sizes = std::vector<int>(size_set.begin(), size_set.end());

    // determine the size of the moment and localizing matrices
    std::cout << "\nAnalysis of the blk vector:" << std::endl;
    
    // second pass: get matrix numbers
    blk_nums = std::vector<int>(blk_sizes.size(), 0);
    for (int i = 0; i < blk.size; i++) {
        for (int j = 0; j < blk_sizes.size(); j++) {
            if (cpu_blk_types[i] == 's' && blk.vals[i] == blk_sizes[j]) {
                blk_nums[j] = blk_nums[j] + 1;
            }
        }
    }

    // print the elements of the map
    for (int i = 0; i < blk_sizes.size(); i++) {
        std::cout << "     " << std::setw(4) << blk_nums[i] << " matrices of size " << std::setw(3) << blk_sizes[i];
        if (is_large_mat(blk_sizes[i], blk_nums[i])) {
            std::cout << " (large)";
        } else {
            std::cout << " (small)";
        }
        std::cout << std::endl;
    }

    return;
}