/*

    block_structure.h

    Analyzes the block structure (blk) of an SDP in a single place: sizes and numbers of the PSD
    blocks, the small/medium/large matrix layouts, the length of the vectorized representation,
    and the maps between the vectorized representation and the dense matrix buffers.

    It computes the same values as analyze_blk + MatrixSizes::init + get_maps, but validates the
    input and checks every size and offset against int overflow.

*/

#ifndef CUADMM_BLOCK_STRUCTURE_H
#define CUADMM_BLOCK_STRUCTURE_H

#include <iostream>
#include <vector>

#include "cuadmm/matrix_sizes.h"

// Description of one block of the blk vector
struct BlockInfo
{
    char type;                   // 's' (PSD), 'u' (free) or 'l' (nonnegative)
    int size;                    // side of the matrix for 's', number of entries for 'u' and 'l'
    MatrixSizeCategory category; // SMALL, MEDIUM or LARGE for 's', FREE for 'u', NONNEGATIVE for 'l'
    int vec_offset;              // start index of the block in the vectorized representation
    int vec_len;                 // number of entries in the vectorized representation: n(n+1)/2 for 's', n otherwise

    // PSD blocks only (-1 for 'u' and 'l' blocks)
    int size_index;      // index of the size in sizes.{small,medium,large}_mat_sizes
    int same_size_index; // number of PSD blocks of the same size that come before this one
    int mat_offset;      // offset of the n x n matrix in the {small,medium,large} matrix buffer
    int W_offset;        // offset of the n eigenvalues in the {small,medium,large} W vector
};

class BlockStructure
{
public:
    // largest PSD block size n such that n * n fits in an int
    static constexpr int MAX_PSD_BLK_SIZE = MAX_PSD_BLOCK_SIZE;

    std::vector<BlockInfo> blocks;  // one entry per block, in input order
    std::vector<int> psd_blk_sizes; // distinct PSD block sizes, in increasing order
    std::vector<int> psd_blk_nums;  // number of PSD blocks of each size
    MatrixSizes sizes;              // small/medium/large layouts (the eig buffer start indices are left empty)
    int vec_len;                    // length of the vectorized representation

    // Maps for the vectorized representation, of length vec_len. For the entry of a PSD block
    // that holds the upper-triangle coefficient (j, i), j <= i:
    // - map_B: MatrixSizeCategory of the block the entry belongs to (FREE / NONNEGATIVE for 'u' / 'l')
    // - map_M1: index of (j, i) in the column-major matrix buffer of the block's category
    // - map_M2: index of the mirrored coefficient (i, j) in the same buffer
    // map_M1 and map_M2 are 0 for the entries of 'u' and 'l' blocks.
    std::vector<int> map_B;
    std::vector<int> map_M1;
    std::vector<int> map_M2;

    BlockStructure(); // structure of an empty blk vector
    BlockStructure(const char *blk_types, const int *blk_sizes, const int blk_num);

    // Analyze blk_num blocks of types blk_types ('s', 'u' or 'l') and sizes blk_sizes.
    // Throws std::invalid_argument on an unknown block type or a non-positive size (or a negative
    // blk_num, or null pointers with blk_num > 0), and std::overflow_error if a PSD block is larger
    // than MAX_PSD_BLK_SIZE or if vec_len or the total size of a matrix buffer does not fit in an
    // int. Nothing is modified when it throws; otherwise the previous content is fully replaced.
    void init(const char *blk_types, const int *blk_sizes, const int blk_num);

    // Print the analysis of the blk vector (same output as analyze_blk)
    void print(std::ostream &os = std::cout) const;
};

#endif // CUADMM_BLOCK_STRUCTURE_H
