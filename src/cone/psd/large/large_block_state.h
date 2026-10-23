/*

    cone/psd/large/large_block_state.h

    LargeBlockState (plan Section 10.4): what persists per large block between iterations: the projection mode,
    the LOBPCG pair count, the ranks and rank tolerance of the last rank analysis, and the warm start V (n x
    k_alloc), D (k_alloc). In Phase 6a the storage is the engine's (large_mode, large_k, large_rank_tol,
    cpu_*_ranks, lobpcg_P, lobpcg_W): the state is a view on it, built per block by PsdCone; Phase 6b moves the
    storage into the cone with the arena. LargeBlockJob is one block's projection: its position in the engine's
    size groups and buffers, and its state.

*/

#ifndef CUADMM_CONE_PSD_LARGE_LARGE_BLOCK_STATE_H
#define CUADMM_CONE_PSD_LARGE_LARGE_BLOCK_STATE_H

#include "cone/psd/large/large_block_policy.h" // LargeMode

namespace cuadmm
{
namespace cone
{
namespace psd
{

struct LargeBlockState
{
    int idx;               // flat index among the large blocks
    int &mode;             // LargeMode (the engine's large_mode[idx])
    int &k;                // LOBPCG pairs of the current mode (large_k[idx])
    double &rank_tol;      // rank tolerance of the last analysis (large_rank_tol[idx])
    int k_alloc;           // columns allocated in V and D (0 without the low-rank path)
    int *d_positive_rank;  // device counters of the rank analysis (null without the low-rank path)
    int *d_negative_rank;
    int &positive_rank;    // their host copies
    int &negative_rank;
    double *V;             // warm start: eigenvectors (n x k_alloc, column-major) and eigenvalues (k_alloc)
    double *D;
};

struct LargeBlockJob
{
    int size_index;      // index of the block's size group in the engine's MatrixSizes
    int index_in_group;  // index of the block within its size group
    int n;               // the block's size
    int mat_offset;      // offset of the block in large_mat / large_mat_tmp / large_mat_P
    int W_offset;        // offset of its eigenvalues in large_W
    LargeBlockState state;
};

} // namespace psd
} // namespace cone
} // namespace cuadmm

#endif // CUADMM_CONE_PSD_LARGE_LARGE_BLOCK_STATE_H
