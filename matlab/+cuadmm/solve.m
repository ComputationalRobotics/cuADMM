function [X, y, S, info] = solve(At, b, C, K, opts)
%CUADMM.SOLVE  Solve a conic program with cuADMM.
%   [X, y, S, info] = cuadmm.solve(At, b, C, K, opts)
%     At (vec_len x m, sparse), b (m x 1), C (vec_len x 1) in svec form; K the cones (fields f, l, s in that
%     order, or fields types and sizes for an arbitrary order); opts a struct of nested option paths
%     (opts.penalty.policy = 'fixed', opts.algorithm.max_iter = 1000, ...; see docs/OPTIONS.md) with the optional
%     warm start opts.warm_start.X / y / S.
    if nargin < 4 || nargin > 5
        error('cuADMM:invalidInput', 'usage: cuadmm.solve(At, b, C, K, opts)');
    end
    if nargin == 4
        opts = struct();
    end
    [X, y, S, info] = cuadmm_mex(At, b, C, K, opts);
end
