function [X, y, S, info] = solve(varargin)
%CUADMM.SOLVE  Solve a conic program with cuADMM-reorg.
%   [X, y, S, info] = cuadmm.solve(At, b, C, K, opts)
%     At (vec_len x m, sparse), b (m x 1), C (vec_len x 1) in svec form; K the cones (fields f, l, s in that
%     order, or fields types and sizes for an arbitrary order); opts a struct of nested option paths
%     (opts.penalty.policy = 'fixed', opts.algorithm.max_iter = 1000, ...; see docs/OPTIONS.md) with the optional
%     warm start opts.warm_start.X / y / S.
%
%   Positional shim (one release, plan decision Q6) for the signature of cuadmm_MATLAB of main:
%   [X, y, S, info] = cuadmm.solve(eig_stream_num_per_gpu, max_iter, stop_tol, At, b, C, blk_types, blk_sizes,
%                                   X0, y0, S0, sig, sig_update_threshold, sig_update_stage_1, sig_update_stage_2,
%                                   switch_admm, switch_proj_iter, switch_proj_tol, sigscale)
%   It maps the arguments onto the option struct (the defaults of the old MEX, 5000 for switch_proj_iter, become
%   psd.large.warmup.switch_iter, which only matters with a warm-up) and warns once.
    if nargin >= 4 && isstruct(varargin{4})
        if nargin == 4
            opts = struct();
        elseif nargin == 5
            opts = varargin{5};
        else
            error('cuADMM:invalidInput', 'usage: cuadmm.solve(At, b, C, K, opts)');
        end
        [X, y, S, info] = cuadmm_mex(varargin{1}, varargin{2}, varargin{3}, varargin{4}, opts);
        return;
    end
    if nargin < 12
        error('cuADMM:invalidInput', ['usage: cuadmm.solve(At, b, C, K, opts), or the positional signature of main ' ...
              '(at least 12 arguments)']);
    end
    warning('cuADMM:positionalSignature', ['the positional signature of cuadmm_MATLAB is kept for one release; ' ...
            'use cuadmm.solve(At, b, C, K, opts)']);
    streams = varargin{1}; max_iter = varargin{2}; stop_tol = varargin{3};
    At = varargin{4}; b = varargin{5}; C = varargin{6};
    blk_types = varargin{7}; blk_sizes = varargin{8};
    X0 = varargin{9}; y0 = varargin{10}; S0 = varargin{11}; sig = varargin{12};
    get = @(k, default) pick(varargin, k, default);
    K = struct('types', char(blk_types(:)'), 'sizes', double(blk_sizes(:)'));
    opts = struct();
    opts.psd.medium_streams = streams;
    opts.algorithm.max_iter = max_iter;
    opts.algorithm.tol = stop_tol;
    opts.penalty.sigma0 = sig;
    opts.penalty.sgs_update_threshold = get(13, 500);
    opts.penalty.sgs_update_stage_1 = get(14, 50);
    opts.penalty.sgs_update_stage_2 = get(15, 100);
    switch_admm = get(16, 0);
    if switch_admm > max_iter
        opts.algorithm.scheme = 'sgs_admm';
        opts.algorithm.tau = 1.95;
    else
        opts.algorithm.scheme = 'admm';
        opts.algorithm.scheme_switch_iteration = switch_admm;
    end
    opts.psd.large.warmup.switch_iter = get(17, 5000);
    opts.psd.large.warmup.switch_tol = get(18, 1e-2);
    opts.penalty.sgs_factor = get(19, 2.0);
    if any(X0(:)) || any(y0(:)) || any(S0(:))
        opts.warm_start = struct('X', double(X0(:)), 'y', double(y0(:)), 'S', double(S0(:)));
    end
    [X, y, S, info] = cuadmm_mex(At, b, C, K, opts);
end

function v = pick(args, k, default)
    if numel(args) >= k && ~isempty(args{k})
        v = args{k};
    else
        v = default;
    end
end
