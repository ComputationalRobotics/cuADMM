%% This file showcases how to retrieve the primal solution from cuADMM.

clear; close all;

addpath("./build");
addpath("../examples/utils");

rng("default")

nb_blocks = 1;    % one block
n = 3;            % of size 3x3
con_num = 3;      % number of constraints
vec_len = 3*4/2;  % size of the vectorized matrix (symmetric matrix)

% build the block structure
blk = cell(nb_blocks, 2);
blk{1, 1} = 's';
blk{1, 2} = n;

% generate random constraint matrix
At = cell(nb_blocks, 1);
At{1, 1} = rand(vec_len, vec_len); % vectorized format
At{1, 1} = sparse(At{1, 1});       % has to be sparse

% generate random cost matrix
C = cell(nb_blocks, 1);
C{1, 1} = rand(vec_len, 1); % vectorized format
C{1, 1} = sparse(C{1, 1});  % has to be sparse

b = rand(vec_len, 1); % in this case the vector is dense
b = sparse(b);        % has to be sparse

At_stack = from_cell_to_array(At);
C_stack = from_cell_to_array(C);
blk_vec = [];
for i = 1: size(blk, 1)
    blk_vec = [blk_vec; blk{i, 2}];
end
X_new = zeros(vec_len, 1);
y_new = zeros(con_num, 1);
S_new = zeros(vec_len, 1);
sig_new = 2e2;

eig_stream_num_per_gpu = 12; % number of eigenvalue streams per GPU
max_iter = 2e3;              % maximum number of iterations
stop_tol = 1e-5;             % stopping tolerance (KKT residuals)

[X_out, y_out, S_out, sig_out] = cuadmm_MATLAB(eig_stream_num_per_gpu,...
                                                max_iter, stop_tol,...
                                                At_stack, b, C_stack, blk_vec,...
                                                X_new, y_new, S_new, sig_new);

disp(X_out); % display the primal solution in svec format

X_opt = cell(size(blk, 1), 1);
offset = 0;
for k = 1: length(X_opt)
    n = blk{k,2};
    X_opt{k} = zeros(n, n);
    count = 1;
    % convert from svec back to full matrix
    for i = 1:n
        for j = 1:i
            X_opt{k}(i, j) = X_out(offset + count);
            if i ~= j
                X_opt{k}(i, j) = X_opt{k}(i, j) / sqrt(2);
                X_opt{k}(j, i) = X_opt{k}(i, j);
            end

            count = count + 1;
        end
    end
    offset = offset + n * (n + 1) / 2;
end

disp(X_opt{1,1}); % display the primal solution in full matrix format
