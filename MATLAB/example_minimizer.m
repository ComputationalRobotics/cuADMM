clear; close all;

% no need to generate a path
addpath("./build");

rng("default")

nb_blocks = 1;
n = 3;
m = 3;
vec_len = 3*4/2;

blk = cell(nb_blocks, 2);
blk{1, 1} = 's';
blk{1, 2} = n;

At = cell(nb_blocks, 1);
At{1, 1} = rand(vec_len, m);
At{1, 1} = sparse(At{1, 1});

C = cell(nb_blocks, 1);
C{1, 1} = rand(vec_len, 1);
C{1, 1} = sparse(C{1, 1});

b = rand(m, 1);
b = sparse(b);

At_stack = from_cell_to_array(At);
C_stack = from_cell_to_array(C);
blk_vec = [];
for i = 1: size(blk, 1)
    blk_vec = [blk_vec; blk{i, 2}];
end
vec_len = size(At_stack, 1);
con_num = size(At_stack, 2);
X_new = zeros(vec_len, 1);
y_new = zeros(con_num, 1);
S_new = zeros(vec_len, 1);
sig_new = 2e2;

% device_num_requested = 2;
eig_stream_num_per_gpu = 12;
max_iter = 2e3;
stop_tol = 1e-5;

[X_out, y_out, S_out, sig_out] = cuadmm_MATLAB(eig_stream_num_per_gpu,...
                                                max_iter, stop_tol,...
                                                At_stack, b, C_stack, blk_vec,...
                                                X_new, y_new, S_new, sig_new);

disp(X_out);

X_opt_vec = X_out;
Xs = cell(size(blk, 1), 1);
Xs{1,1} = zeros(blk{1, 2}, blk{1, 2});

X_opt = cell(Xs);
offset = 0;
for k = 1: length(Xs)
    X_opt{k} = zeros(size(Xs{k}));
    n = size(Xs{k}, 1);
    % convert from svec back to full matrix
    count = 1;
    for i = 1:n
        for j = 1:i
            X_opt{k}(i, j) = X_opt_vec(offset + count);
            if i ~= j
                X_opt{k}(i, j) = X_opt{k}(i, j) / sqrt(2);
                X_opt{k}(j, i) = X_opt{k}(i, j);
            end

            count = count + 1;
        end
    end
    offset = offset + n * (n + 1) / 2;
end

disp(X_opt{1,1});
    


function v = from_cell_to_array(c)
    v = [];
    for i = 1: length(c)
        v = [v; c{i}];
    end
end
