%%% This file showcases how to use the MATLAB interface with a "true_data" example.

clear; close all;

addpath("./build");
addpath("../examples/utils");

prefix = "/home/jordan/ksc/2023-traopt/pendulum/pendulum_archive/data/cuda_test/N=80/";
load(prefix + "true_data.mat");
At = true_data.At;
blk = true_data.blk;
C = true_data.C;
b = true_data.b;


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
max_iter = 2e2;
stop_tol = 1e-3;

[X_out, y_out, S_out, sig_out] = cuadmm_MATLAB(eig_stream_num_per_gpu,...
                                                max_iter, stop_tol,...
                                                At_stack, b, C_stack, blk_vec,...
                                                X_new, y_new, S_new, sig_new);
fprintf("|| X_out ||: %3.2e \n", norm(X_out));
fprintf("|| y_out ||: %3.2e \n", norm(y_out));
fprintf("|| S_out ||: %3.2e \n", norm(S_out));    
