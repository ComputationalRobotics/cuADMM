%% This file showcases how to use the MATLAB interface with a problem in MOSEK format.

clear; close all;
addpath("./build");
addpath("../examples/utils");
addpath("../examples/mexfiles")

sdpt3 = load("../examples/sjtu-issues/N20_2.mat");
sdpt3 = sdpt3.N20;

%% Convert SDPT3 to cuADMM format
[At, b, C, blk] = sdpt3_to_cuadmm(sdpt3);
b = sparse(b); % don't forget to convert b to sparse format

%% Solve with cuADMM
vec_len = size(At, 1);
con_num = size(At, 2);
X_new = zeros(vec_len, 1);
y_new = zeros(con_num, 1);
S_new = zeros(vec_len, 1);
sig_new = 2e2;

eig_stream_num_per_gpu = 12;
max_iter = 1e3;
stop_tol = 1e-3;

blk_types = char(zeros(size(blk, 1), 1));
for i = 1: size(blk, 1)
    blk_types(i) = 's'; % SDP blocks
end

[X_out, y_out, S_out, sig_out] = cuadmm_MATLAB(eig_stream_num_per_gpu,...
                                                max_iter, stop_tol,...
                                                At, b, C, blk_types, blk,...
                                                X_new, y_new, S_new, sig_new);

function [cuda_At, cuda_b, cuda_C, cuda_blk] = sdpt3_to_cuadmm(sdpt3)
    cuda_At = from_cell_to_array(sdpt3.At);
    cuda_C = from_cell_to_array(svecADMM(sdpt3.blk, sdpt3.C));
    cuda_b = sdpt3.b;
    cuda_blk = zeros(size(sdpt3.blk, 1), 1);
    for i = 1: size(sdpt3.blk, 1)
        cuda_blk(i) = sdpt3.blk{i, 2};
    end
end