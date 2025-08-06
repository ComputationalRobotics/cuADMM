%% This file showcases how to use the MATLAB interface with a problem in SDPT3 format.

clear; close all;
addpath("./build");
addpath("../examples/utils");
addpath("../examples/mexfiles")

sdpt3 = load("/home/jordan/antoine/sdp_problems/sdpt3/rose13.mat");

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
max_iter = 1e8;
stop_tol = 1e-4;

[X_out, y_out, S_out, sig_out] = cuadmm_MATLAB(eig_stream_num_per_gpu,...
                                                max_iter, stop_tol,...
                                                At, b, C, blk,...
                                                X_new, y_new, S_new, sig_new);

function [cuda_At, cuda_b, cuda_C, cuda_blk] = sdpt3_to_cuadmm(sdpt3)
    if iscell(sdpt3.At)
        cuda_At = from_cell_to_array(sdpt3.At);
    else
        cuda_At = sdpt3.At;
    end
    if iscell(sdpt3.C)
        cuda_C = from_cell_to_array(svecADMM(sdpt3.blk, sdpt3.C));
    else
        % wrong size?
        cuda_C = sdpt3.C;
    end
    cuda_b = sdpt3.b;
    cuda_blk = zeros(size(sdpt3.blk, 1), 1);
    for i = 1: size(sdpt3.blk, 1)
        if sdpt3.blk{i, 1} == 's'
            cuda_blk(i) = sdpt3.blk{i, 2};
        else
            error("Free variables are not supported by the MATLAB interface yet.");
        end
    end
end