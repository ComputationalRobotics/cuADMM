%% This file showcases how to use the MATLAB interface with a problem in MOSEK format.

clear; close all;
addpath("./build");
addpath("../examples/utils");
addpath("../examples/mexfiles")

% loads the variable "prob" containing the problem in MOSEK format
load("../examples/SPOT/data/MOSEK/PushBot_N=1_MOMENT.mat");

%% Convert the problem to SeDuMi format
problem = column2row_recursive(prob);
problem.blx = ones(size(problem.c'));

% handle the case where the problem does not have a barc field
no_barc = ~isfield(problem, 'barc');
if no_barc
    problem.barc.subj = [];
    problem.barc.subl = [];
    problem.barc.subk = [];
    problem.barc.val = [];
    problem.c = -problem.c';
end
[A, b, c, K] = convert_mosek2sedumi(problem);
sedumi.A = A;
sedumi.b = b;
sedumi.c = c;
sedumi.K = K;

%% Convert SeDuMi to SDPT3 format
[sdpt3_blk, sdpt3_At, sdpt3_C, sdpt3_b, ~] = read_sedumi(sedumi.A, sedumi.b, sedumi.c, sedumi.K, 0);
sdpt3.At = sdpt3_At;
sdpt3.C = sdpt3_C;
sdpt3.b = sdpt3_b;
sdpt3.blk = sdpt3_blk;

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
max_iter = 2e2;
stop_tol = 1e-3;

[X_out, y_out, S_out, sig_out] = cuadmm_MATLAB(eig_stream_num_per_gpu,...
                                                max_iter, stop_tol,...
                                                At, b, C, blk,...
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