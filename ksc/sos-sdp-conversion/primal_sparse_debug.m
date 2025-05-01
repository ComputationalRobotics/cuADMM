%# ok<*AGROW>
%# ok<*SAGROW>
clc; clear all; close all; restoredefaultpath;

spotpath = "~/ksc/matlab-install/spotless";
mosekpath = "~/ksc/matlab-install/mosek/10.1/toolbox/r2017a";
sdpt3path = "~/ksc/matlab-install/SDPT3-4.0";
libpath = "../lib";
utilspath = "~/ksc/matlab-install/lab-code/utils";

addpath(genpath(spotpath));
addpath(genpath(sdpt3path));
addpath(genpath(mosekpath));
addpath(genpath(libpath));
addpath(genpath(utilspath));

x = msspoly('x', 4);
x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

problem.vars = {[x1; x2], [x2; x3], [x3; x4]};
problem.objective = x1 + x2 + x3 + x4;
problem.inequality = {[x1], [x3], [x3; x4]};
problem.equality = {x1^2 + x2^2 - 1, x2^2 + x3^2 - 1, x3^2 + x4^2 - 1};
problem.relaxation_order = [2, 2, 2];
SDP = sparse_sdp_relax(problem);

% xopt = [sqrt(2)/2; sqrt(2)/2; sqrt(2)/2; sqrt(2)/2];
% v = monomials(x, 0: 2);

% Xmom = mykron(v, v);
% Xmom_opt = subs(Xmom, x, xopt);
% Xmom_opt = reshape(Xmom_opt, length(v), length(v));