clc
clear
close all
restoredefaultpath

load('../pendulum/data/N=15_licols.mat');
SDP = SDP.sdpt3;

addpath(genpath(pwd))

optionsADMM.maxIter = 5000;
optionsADMM.stopTol = 1e-6;
optionsADMM.printyes = 1;
optionsADMM.scaleA = 1;
optionsADMM.scaleData = 1;
optionsADMM.sig = 1e3;
optionsADMM.partialProj = 0;
optionsADMM.useLowRankEigs = 0;
[X, y, S, infoADMM] = ADMM(SDP.blk, SDP.At, SDP.b, SDP.C, optionsADMM);