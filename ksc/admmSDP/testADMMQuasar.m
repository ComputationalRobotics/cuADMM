clc;
close all;
clear;
warning off;

addpath(genpath(pwd));
%quasarpath = '/Users/lingliang/Documents/MATLL/STRIDE/sdp-solver-certifiable-perception/QUASAR/data';
quasarpath = 'E:\MATLL\sdp-solver-certifiable-perception-master\QUASAR\data';
addpath(quasarpath);

ds = [50, 100, 200, 500];
runs = 2;
runADMM = 1;


for didx = 2%:length(ds)
    for runidx = 1:length(runs)
        N = ds(didx);
        run = runs(runidx);
        datapath = sprintf('%s/quasar_%d_%d.mat',quasarpath,N,run);
        load(datapath);
        
        if (runADMM == 1)
            optionsADMM.maxIter = 10000;
            optionsADMM.stopTol = 1e-6;
            optionsADMM.printyes = 1;
            optionsADMM.scaleA = 1;
            optionsADMM.scaleData = 1;
            optionsADMM.sig = 1e0;
            optionsADMM.partialProj = 1;
            optionsADMM.useLowRankEigs = 0;
            [X, y, S, infoADMM] = ADMM(SDP.blk, {SDP.At}, full(SDP.b), {SDP.C}, optionsADMM);
        end
        
    end
end