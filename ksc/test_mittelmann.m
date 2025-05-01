clear; close all;

addpath("./pathinfo/");
ksc;
% jordan;
pathinfo("lib") = "./lib";
pathinfo("sos-sdp-conversion") = "./sos-sdp-conversion";
pathinfo("utils") = "./utils";
pathinfo("admmsdp") = "./admmSDP";
pathinfo("kscutils") = "./ksc-utils";
pathinfo("modules") = "./modules";

keys = pathinfo.keys;
for i = 1: length(keys)
    key = keys(i);
    addpath(genpath(pathinfo(key)));
end

% [blk, At, C, b] = read_sdpa("./data/biggs.dat-s");

% optionsADMM.maxIter = 5000;
% optionsADMM.stopTol = 1e-4;
% optionsADMM.printyes = 1;
% optionsADMM.scaleA = 1;
% optionsADMM.scaleData = 1;
% optionsADMM.sig = 1e2;
% optionsADMM.partialProj = 0;
% optionsADMM.useLowRankEigs = 0;
% [X, y, S, infoADMM] = ADMM(blk, At, b, C, optionsADMM);

% prob = sdpt2mosek_seiveSDP(blk, At, C, b);

load("~/ksc/my-packages/cuADMM/examples/SPOT/data/MOSEK/pushbox-10.mat");

[~, res] = mosekopt('minimize info', prob);
prob.blx = [];
prob = column2row_recursive(prob);



[A_sedumi, b_sedumi, c_sedumi, K_sedumi] = convert_mosek2sedumi_seiveSDP(prob);
prob_1 = convert_sedumi2mosek_seiveSDP(A_sedumi, b_sedumi, c_sedumi, K_sedumi);
[~, res_1] = mosekopt('minimize info', prob_1);



%% helper functions
function S = column2row_recursive(S)
    % Recursively convert every numeric column vector in S to a row vector.

    flds = fieldnames(S);

    for k = 1:numel(flds)
        v = S.(flds{k});

        if isstruct(v)                %–– nested struct or struct array ––
            % Apply the function to every element in the struct array:
            S.(flds{k}) = arrayfun(@column2row_recursive, v);

        elseif isnumeric(v) && isvector(v) && size(v,2) == 1
            S.(flds{k}) = v.';        %–– flip column → row ––
        end
        % (Anything else—cell arrays, strings, matrices—stays untouched.)
    end
end






