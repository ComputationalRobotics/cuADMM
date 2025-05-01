function [Xp, par] = projsdpADMM(blk, X, par)
% matrices are stored in vectorized format by default
if (~isfield(par, 'smtol'))
    par.smtol = 1e-16;
end
if (~isfield(par, 'partialProj'))
    par.partialProj = 0;
end
numblk = size(blk, 1);
Xmat = smatADMM(blk, X);
Xp = cell(numblk, 1);
rrank = par.rrank;
if (par.partialProj == 0)
    P1 = cell(numblk, 1);
    P2 = cell(numblk, 1);
    P1t = cell(numblk, 1);
    P2t = cell(numblk, 1);
    eigX = cell(numblk, 1);
    nu = cell(numblk, 1);
    for k = 1:numblk
        kblk = blk(k, :);
        Xmatk = Xmat{k};
        if (strcmp(kblk{1}, 's'))
            [Xp{k}, P1{k}, P2{k}, eigX{k}, nu{k}, rrank(k)] = projsdpsub(Xmatk, par.smtol);
            P1t{k} = P1{k}';
            P2t{k} = P2{k}';
        elseif (strcmp(kblk{1}, 'q'))
            Xp{k} = projsocpsub(Xmatk);
            eigX{k} = []; % % todo
            P1{k} = [];
            P2{k} = [];
            P1t{k} = [];
            P2t{k} = [];
            nu{k} = [];
        else
            n = size(Xmatk, 1);
            idxpos = find(Xmatk > par.smtol);
            Xp{k} = zeros(n, 1);
            eigX{k} = zeros(n, 1);
            Xp{k}(idxpos) = Xmatk(idxpos);
            eigX{k}(idxpos) = ones(length(idxpos), 1);
            P1{k} = [];
            P2{k} = [];
            P1t{k} = [];
            P2t{k} = [];
            nu{k} = [];
        end
    end
    par.P1 = P1;
    par.P2 = P2;
    par.P1t = P1t;
    par.P2t = P2t;
    par.eigX = eigX;
    par.nu = nu;
elseif (par.partialProj == 1)
    for k = 1:numblk
        kblk = blk(k, :);
        Xmatk = Xmat{k};
        if (strcmp(kblk{1}, 's'))
            [Xp{k}, rrank(k)] = projsdpsubPartial(Xmatk, rrank(k), par);
        elseif (strcmp(kblk{1}, 'q'))
            Xp{k} = projsocpsub(Xmatk);
        else
            Xp{k} = max(Xmatk, par.smtol);
        end
    end
end
par.rrank = rrank;
end
%%=====================================================
%% sub-routines
%%=====================================================

%%
%% projection onto psd cone
%% 
function [Xp, P1, P2, eigX, nu, rrank] = projsdpsub(Xmat, smtol)
n = size(Xmat, 1);
try 
    [P, D, ~] = mexeig(Xmat);
catch
    [P, D] = eig(Xmat);
end
d = full(diag(D));
[eigX, idx] = sort(d, 'descend');
P = P(:, idx);
idxpos = find(eigX > smtol);
idxngt = setdiff(1:n, idxpos);
rrank = length(idxpos);
P1 = P(:, idxpos);
P2 = P(:, idxngt);
d1 = eigX(idxpos);
d2 = eigX(idxngt);
Xp = P1 * (d1 .* P1');
blk{1,1} = 's';
blk{1,2} = n;
Xp = svecADMM(blk, Xp);
nu = (d1 - zeros(1, length(d2))) ./ (d1 - d2');
end

%%
%% projection onto second-order cone
%% 
function Xp = projsocpsub(Xmat)
Xmatk0 = Xmat(1);
Xmatkt = Xmat(2:end);
normXmatkt = norm(Xmatkt);
if (normXmatkt <= Xmatk0)
    Xp = Xmat;
elseif (normXmatkt <= -Xmatk0)
    Xp = zeros(length(Xmat), 1);
else
    Xp = (0.5 * (Xmatk0 + normXmatkt)) * [1; Xmatkt / normXmatkt];
end
end

%%
%% partial projection onto sdp cone
%%
function [Xp, rrank] = projsdpsubPartial(Xmat, rrank, par)
smtol = par.smtol;
if (isfield(par, 'iter')); iter = par.iter; else; iter = 0; end
n = size(Xmat, 1);
userrank = 0;
if (iter > 200 || par.maxfeas < 1e-3) 
    ranktmp = 3;
    if (userrank == 1); ranktmp = rrank; end
    if ~isfield(par, 'useLowRankEigs'); par.useLowRankEigs = 0; end
    if (par.useLowRankEigs == 0)
        [P, D] = eigs(Xmat, ranktmp+1, 'largestreal');
    else
        [Xp, ~, dX, ~] = lmLowRankProjSDP(Xmat, ranktmp, [], []);
        rrank = length(dX);
        blk{1,1} = 's';
        blk{1,2} = n;
        Xp = svecADMM(blk, Xp);
        return;
    end
else
    try 
        [P, D, ~] = mexeig(Xmat);
    catch
        [P, D] = eig(Xmat);
    end
end
d = full(diag(D));
[d, idx] = sort(d, 'descend');
P = P(:, idx);
idxpos = find(d > smtol);
d1 = d(idxpos);
P1 = P(:, idxpos);
Xp = P1 * (d1 .* P1');
rrank = length(idxpos);
blk{1,1} = 's';
blk{1,2} = n;
Xp = svecADMM(blk, Xp);
end

