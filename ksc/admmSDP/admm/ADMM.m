function [X, y, S, info] = ADMM(blk, At, b, C, options, X0, y0, S0)
tstart = clock;
maxIter = 1000;
stopTol = 1e-6;
printyes = 1;
scaleA = 1;
scaleData = 1;
sig = 1e0;   
partialProj = 1;
useLowRankEigs = 0;

if isfield(options, 'maxIter'); maxIter = options.maxIter; end
if isfield(options, 'stopTol'); stopTol = options.stopTol; end
if isfield(options, 'printyes'); printyes = options.printyes; end
if isfield(options, 'scaleA'); scaleA = options.scaleA; end
if isfield(options, 'scaleData'); scaleData = options.scaleData; end
if isfield(options, 'sig'); sig = options.sig; end
if isfield(options, 'partialProj'); partialProj = options.partialProj; end

%% initial points
m = length(b);
if (nargin < 6)
    X = opsADMM(C, '.*', 0);
    S = X;
    y = zeros(m, 1);
else 
    X = X0;
    S = S0;
    y = y0;
end
borg = b;
Corg = C;
normborg = 1 + norm(borg);
normCorg = 1 + opsADMM(Corg, 'norm');

%% scale A
if (scaleA)
    normA = zeros(m, 1);
    for k = 1:size(blk, 1)
        normA = normA + sum(At{k}.*At{k}, 1)';
    end
    normA = max(1.0, sqrt(normA));
    DA = spdiags(1./normA, 0, m, m);
    for k = 1:size(blk, 1)
        At{k} = At{k} * DA;
    end
    b = b ./ normA;
    y = normA .* y;
else
    normA = 1.0;
end

%% scale data
normb = 1 + norm(b);
normC = 1 + opsADMM(C, 'norm');
if (scaleData)
    bscale = normb;
    Cscale = normC;
else
    bscale = 1;
    Cscale = 1;
end
objscale = bscale * Cscale;
b = b / bscale;
C = opsADMM(C, './', Cscale);
X = opsADMM(X, './', bscale);
S = opsADMM(S, './', Cscale);
y = y / Cscale;

%% AAT solver
par.rrank = zeros(size(blk, 1), 1);
par.partialProj = partialProj;
par.useLowRankEigs = useLowRankEigs;
par.epsy = 1e-16;
par.mu = 1.0;
LAAT = cholADMM(blk, At, par);
AATsovler = @(rhs) linSysSolveADMM(LAAT, rhs);

%% matrices are stored in vectorized format by default
X = svecADMM(blk, X);
S = svecADMM(blk, S);
C = svecADMM(blk, C);

%% initial kkt residual
Aty = AtymapADMM(blk, At, y);
Rp = b - AXmapADMM(blk, At, X);
SmC = opsADMM(S, '-', C);
Rd = opsADMM(Aty, '+', SmC);
Rporg = (normA .* Rp) * bscale;
Rdorg = opsADMM(Rd, '.*', Cscale);
errRp = norm(Rporg) / normborg;
errRd = opsADMM(Rdorg, 'norm') / normCorg;
maxfeas = max(errRp, errRd);
pobj = blkTraceADMM(blk, C, X) * objscale;
dobj = sum(b .* y) * objscale;
relgap = abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj));
info.errRp(1) = errRp;
info.errRd(1) = errRd;
info.relgap(1) = relgap;
info.pobj(1) = pobj;
info.dobj(1) = dobj;

%% print header 
if (printyes)
    fprintf("\n ---------------------------------------------------------------");
    fprintf("---------------------------------------------------------------");
    fprintf("\n An ADMM for SDP");
    fprintf("\n normC = %2.1e, normb = %2.1e", normCorg, normborg);
    fprintf("\n ---------------------------------------------------------------");
    fprintf("---------------------------------------------------------------");
end

msg = [];
breakyes = 0;
prim_win = 0;
dual_win = 0;
rescale = 1;
sigfix = 0;
%% main loop
for iter = 1:maxIter+1
    %% check termination
    if (max([maxfeas, relgap]) < stopTol)
        breakyes = 1;
        msg = 'Convergent!';
    end
    if (iter > maxIter) 
        breakyes = 2;
        msg = 'Maximum iteration reached!';
    end

    %% print iterate
    if (breakyes > 0 ...
            || (printyes && (iter <= 200 && rem(iter, 50) == 1) ...
            || (iter > 200 && rem(iter, 100) == 1)))
        fprintf("\n %4d | %3.2e %3.2e| %- 5.4e %- 5.4e %3.2e| %5.1f| %2.1e| %d", ...
            iter-1, full(errRp), full(errRd), full(pobj), full(dobj), full(relgap), ...
            etime(clock, tstart), sig, computeMaxRank(blk, par.rrank));
    end

    if (breakyes > 0)
        if (printyes)
            fprintf("\n ---------------------------------------------------------------");
            fprintf("---------------------------------------------------------------");
            fprintf("\n %s\n", msg);
            fprintf("\n cputime   = %s", mytimeADMM(etime(clock, tstart)));
            fprintf("\n iter      = %d", iter - 1);
            fprintf("\n primal infeas = %2.1e \n dual   infeas = %2.1e \n relative gap  = %2.1e", ...
                errRp, errRd, relgap);
            fprintf("\n primal objective = %- 9.8e \n dual   objective = %- 9.8e", ...
                pobj, dobj);
            fprintf("\n ---------------------------------------------------------------");
            fprintf("---------------------------------------------------------------\n");
            break;
        end
    end 

    %% rescaling 
    if ((rescale == 1 && maxfeas < 5e2 && iter > 21 && relgap < 2e-1) ...
        || (rescale == 2 && maxfeas < 1e-2 && iter > 40 && relgap < 5e-2) ...
        || (rescale >= 3 && max(normX/normyS, normyS/normX) > 1.2 && rem(iter, 203) == 0))
        normy = norm(y);
        normAty = opsADMM(Aty, 'norm');
        normX = opsADMM(X, 'norm');
        normS = opsADMM(S, 'norm');
        normyS = max([normy, normAty, normS]);
        bscale2 = normX;
        Cscale2 = normyS;
        bscale = bscale * bscale2;
        Cscale = Cscale * Cscale2;
        objscale = objscale * bscale2 * Cscale2;
        b = b / bscale2;
        C = opsADMM(C, './', Cscale2);
        X = opsADMM(X, './', bscale2);
        Rp = Rp / bscale2;
        SmC = opsADMM(SmC, './', Cscale2);
        sig = sig * (Cscale2 / bscale2);
        if (printyes)
            fprintf('\n      [rescale = %d:%4d| %2.1e, %2.1e, %2.1e, %2.1e| %2.1e %2.1e| %2.1e]', ...
                rescale, iter-1, normX, normy, normAty, normS, bscale, Cscale, sig);
        end
        rescale = rescale + 1;
        prim_win = 0;
        dual_win = 0;
    end

    %% compute y
    ASmC = AXmapADMM(blk, At, SmC);
    rhsy = Rp / sig - ASmC;
    y = AATsovler(rhsy);
    Aty = AtymapADMM(blk, At, y);
    Rd1 = opsADMM(Aty, '-', C);
    %% compute X
    Xold = X;
    Xinput = opsADMM(X, '+', opsADMM(Rd1, '.*', sig));
    par.iter = iter;
    par.maxfeas = max(errRp, errRd);
    [X, par] = projsdpADMM(blk, Xinput, par);
    XdiffSig = opsADMM(opsADMM(X, '-', Xold), './', sig);
    %% compute S
    S = opsADMM(XdiffSig, '-', Rd1);
    SmC = opsADMM(S, '-', C);
    %% project x onto {X|A(X) = b}
    projyes = 0;
    if (projyes == 1 && rem(iter, 31) == 1)
        %fprintf('[p]');
        Rp = b - AXmapADMM(blk, At, X);
        Xtmp = AATsovler(Rp);
        ATXtmp = AtymapADMM(blk, At, Xtmp);
        X = opsADMM(X, '+', ATXtmp);
    end

    %% compute KKT residuals
    Rp = b - AXmapADMM(blk, At, X);
    Rd = opsADMM(Rd1, '+', S);
    Rporg = (normA .* Rp) * bscale;
    Rdorg = opsADMM(Rd, '.*', Cscale);
    errRp = norm(Rporg) / normborg;
    errRd = opsADMM(Rdorg, 'norm') / normCorg;
    maxfeas = max(errRp, errRd);
    pobj = blkTraceADMM(blk, C, X) * objscale;
    dobj = sum(b .* y) * objscale;
    relgap = abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj));
    info.errRp(1+iter) = errRp;
    info.errRd(1+iter) = errRd;
    info.relgap(1+iter) = relgap;
    info.pobj(1+iter) = pobj;
    info.dobj(1+iter) = dobj;

    %% update sig
    if (sigfix == 0)
        ratioconst = 1e0;
        feasratio = ratioconst*errRp / errRd;
        info.feasratio(iter) = feasratio;
        if (feasratio < 1)
            prim_win = prim_win + 1;
        else
            dual_win = dual_win + 1;
        end
        sigmax = 1e6;
        sigmin = 1e-4;
        sigscale = 1.15;
        if ((iter <= 200 && rem(iter, 20) == 1) || (iter > 200 && rem(iter, 50) == 1))
            if (prim_win > 1.35*dual_win)
                prim_win = 0;
                sig = min(sigmax, sig*sigscale);
            elseif (dual_win > 1.35*prim_win)
                dual_win = 0;
                sig = max(sigmin, sig/sigscale);
            end
        end
    end
end
%% recover the original solution
X = opsADMM(X, '.*', bscale);
X = smatADMM(blk, X);
y = (y ./ normA) * Cscale;
S = opsADMM(S, '.*', Cscale);
S = smatADMM(blk, S);
info.msg = msg;
info.iter = iter-1;
info.cputime = etime(clock, tstart);
end