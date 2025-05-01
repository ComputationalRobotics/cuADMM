function L = cholADMM(blk, At, par)
epsy = par.epsy;
mu = par.mu;
m = size(At{1}, 2);
AAt = sparse(m, m);
%% compute AAt
for k = 1:size(blk, 1)
    Atk = At{k};
    AAt = AAt + Atk'*Atk;
end
M = epsy*speye(m, m) + mu*AAt;
if (nnz(M)/m^2 < 0.4) 
    L.matfct_options = 'spchol';
    if (~issparse(M))
        M = sparse(M);
    end
    [L.R, ~, L.perm] = chol(M, 'vector');
    L.Rt = L.R';
else
    L.matfct_options = 'chol';
    if (issparse(M))
        M = full(M);
    end
    [L.R, ~] = chol(M);
    L.perm = 1:m;
end
end