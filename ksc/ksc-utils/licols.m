function [idx_total, idx_remain, idx_remove] = licols(X, tol)
    if size(X, 2) == 1
        idx_total = 1;
        idx_remain = 1;
        idx_remove = [];
        return;
    end
    idx_total = 1: size(X, 2);
    idx_total = idx_total(:);
    [~, R, E] = qr(X, "econ", "vector");
    diagr = abs(diag(R)); 
    rmax = max(diagr);
    idx_remain = sort(E(diagr >= tol*rmax));
    idx_remain = idx_remain(:);
    idx_remove = setdiff(idx_total, idx_remain);
end