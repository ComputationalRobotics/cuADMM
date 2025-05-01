function m = mymonomial(vars, degree)
    p = mymonomial_core(uint64(length(vars)), uint64(degree));
    m = recomp(vars, p, speye(size(p, 1)));
end

