function X_svec = svec_single(X_smat)
    mat_size = size(X_smat, 1);
    blk = {['s'], [mat_size]};
    X_svec = svec(blk, {X_smat});
    X_svec = X_svec{1};
end