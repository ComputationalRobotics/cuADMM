function X_smat = smat_single(X_svec)
    vec_len = size(X_svec, 1);
    mat_size = (-1 + sqrt(1 + 8 * vec_len)) / 2;
    mat_size = round(mat_size);
    blk = {['s'], [mat_size]};
    X_smat = smat(blk, {X_svec});
    X_smat = X_smat{1};
end