function [X_svec, eigvals, eigvecs] = proj_single(X_svec, input_info)
    X_smat = smat_single(X_svec);
    % [V, D] = eigs(X_smat, 6, 'largestreal');
    [V, D] = sorteig(X_smat);
    % bar(diag(D));

    if (nargin == 1)
        X_smat = V * max(D, 0) * V';
    else
        pos_tol = input_info.pos_tol; 
        pos_num = input_info.pos_num;
        d = diag(D);
        d(pos_num+1: end) = 0;
        d(d < 0) = 0;
        X_smat = V * diag(d) * V';
    end

    X_svec = svec_single(X_smat);
    if (nargout == 2)
        eigvals = diag(D);
    elseif (nargout == 3)
        eigvals = diag(D);
        eigvecs = V;
    end
end