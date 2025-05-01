function [V, D, output_info] = sorteig_LOPBCG(A, input_info)
    m = input_info.m;
    maxiter = input_info.maxiter;
    tol = input_info.tol;
    X_k = input_info.X_0;
    X_0 = input_info.X_0;
    Delta_X_k = input_info.Delta_X_0;
    if_qr_first_X = input_info.if_qr_first_X;
    if_new_Delta_X = input_info.if_new_Delta_X;
    verbose = input_info.verbose;
    if_strict_criteria = input_info.if_strict_criteria;
    pre_stop_miniter = input_info.pre_stop_miniter;
    pre_stop_gap = input_info.pre_stop_gap;

    assert(m == size(X_k, 2));
    assert(m == size(Delta_X_k, 2));
    
    % generate Lam_k and new X_k (new X_k is always orthogonal)
    if if_qr_first_X
        [Q, ~] = qr(X_k, 'econ');
    else
        Q = X_k;
    end
    T = Q' * A * Q;
    T = 0.5 * (T + T');
    [Y, Lam_k] = sorteig(T);
    X_k = Q * Y;

    if if_new_Delta_X
        Delta_X_k = X_k - X_0;
    else
        Delta_X_k = input_info.Delta_X_0;
    end

    for iter = 1: maxiter
        R_k = A * X_k - X_k * Lam_k;
        if verbose 
            fprintf("LOPBCG iter: %d, || R_k ||_F: %3.2e \n", iter, norm(R_k, 'fro'));
        end
        if if_strict_criteria
            if norm(R_k, 'fro') < tol
                if verbose
                    fprintf("LOPBCG convergent with strict criteria! \n");
                end
                break;
            end
        elseif iter > pre_stop_miniter
            % weak criteria: when some of the eigenvalues are very hard to
            % solve, detect them and pre-stop
            r_err_list = [];
            for k = 1: m
                r = R_k(:, k);
                r_err_list = [r_err_list; norm(r)];
            end
            strict_tol = pre_stop_gap * tol;
            stop_sign_list = zeros(size(r_err_list));
            for k = 1: m
                r = r_err_list(k);
                if r <= strict_tol 
                    stop_sign_list(k) = 1;
                elseif r >= tol 
                    stop_sign_list(k) = 2;
                end     
            end
            if nnz(stop_sign_list > 1) < m && nnz(stop_sign_list > 0) == m
                if verbose
                    fprintf("LOPBCG convergent with weak criteria! \n");
                end
                break;
            end
        end

        [Q, ~] = qr([X_k, R_k, Delta_X_k], 'econ');
        T = Q' * A * Q;
        T = 0.5 * (T + T');
        [Y, Lam_all] = sorteig(T);
        tmp = Q * Y;
        X_kp1 = tmp(:, 1:m);
        Lam_kp1 = Lam_all(1:m, 1:m);
        Delta_X_kp1 = X_kp1 - X_k;

        X_k = X_kp1;
        Lam_k = Lam_kp1;
        Delta_X_k = Delta_X_kp1;
    end

    V = X_k;
    D = Lam_k;
    output_info.Delta_X = Delta_X_k;
    output_info.iter = iter;
    output_info.R_err = norm(R_k, 'fro');
    r_err_list = [];
    for k = 1: m
        r = R_k(:, k);
        r_err_list = [r_err_list; norm(r)];
    end
    output_info.r_err_list = r_err_list;
end



