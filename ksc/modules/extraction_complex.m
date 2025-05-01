function [xi_recover, w_recover] = extraction_complex(mom_mat_numerical, input_info)
    x = input_info.x;
    n = input_info.n;
    kappa = input_info.kappa;
    mom_mat_symbolic = input_info.mom_mat_symbolic;
    
    % generate TMS
    if ~isfield(input_info, "tms")
        num = nchoosek(n + 2 * kappa, n);
        tms = zeros(num, 1);
        tms_filled = false(num, 1);
        mom_mat_size = size(mom_mat_numerical, 1);
        for i = 1: mom_mat_size
            for j = i: mom_mat_size
                monomial = mom_mat_symbolic(i, j);
                id = moment_index(x, monomial);
                if ~tms_filled(id)
                    tms_filled(id) = true;
                    tms(id) = mom_mat_numerical(i, j);
                end
            end
        end
    else
        tms = input_info.tms;
    end

    % extract solutions: Sr, Ur, Vr
    num_sub = nchoosek(n + kappa - 1, n);
    mom_sub_symbolic = mom_mat_symbolic(1: num_sub, 1: num_sub);
    mom_sub_numerical = mom_mat_numerical(1: num_sub, 1: num_sub);
    [U, S, V] = svd(mom_sub_numerical);
    s = diag(S);
    if_truncate = false;
    for k = 1: length(s) - 1
        if s(k+1) / s(1) < input_info.eps
            if_truncate = true;
            break;
        end
    end
    if if_truncate
        Sr = S(1:k, 1:k);
        Ur = U(:, 1:k);
        Vr = V(:, 1:k);
    else
        Sr = S;
        Ur = U;
        Vr = V;
    end
    Sr_inv = inv(Sr);
    
    % extract solutions: H and M
    Hs = cell(n, 1);
    Ms = cell(n, 1);
    for k = 1: n
        H_symbolic = x(k) * mom_sub_symbolic;
        H_numerical = zeros(size(H_symbolic));
        for i = 1: num_sub
            for j = 1: num_sub
                monomial = H_symbolic(i, j);
                id = moment_index(x, monomial);
                H_numerical(i, j) = tms(id);
            end
        end
        M = Sr_inv * Ur' * H_numerical * Vr;
        Hs{k} = H_numerical;
        Ms{k} = M;
    end
    
    % recover xi and w
    M_random = zeros(size(Ms{1}));
    for k = 1: n
        rand_num = 2 * rand() - 1;
        M_random = M_random + Ms{k} * rand_num;
    end
    [eig_vecs, ~] = eig(M_random); 
    xi_recover = zeros(n, size(Sr, 1));
    for j = 1: size(Sr, 1)
        for i = 1: n
            Mi = Ms{i};
            vj = eig_vecs(:, j);
            b = Mi * vj; 
            a = vj;
            % solve a * tmp = b
            xi_recover(i, j) = a' * b / (a' * a);
        end
    end
    one_vec = zeros(size(Hs{1}, 1), 1);
    one_vec(1) = 1;
    w_recover = zeros(1, size(Sr, 1));
    monomial_symbolic = mom_sub_symbolic(:, 1);
    for j = 1: size(Sr, 1)
        vj = eig_vecs(:, j);
        xi_j = xi_recover(:, j);
        tmp1 = one_vec' * mom_sub_numerical * Vr * vj;
        monomial = double(subs(monomial_symbolic, x, xi_j));
        tmp2 = monomial' * Vr * vj;
        w_recover(j) = tmp1 / tmp2;
    end
end

