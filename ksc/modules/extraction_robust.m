function [xi_recover, v_recover, output_info] = extraction_robust(mom_mat_numerical, input_info)
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
        output_info.tms = tms;
    else
        tms = input_info.tms;
        output_info.tms = tms;
    end

    % extract solutions: S, U
    num_sub = nchoosek(n + kappa - 1, n);
    mom_sub_symbolic = mom_mat_symbolic(1: num_sub, 1: num_sub);
    mom_sub_numerical = mom_mat_numerical(1: num_sub, 1: num_sub);
    [U, S] = sorteig(mom_sub_numerical);
    s = diag(S);
    if_truncate = false;
    for k = 1: length(s) - 1
        if s(k+1) / s(1) < input_info.eps
            if_truncate = true;
            break;
        end
    end
    if if_truncate
        S = S(1:k, 1:k);
        U = U(:, 1:k);
    end
    s = diag(S);
    S_sqrt = diag(sqrt(s));
    S_sqrt_inv = diag(1 ./ sqrt(s));
    
    % get localizing matrices K and YK
    Ks = cell(n, 1);
    YKs = cell(n, 1);
    for k = 1: n
        K_symbolic = x(k) * mom_sub_symbolic;
        K_numerical = zeros(size(K_symbolic));
        for i = 1: num_sub
            for j = 1: num_sub
                monomial = K_symbolic(i, j);
                id = moment_index(x, monomial);
                K_numerical(i, j) = tms(id);
            end
        end
        Ks{k} = K_numerical;
        YK = S_sqrt_inv * U' * K_numerical * U * S_sqrt_inv;
        YKs{k} = YK;
    end
    
    % extract solution
    YK_random = zeros(size(YKs{1}));
    for k = 1: n
        rand_num = 2 * rand() - 1;
        YK_random = YK_random + YKs{k} * rand_num;
    end
    YK_random = 0.5 * (YK_random + YK_random');
    [O, ~] = eig(YK_random);
    Ys = cell(size(S, 1), 1);
    for k = 1: n
        Y = O' * YKs{k} * O;
        Ys{k} = diag(Y);
    end
    xi_recover = zeros(n, size(S, 1));
    for k = 1: n
        xi_recover(k, :) = Ys{k};
    end
    tmp = O' * S_sqrt * U';
    v_recover = abs(tmp(:, 1)');
    output_info.w_recover = v_recover.^2;
end

