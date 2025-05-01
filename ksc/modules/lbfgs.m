function [xopt, output_info] = lbfgs(val_grad_func, input_info)
    x = input_info.x0;
    lbfgs_maxiter = input_info.lbfgs_maxiter;
    lbfgs_memory = input_info.lbfgs_memory;
    tol = input_info.tol;
    S = zeros(size(x, 1), lbfgs_memory);
    Y = S;
    % use GD for step 0
    [val, grad] = val_grad_func(x, input_info);
    input_info.val_0 = val;
    input_info.grad_0 = grad;
    p = -grad;
    [alpha, line_search_step] = line_search(x, p, val_grad_func, input_info);
    x_old = x;
    grad_old = grad;
    x = x + alpha * p;
    [val, grad] = val_grad_func(x, input_info);
    input_info.val_0 = val;
    input_info.grad_0 = grad;
    s_tmp = x - x_old;
    y_tmp = grad - grad_old;

    alpha_list = [alpha];
    gradnorm_list = [norm(grad)];
    line_search_step_list = [line_search_step];
    for iter = 1: lbfgs_maxiter
        if alpha < 0 || isnan(gradnorm_list(end)) 
            if input_info.verbose == 1
                fprintf("L-BFGS terminated due to abnormal line search! \n");
            end
            break;
        end
        if gradnorm_list(end) < tol || norm(y_tmp) < 1e-14
            break;
        end
        if input_info.verbose == 1
            fprintf("iter: %d, line search step num: %d, alpha: %3.2e, || grad ||: %3.2e \n",...
                    iter, line_search_step_list(end), alpha_list(end), gradnorm_list(end));
        end

        % compute gamma
        gamma = (s_tmp' * y_tmp) / norm(y_tmp)^2;
        % compute p
        if (iter <= lbfgs_memory)
            S(:, iter) = s_tmp;
            Y(:, iter) = y_tmp;
            p = -compute_Hg(grad, S(:, 1:iter), Y(:, 1:iter), gamma);
        else
            S(:, 1: lbfgs_memory-1) = S(:, 2: lbfgs_memory);
            Y(:, 1: lbfgs_memory-1) = Y(:, 2: lbfgs_memory);
            S(:, lbfgs_memory) = s_tmp;
            Y(:, lbfgs_memory) = y_tmp;
            p = -compute_Hg(grad, S, Y, gamma);
        end
        [alpha, line_search_step] = line_search(x, p, val_grad_func, input_info);
        x_old = x;
        grad_old = grad;
        x = x + alpha * p;
        [val, grad] = val_grad_func(x, input_info);

        input_info.val_0 = val;
        input_info.grad_0 = grad;
        s_tmp = x - x_old;
        y_tmp = grad - grad_old;

        alpha_list = [alpha_list; alpha];
        gradnorm_list = [gradnorm_list; norm(grad)];
        line_search_step_list = [line_search_step_list; line_search_step];
    end

    xopt = x;
    output_info.alpha_list = alpha_list;
    output_info.gradnorm_list = gradnorm_list;
    output_info.line_search_step = line_search_step_list;
    % output_info.relgap_list = relgap_list;
end

function r = compute_Hg(grad, S, Y, gamma)
    % compute rho_i = 1 / <s_i, y_i>
    rho = zeros(size(S, 2), 1);
    for i = 1: size(S, 2)
        rho(i) = 1.0 / (S(:, i)' * Y(:, i));
    end
    q = zeros(size(S, 1), size(S, 2) + 1);
    alpha = zeros(size(S, 2), 1);
    beta = zeros(size(S, 2), 1);
    % initial
    q(:, size(S, 2)+1) = grad;
    % backward
    for i = size(S, 2): -1: 1
        alpha(i) = rho(i) * (S(:, i)' * q(:, i+1));
        q(:, i) = q(:, i+1) - alpha(i) * Y(:, i);
    end
    % gamma * q
    r = gamma * q(:, i);
    % forward
    for i = 1: size(S, 2)
        beta(i) = rho(i) * (Y(:, i)' * r);
        r = r + S(:, i) * (alpha(i) - beta(i));
    end
end

function [alpha, step] = line_search(x, p, val_grad_func, input_info)
    % based on p = -H_k grad(x_k), use strong Wolfe conditions to search for alpha:
    % 1. f(x_k + alpha_k * p_k) <= f(x_k) + c1 * alpha_k * grad(x_k)' * p_k
    % 2. | grad(x_k + alpha_k * p_k)' * p_k | <= c2 * | grad(x_k)' * p_k | 
    c1 = input_info.c1;
    c2 = input_info.c2;
    line_search_maxiter = input_info.line_search_maxiter;
    val_0 = input_info.val_0;
    grad_0 = input_info.grad_0;
    prod_0 = grad_0' * p; % prod = grad' * p
    if prod_0 > 0
        fprintf("grad_0' * p should be negative, but now its value is: %3.2e: \n", prod_0);
        alpha = -1;
        step = 0;
        return;
    end
    alpha_const = 0.5;
    for k = 1: line_search_maxiter
        if k == 1
            alpha = 1.0;
            LB = 0.0;
            UB = 1.0;
        else
            alpha = alpha_const * (LB + UB);
        end
        x_new = x + alpha * p;
        [val_new, grad_new] = val_grad_func(x_new, input_info);
        prod_new = grad_new' * p;
        if k == 1
            prod_LB = prod_0;
            prod_UB = prod_new;
            if sign(prod_LB) * sign(prod_UB) > 0
                step = k;
                return;
            end
        end
        cond1 = ( val_new < val_0 + c1 * alpha * prod_0 - 1e-8 / max(1, abs(val_0)) );
        cond2 = ( abs(prod_new) < c2 * abs(prod_0) );
        if cond1 && cond2
            step = k;
            return;
        end
        if sign(prod_new) * sign(prod_UB) < 0
            LB = alpha;
            prod_LB = prod_new;
        elseif sign(prod_new) * sign(prod_LB) < 0
            UB = alpha;
            prod_UB = prod_new;
        end
    end
    step = k;
end