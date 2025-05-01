function [xopt, output_info] = newton_cg(val_func, ...
                                        matvec_mul_func, precond_solve_func,...
                                        update_func, input_info)
    x = input_info.x0;
    d = input_info.d0;
    eta_bar = input_info.eta_bar;
    maxiter = input_info.maxiter;
    tau = input_info.tau;
    tol = input_info.tol;

    alpha_list = [];
    line_search_step_list = [];
    gradnorm_list = [];
    eps_list = [];
    eta_list = [];
    relgap_list = [];
    cg_step_num_list = [];
    cg_residual_list = [];
    for j = 1: maxiter
        % update val_0, grad_0, Q, Omega, eps, M_inv, relgap
        input_info = update_func(x, input_info);

        % figure(1);
        % plot(abs(input_info.pre_store.grad_0));
        % figure(2);
        % plot(log10(abs(input_info.pre_store.grad_0) + 1e-16));

        if norm(input_info.pre_store.grad_0) < tol
            break;
        end
        
        input_info.cg.tol = min([eta_bar, norm(input_info.pre_store.grad_0)^(1+tau)]);
        if input_info.pre_store.if_direct
            tmp = input_info.pre_store.M;
            tmp = tmp + input_info.cg.tol * speye(size(tmp, 1));
            d = tmp \ (-input_info.pre_store.grad_0);
        else
            [d, cg_output_info] = precond_cg(d, -input_info.pre_store.grad_0, matvec_mul_func, precond_solve_func, input_info);
        end

        [alpha, line_search_step] = line_search(x, d, val_func, input_info);
        fprintf("outer iter: %d, line search step num: %d, alpha: %3.2e \n", j, line_search_step, alpha);
        x = x + alpha * d;

        alpha_list = [alpha_list; alpha];
        line_search_step_list = [line_search_step_list; line_search_step];
        gradnorm_list = [gradnorm_list; norm(input_info.pre_store.grad_0)];
        eps_list = [eps_list; input_info.pre_store.eps];
        eta_list = [eta_list; input_info.cg.tol];
        relgap_list = [relgap_list; input_info.pre_store.relgap];
        if ~input_info.pre_store.if_direct
            cg_step_num_list = [cg_step_num_list; cg_output_info.cg_step_num];
            cg_residual_list = [cg_residual_list; cg_output_info.residual_list(end)];
            fprintf("outer iter: %d, eps for CG: %3.2e, eta for CG: %3.2e, errRp: %3.2e, relgap: %3.2e \n",...
                j, eps_list(end), eta_list(end), gradnorm_list(end), relgap_list(end));
        else
            fprintf("outer iter: %d, eps: %3.2e, errRp: %3.2e, relgap: %3.2e \n",...
                j, eps_list(end), gradnorm_list(end), relgap_list(end));
        end
    end

    xopt = x;
    output_info.alpha_list = alpha_list;
    output_info.line_search_step_list = line_search_step_list;
    output_info.gradnorm_list = gradnorm_list;
    output_info.eps_list = eps_list;
    output_info.eta_list = eta_list;
    output_info.relgap_list = relgap_list;
    output_info.cg_step_num_list = cg_step_num_list;
    output_info.cg_residual_list = cg_residual_list;
end

function [xopt, output_info] = precond_cg(x, b, matvec_mul_func, precond_solve_func, input_info)
    residual_list = [];
    % r = A * x - b
    r = matvec_mul_func(x, input_info) - b;
    % y = M \ r
    y = precond_solve_func(r, input_info);
    p = -y;
    for k = 1: input_info.cg.maxiter
        r_old = r;
        y_old = y;
        residual_list = [residual_list; norm(r)];
        fprintf("CG iter: %d, residual: %3.2e \n", k, residual_list(end));
        if residual_list(end) < input_info.cg.tol
            break;
        end
        % alpha = r' * y / (p' * A * p)
        Ap = matvec_mul_func(p, input_info);
        alpha = (r' * y) / (p' * Ap);
        x = x + alpha * p;
        r = r + alpha * Ap;
        % y = M \ r
        y = precond_solve_func(r, input_info);
        % beta = r' * y / (r_old' * y_old)
        beta = (r' * y) / (r_old' * y_old);
        p = -y + beta * p;
    end
    
    xopt = x;
    output_info.residual_list = residual_list;
    output_info.cg_step_num = k;
end

function [alpha, step] = line_search(x, d, val_func, input_info)
    % based on d, s.t. (V + eps * I) * d = -grad, we need to find m, s.t.:
    % f(x + delta^m) < f(x) + mu * delta^m * (grad' * d)
    mu = input_info.line_search.mu;
    delta = input_info.line_search.delta;
    maxiter = input_info.line_search.maxiter;
    val_0 = input_info.pre_store.val_0;
    grad_0 = input_info.pre_store.grad_0;
    prod_0 = grad_0' * d;
    for step = 1: maxiter
        val = val_func(x + delta^step * d, input_info);
        if val <= val_0 + mu * delta^step * prod_0
            break;
        end
    end
    alpha = delta^step;
end