% cone projection for Newton-CG
function [Xproj, output_info] = moment_cone_proj_newton_cg(Zp, At, input_info)
    % input and output are both svec, At' * At = I
    A = At';
    input_info.A = A;
    input_info.At = At;
    input_info.AAt = A * At;
    input_info.AAt_diag = diag(input_info.AAt);
    input_info.AAt_diag = full(input_info.AAt_diag);
    input_info.Zp = Zp;
    input_info.x0 = input_info.y_init;
    input_info.d0 = zeros(size(At, 2), 1);

    function val = proj_val_func(y, input_info)
        tmp = proj_single(input_info.Zp + input_info.At * y);
        val = 0.5 * norm(tmp)^2;
    end
    
    function input_info = proj_upadte_func(y, input_info)
        % update val_0, grad_0, Q, Omega, eps, M_inv, relgap
        Xb = input_info.Zp + input_info.At * y;
        Xb_mat = smat_single(Xb);
        [Q, eigvals] = sorteig(Xb_mat);
        eigvals = diag(eigvals);

        mat_size = length(eigvals);
        idx = 1: mat_size;
        gamma = idx(eigvals < 0);
        gamma_bar = idx(~(eigvals < 0));
        E = ones(length(gamma_bar));
        lam_gamma = eigvals(gamma);
        lam_gamma = lam_gamma(:);
        lam_gamma_bar = eigvals(gamma_bar);
        lam_gamma_bar = lam_gamma_bar(:);
        tmp = lam_gamma_bar - lam_gamma';
        tmp = 1 ./ tmp;
        nu_col_num = length(gamma);
        lam_gamma_bar = diag(lam_gamma_bar);
        nu = lam_gamma_bar * tmp;
        Omega = [E, nu; nu', zeros(nu_col_num)];

        input_info.pre_store.Omega = Omega;
        input_info.pre_store.Q = Q;

        eigvals = diag(eigvals);
        Xproj = Q * max(0, eigvals) * Q';
        Xproj = svec_single(Xproj);
        val_0 = 0.5 * norm(Xproj)^2;
        grad_0 = input_info.A * Xproj;
        input_info.pre_store.val_0 = val_0;
        input_info.pre_store.grad_0 = grad_0;
        
        tau1 = input_info.tau1;
        tau2 = input_info.tau2;
        input_info.pre_store.eps = tau1 * min([tau2, norm(grad_0)]);
        M_inv_diag = 1 ./ (input_info.AAt_diag + input_info.pre_store.eps);
        input_info.pre_store.M_inv = spdiags(M_inv_diag, 0, size(input_info.At, 2), size(input_info.At, 2));

        Zp = input_info.Zp;
        X = Xproj;
        pobj = 0.5 * norm(X - Zp)^2;
        dobj = 0.5 * norm(Zp)^2 - 0.5 * norm(X)^2;
        input_info.pre_store.relgap = abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj));

        % if we choose to use direct method:
        if input_info.pre_store.if_direct
            Omega = reshape(Omega, [], 1);
            Omega = spdiags(Omega, 0, length(Omega), length(Omega));
            Qt = Q';
            kron_Q = kron(Q, Q);
            kron_Qt = kron(Qt, Qt);
            At_sedumi = input_info.pre_store.At_sedumi;
            A_sedumi = input_info.pre_store.A_sedumi;
            tmp = kron_Q * Omega * kron_Qt;
            input_info.pre_store.M = A_sedumi * tmp * At_sedumi;

            % if norm(input_info.A * X) < 1e-10
            %     tmp = 0.5 * (input_info.pre_store.M + input_info.pre_store.M');
            %     eigvals = eig(tmp);
            %     bar(eigvals);
            %     data.M_eigvals = eigvals;
            %     save("./data/" + input_info.pre_store.prefix + "data.mat", "data");
            %     disp("AAA");
            % end
        end
    end

    function Vd = proj_matvec_mul_func(d, input_info)
        Q = input_info.pre_store.Q;
        Omega = input_info.pre_store.Omega;
        epsilon = input_info.pre_store.eps;
        Atd = input_info.At * d;
        Atd_mat = smat_single(Atd);
        tmp1 = Q' * Atd_mat * Q;
        tmp2 = Omega .* tmp1;
        tmp3 = Q * tmp2 * Q';
        tmp3 = svec_single(tmp3);
        Vd = input_info.A * tmp3;
        Vd = Vd + epsilon * d;
    end

    function y = proj_precond_solve_func(r, input_info)
        y = input_info.pre_store.M_inv * r;
    end

    val_func = @(y, input_info) proj_val_func(y, input_info);
    update_func = @(y, input_info) proj_upadte_func(y, input_info);
    matvec_mul_func = @(d, input_info) proj_matvec_mul_func(d, input_info);
    precond_solve_func = @(r, input_info) proj_precond_solve_func(r, input_info);

    [yopt, output_info] = newton_cg(val_func, ...
                                    matvec_mul_func, precond_solve_func,...
                                    update_func, input_info);
    Xproj = proj_single(Zp + At * yopt);
    output_info.y_final = yopt;
end