% cone projection for L-BFGS
function [Xproj, output_info] = moment_cone_proj_lbfgs(Zp, At, input_info)
    % input and output are both svec, At' * At = I
    A = At';
    input_info.A = A;
    input_info.At = At;
    input_info.Zp = Zp;

    function [val, grad, relgap] = proj_val_grad_func(y, input_info)
        A = input_info.A;
        At = input_info.At;
        Zp = input_info.Zp;
        X = proj_single(Zp + At * y);
        val = 0.5 * norm(X)^2;
        grad = A * X;
        pobj = 0.5 * norm(X - Zp)^2;
        dobj = 0.5 * norm(Zp)^2 - 0.5 * norm(X)^2;
        relgap = abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj));
    end
    
    val_grad_func = @(y, input_info) proj_val_grad_func(y, input_info);
    input_info.x0 = input_info.y_init;
    [yopt, output_info] = lbfgs(val_grad_func, input_info);
    Xproj = proj_single(Zp + At * yopt);
    output_info.y_final = yopt;
end