function [cleaned_pop, scale_vec] = msspoly_clean(pop, v, tol, if_scale)
    % pop: msspoly polynomial vector of size (N, 1)
    % v: msspoly variable of size (n, 1)
    % tol: numerical tolerance
    % if_scale: scale each coefficient to [-1, 1] or not
    % scale_vec: contain the max absolute value of coefficients in each polynomial, of size (N, 1)
    N = size(pop, 1);
    pop = [pop; v'*v];
    [~, degmat, coefmat] = decomp(pop); % degmat of size (N_term, n), coefmat of size (N, N_term)

    cleaned_pop = [];
    scale_vec = [];
    for k = 1: N
        p = pop(k);
        p_coef = coefmat(k, :);
        [row, ~, val] = find(p_coef');
        max_abs_val = max(abs(val));                
        if if_scale
            val = val / max_abs_val;
            scale_vec = [scale_vec; max_abs_val];
        else
            scale_vec = [scale_vec; 1];
        end
        cleaned_p = 0;
        for i = 1: length(row)
            term_degmat = degmat(row(i), :);
            [row_deg, ~, val_deg] = find(term_degmat');
            term = 1;
            for j = 1: length(row_deg)
                var_id = row_deg(j);
                term = term * v(var_id)^val_deg(j);
            end
            if abs(val(i)) > tol
                cleaned_p = cleaned_p + val(i) * term;
            end
        end
        cleaned_pop = [cleaned_pop; cleaned_p];
    end
end