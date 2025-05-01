function homo_pop = msspoly_homogenize(pop, x, z)
    % pop: msspoly polynomial vector of size (N, 1)
    % x: msspoly variable of size (n, 1)
    % z: additional homogenization variable of size (1, 1)
    % homo_pop: homogenized vector of size (N, 1)
    % where homo_pop(i) = z^deg(pop(i)) * pop(i)(x/z)
    N = size(pop, 1);
    pop = [pop; x'*x];
    [~, degmat, coefmat] = decomp(pop); % degmat of size (N_term, n), coefmat of size (N, N_term)
    
    homo_pop = [];
    for k = 1: N
        p = pop(k);
        deg_p_max = deg(p);
        p_coef = coefmat(k, :);
        [row, ~, val] = find(p_coef');
        tilde_p = 0;
        for i = 1: length(row)
            term_degmat = degmat(row(i), :);
            [row_deg, ~, val_deg] = find(term_degmat');
            deg_sum = 0;
            term = 1;
            for j = 1: length(row_deg)
                var_id = row_deg(j);
                term = term * x(var_id)^val_deg(j);
                deg_sum = deg_sum + val_deg(j);
            end
            if deg_sum < deg_p_max
                term = term * z^(deg_p_max - deg_sum); 
            end
            tilde_p = tilde_p + val(i) * term;
        end
        homo_pop = [homo_pop; tilde_p];
    end
end