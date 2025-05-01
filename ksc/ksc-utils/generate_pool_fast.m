function pool = generate_pool_fast(SDP, chol_eps, tol)
    At = from_cell_to_array(SDP.sdpt3.At);
    m = size(At, 2);

    % adapt chol_eps
    [R, ~, P] = chol(At' * At + chol_eps * speye(m));
    while size(R, 1) ~= size(R, 2)
        chol_eps = chol_eps * 10;
        [R, ~, P] = chol(At' * At + chol_eps * speye(m));
    end

    % get pool
    mag = 10^0.125;
    D = full(diag(R));
    D_max = max(D);

    indices = (D > (D_max * tol));
    indices = P * indices;
    indices = (indices > 0);
    At_new = At(:, indices);
    [R_new, ~, ~] = chol(At_new' * At_new);
    while svds(At_new, 1, 'smallest') < 1e-12
        tol = tol * mag;
        indices = (D > (D_max * tol));
        indices = P * indices;
        indices = (indices > 0);
        At_new = At(:, indices);
        [R_new, ~, ~] = chol(At_new' * At_new);
    end

    pool = indices;
end


function v = from_cell_to_array(c)
    v = [];
    for i = 1: length(c)
        v = [v; c{i}];
    end
end