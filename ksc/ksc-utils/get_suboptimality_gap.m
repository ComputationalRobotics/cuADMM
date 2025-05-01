function eta = get_suboptimality_gap(upperbound, y, blk, At, C, b, dual_info)
    % since in POP, each variable is confined between [-1, 1],
    % g's coefficient is beween [-1, 1],
    % here trace upperbound estimation is very simple: 
    % for moment matrix: |tr(X)| <= size(X) 
    % for localizing matrix : | tr(X) | <= size(X) * num_of_terms_in_g
    tmp = AtymapADMM(blk, At, y);
    tmp = smatADMM(blk, tmp);
    S_bound = opsADMM(C, '-', tmp);
    lowerbound = b' * y;
    
    idx = 0;
    LARGE = blk{1, 2};
    SMALL = blk{2, 2};
    for i = 1: length(S_bound)
        D = eig(S_bound{i});
        M = size(S_bound{i}, 1);
        if size(S_bound{i}) == SMALL
            idx = idx + 1;
            w = dual_info.g(idx);
            [~, ~, units] = decomp(w);
            num_of_terms_in_g = nnz(units);
        else
            num_of_terms_in_g = 1;
        end
        lowerbound = lowerbound + num_of_terms_in_g * min(0, D(1)) * M;
    end
    assert(idx == length(dual_info.g));
    eta = (upperbound - lowerbound) / (1 + abs(upperbound) + abs(lowerbound));
    fprintf("upperbound: %3.2e, lowerbound: %3.2e, eta: %3.2e \n", upperbound, lowerbound, eta);
end