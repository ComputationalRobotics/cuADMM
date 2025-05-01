function eta = get_eta(upperbound, y, blk, At, C, b)
    % only for POP without inequality constraints!!!
    tmp = AtymapADMM(blk, At, y);
    tmp = smatADMM(blk, tmp);
    S_bound = opsADMM(C, '-', tmp);
    lowerbound = b' * y;
    % since in POP, each variable is confined between [-1, 1],
    % g, h, obj's coefficient is beween [-1, 1],
    % here trace upperbound estimation is very simple: |tr(X)| <= size(X) 
    for i = 1: length(S_bound)
        D = eig(S_bound{i});
        M = size(S_bound{i}, 1);
        lowerbound = lowerbound + min(D(1), 0) * M;
    end
    eta = (upperbound - lowerbound) / (1 + abs(upperbound) + abs(lowerbound));
    fprintf("upperbound: %3.2e, lowerbound: %3.2e, eta: %3.2e \n", upperbound, lowerbound, eta);
end