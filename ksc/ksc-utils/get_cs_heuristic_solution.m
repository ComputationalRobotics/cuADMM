function vopt = get_cs_heuristic_solution(info, cliques, v)
    % Sopt: dual variables from correlative sparse SOSProgram
    % blk: variable structure
    % cliques: CS pattern
    % v: msspoly variables
    % vopt: approximated optimal value
    vopt = zeros(length(v), 1);
    vopt_count_occurrence = zeros(length(v), 1);
    p = length(cliques);
    for i = 1: p
        U = info{i, 1}; % eigenvectors
        D = info{i, 2};
        lam = D(1);
        monomials = U(:, 1) * sqrt(lam); % eigenvector corresponding to the largest eigenvalue
        clique_ids = cliques{i};
        clique_size = length(clique_ids);
        vopt_clique = monomials(2: clique_size + 1);
        if monomials(1) < 0
            vopt_clique = -vopt_clique;
        end
        vopt_clique = vopt_clique / abs(monomials(1)); 
        for j = 1: clique_size
            id = clique_ids(j);
            vopt_count_occurrence(id) = vopt_count_occurrence(id) + 1;
            vopt(id) = vopt(id) + vopt_clique(j);
        end

        % fprintf("constant moment: \n");
        % disp(monomials(1));
        % fprintf("clique ids: \n");
        % disp(clique_ids);
        % fprintf("vopt in the clique: \n");
        % disp(vopt_clique);
    end
    
    vopt = vopt ./ vopt_count_occurrence;
    % fprintf("vopt: \n");
    % disp(vopt);
end