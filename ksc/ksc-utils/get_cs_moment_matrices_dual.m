function info = get_cs_moment_matrices_dual(Sopt, blk, cliques)
    % Sopt: dual variables from correlative sparse SOSProgram
    % blk: variable structure
    % cliques: CS pattern
    % info: {[U1, D1], ... [Up, Dp]}, p is the clique number
    % where Di is the i's clique's eigenvalues, Ui is its eigenvectors
    N_cliques = length(cliques);
    N_clusters = size(blk, 1);
    info = [];
    if_moment_matrix = true;
    clique_count = 1;
    for i = 1: N_clusters
        if blk{i, 1} == 's'
            if if_moment_matrix
                Xmom = Sopt{i};
                [U, D] = sorteig(Xmom);
                D = diag(D);
                info = [info; {U, D}];
                clique_count = clique_count + 1;
                if_moment_matrix = false;
            end
        else
            if_moment_matrix = true;
        end
    end
    assert(N_cliques + 1 == clique_count); 
end