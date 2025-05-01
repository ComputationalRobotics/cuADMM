function pool = generate_pool(SDP, info_sdp, tolerance, env)
    con_num = size(SDP.sedumi.At, 2);
    pool = true(con_num, 1);

    fprintf("stage 1: \n");
    fprintf("clean between equality constraints in one clique \n");
    info_stage1.idx_remain = {};
    info_stage1.At_remain = {};
    for i = 1: env.N
        idx_origin = info_sdp.At_clique(i).equality;
        At_origin = SDP.sedumi.At(:, idx_origin);
        sigma_before = svds(At_origin, 1, "smallest");
        if i == 1
            idx_regularization = info_sdp.At_clique(i).regularization;
            At_regularization = SDP.sedumi.At(:, idx_regularization);
            idx_origin = [idx_regularization; idx_origin];
            At_origin = [At_regularization, At_origin];
        end
        [~, relidx_remain, relidx_remove] = licols(At_origin, tolerance);
        for j = 1: length(relidx_remove)
            pool(idx_origin(relidx_remove(j))) = false;
        end
        idx_remain = [];
        for j = 1: length(relidx_remain)
            idx_remain = [idx_remain; idx_origin(relidx_remain(j))];
        end
        At_remain = At_origin(:, relidx_remain);
        sigma_after = svds(At_remain, 1, "smallest");
        fprintf("stage 1, clique: %d, before: %3.2e, after: %3.2e \n", i, sigma_before, sigma_after);
        info_stage1.idx_remain{i} = idx_remain;
        info_stage1.At_remain{i} = At_remain;
    end

    fprintf("stage 2: \n");
    fprintf("clean equality and consensus constraints in two adjacent cliques \n");
    for i = 1: env.N - 1
        idx_consensus = info_sdp.At_clique(i+1).consensus;
        At_consensus = SDP.sedumi.At(:, idx_consensus);
        idx_origin = [idx_consensus; info_stage1.idx_remain{i}; info_stage1.idx_remain{i+1}];
        At_origin = [At_consensus, info_stage1.At_remain{i}, info_stage1.At_remain{i+1}];
        [~, relidx_remain, relidx_remove] = licols(At_origin, tolerance);
        sigma_before = svds(At_origin, 1, "smallest");
        sigma_after = svds(At_origin(:, relidx_remain), 1, "smallest");
        fprintf("stage 2, cliques: %d and %d, before: %3.2e, after: %3.2e \n", i, i+1, sigma_before, sigma_after);
        for j = 1: length(relidx_remove)
            pool(idx_origin(relidx_remove(j))) = false;
        end
    end

    sigma_final = svds(SDP.sedumi.At(:, pool), 1, "smallest");
    fprintf("final sigma: %3.2e \n", sigma_final);
end







