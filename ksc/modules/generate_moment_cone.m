function [At, B, others] = generate_moment_cone(n)
    x = msspoly('x', n);
    problem.vars = {[x]};
    problem.objective = msspoly(1);
    problem.inequality = {[msspoly()]};
    problem.equality = {[msspoly()]};
    problem.relaxation_order = 2;
    problem.rip_predecessor = [0];
    problem.regularization.expression = msspoly(1);
    problem.regularization.value = 1;
    [SDP, info] = sparse_sdp_relax(problem);
    
    % randomly generate a moment matrix
    At = SDP.sdpt3.At{1};
    At = At(:, info.At_clique.moment);
    [At, ~] = qr(At, 'econ');
    B = create_Bmom(n, x);

    % others
    fake_At = {At};
    fake_blk = SDP.sdpt3.blk;
    fake_C = SDP.sdpt3.C;
    fake_b = SDP.sdpt3.b(1:size(At, 2));
    [At_sedumi, ~, ~, ~] = SDPT3data_SEDUMIdata(fake_blk, fake_At, fake_C, fake_b);
    others.At_sedumi = At_sedumi;
end

function B = create_Bmom(n, x)
    kappa = 2;
    [mom_mat_symbolic, ~] = moment_variable(x, kappa);
    mom_mat_size = size(mom_mat_symbolic, 1);
    mom_svec_size = 1/2 * mom_mat_size * (mom_mat_size + 1);
    num = nchoosek(n + 2 * kappa, n);

    rows = [];
    cols = [];
    vals = [];
    for i = 1:  mom_mat_size
        for j = 1: i
            monomial = mom_mat_symbolic(i, j);
            col = moment_index(x, monomial);
            row = 0.5 * i * (i-1) + j;
            if i == j
                val = 1;
            else
                val = sqrt(2);
            end
            rows = [rows, row];
            cols = [cols, col];
            vals = [vals, val];
        end
    end

    B = sparse(double(rows), double(cols), double(vals), double(mom_svec_size), double(num));
end
