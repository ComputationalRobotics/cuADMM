function solve_with_scs(problem)
    addpath("./utils");
    addpath("./mexfiles");
    addpath(genpath("~/antoine/scs-matlab/"));

    [sdpt3_blk, sdpt3_At, sdpt3_C, sdpt3_b, ~] = read_sedumi(problem.A, problem.b, problem.c, problem.K, 0);
    sdpt3.At = sdpt3_At;
    sdpt3.C = sdpt3_C;
    sdpt3.b = sdpt3_b;
    sdpt3.blk = sdpt3_blk;
    [At, b, C, blk] = data_sdpt3_to_admmSDPcuda(sdpt3);

    % SCS solves a slightly different problem, hence we
    % need to make some adjustements

    settings = [];
    data.A = At;
    data.b = full(-C);
    data.c = full(b);
    data.P = sparse([], [], [], size(data.c, 1), size(data.c, 1));
    cones.s = blk;

    [x, y, s, info] = scs(data, cones, settings);
end