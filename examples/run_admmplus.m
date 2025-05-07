function run_admmplus(prob)
    addpath(genpath("~/matlab-install/SDPNAL+v1.0/"));
    SDPNAL_options.tol = 1e-4;
    SDPNAL_options.maxiter = 1.2e4;
    SDPNAL_options.printlevel = 1;
    problem = column2row_recursive(prob);
    problem.blx = [];
    [A, b, c, K] = convert_mosek2sedumi(problem);
    [sdpt3_blk, sdpt3_At, sdpt3_C, sdpt3_b, ~] = read_sedumi(A, b, c, K, 0);
    [~, ~, ~, ~, ~, ~, ~, ~, ~, SDPNAL_runhist] = admmplus(sdpt3_blk,...
        sdpt3_At,...
        sdpt3_C,...
        sdpt3_b,...
        [], [], [], [], [],...
        SDPNAL_options);
    disp(SDPNAL_runhist);
end