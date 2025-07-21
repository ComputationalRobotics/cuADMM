function sedumi_to_txt(problem, output_dir)
    addpath("~/matlab-install/mosek/10.1/toolbox/r2017a")
    addpath("./mexfiles")
    addpath("./utils")

    if isfield(problem, 'At')
        problem.A = problem.At';
    end

    % SeDuMi -> SDPT3
    [sdpt3_blk, sdpt3_At, sdpt3_C, sdpt3_b, ~] = read_sedumi(problem.A, problem.b, problem.c, problem.K, 0);
    sdpt3.At = sdpt3_At;
    sdpt3.C = sdpt3_C;
    sdpt3.b = sdpt3_b;
    sdpt3.blk = sdpt3_blk;
    % SDP.At = sdpt3_At;
    % SDP.C = sdpt3_C;
    % SDP.b = sdpt3_b;
    % SDP.blk = sdpt3_blk;

    % save(fullfile("/home/jordan/antoine/admmSDP/examples", 'PushBox_N=10_SOS.mat'), 'SDP');
    % return;

    % SDPT3 -> cuADMM
    [cuda_At, cuda_b, cuda_C, cuda_blk] = data_sdpt3_to_admmSDPcuda(sdpt3);
    store_sparse_mat(cuda_C, fullfile(output_dir, 'C.txt'));
    store_sparse_mat(cuda_b, fullfile(output_dir, 'b.txt'));
    store_sparse_mat(cuda_At, fullfile(output_dir, 'At.txt'));
    store_blk(cuda_blk, fullfile(output_dir, 'blk.txt'));
    store_sparse_vec(size(cuda_At, 2), fullfile(output_dir, 'con_num.txt'));

    % SeDuMi -> MOSEK
    prob = convert_sedumi2mosek(problem.A, problem.b, problem.c, problem.K);
    
    % solve with MOSEK
    [~, ~] = mosekopt('minimize info', prob);

    % solve with ADMM+
    run_admmplus(sdpt3_blk, sdpt3_At, sdpt3_C, sdpt3_b);
end

function v = from_cell_to_array(c)
    v = [];
    for i = 1: length(c)
        v = [v; c{i}];
    end
end

function [cuda_At, cuda_b, cuda_C, cuda_blk] = data_sdpt3_to_admmSDPcuda(sdpt3)
    cuda_At = from_cell_to_array(sdpt3.At);
    cuda_C = from_cell_to_array(svecADMM(sdpt3.blk, sdpt3.C));
    cuda_b = sdpt3.b;
    cuda_blk = sdpt3.blk;
    % cuda_blk = zeros(size(sdpt3.blk, 1), 1);
    % for i = 1: size(sdpt3.blk, 1)
    %     cuda_blk(i) = sdpt3.blk{i, 2};
    % end
end

function store_sparse_vec(vec, output_name)
    fileID = fopen(output_name, 'w');
    for i = 1: length(vec)
        fprintf(fileID, "%d\n", fix(vec(i)));
    end
    fclose(fileID);
end

function store_blk(blk, output_name)
    fileID = fopen(output_name, 'w');
    for i = 1: length(blk)
        if blk{i, 1} == 's' || blk{i, 1} == 'u'
            fprintf(fileID, "%c %d\n", blk{i, 1}, fix(blk{i, 2}));
        else
            fprintf("ERROR: unsupported block type %s\n", blk{i, 1});
            fclose(fileID);
            return;
        end
    end
    fclose(fileID);
end

function store_sparse_mat(mat, output_name)
    [rows, cols, vals] = find(mat);
    % we use 0-base in c++ and cuda
    rows = rows - 1;
    cols = cols - 1;
    % remember sort rows!
    [rows, indices] = sort(rows);
    cols = cols(indices);
    vals = vals(indices);
    fileID = fopen(output_name, 'w');
    for i = 1: length(rows)
        fprintf(fileID, "%d %d %.16f\n", rows(i), cols(i), vals(i));
    end
    fclose(fileID);
end