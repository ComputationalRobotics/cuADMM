function sdpa_to_txt(input_dir, output_dir)
    addpath("./mexfiles")
    addpath("./utils")
    [blk, At, C, b] = read_sdpa(input_dir);
    sdpt3.At = At;
    sdpt3.C = C;
    sdpt3.b = b;
    sdpt3.blk = blk;
    [cuda_At, cuda_b, cuda_C, cuda_blk] = data_sdpt3_to_admmSDPcuda(sdpt3);
    store_sparse_mat(cuda_C, fullfile(output_dir, 'C.txt'));
    store_sparse_mat(cuda_b, fullfile(output_dir, 'b.txt'));
    store_sparse_mat(cuda_At, fullfile(output_dir, 'At.txt'));
    store_sparse_vec(cuda_blk, fullfile(output_dir, 'blk.txt'));
    store_sparse_vec(size(cuda_At, 2), fullfile(output_dir, 'con_num.txt'));
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
    cuda_blk = zeros(size(sdpt3.blk, 1), 1);
    for i = 1: size(sdpt3.blk, 1)
        cuda_blk(i) = sdpt3.blk{i, 2};
    end
end

function store_sparse_vec(vec, output_name)
    fileID = fopen(output_name, 'w');
    for i = 1: length(vec)
        fprintf(fileID, "%d\n", fix(vec(i)));
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