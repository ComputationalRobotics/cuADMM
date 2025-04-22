function sedumi_to_txt(problem, output_dir)
    store_sparse_mat(problem.c, fullfile(output_dir, 'C.txt'));
    store_sparse_mat(problem.b, fullfile(output_dir, 'b.txt'));
    store_sparse_mat(problem.A', fullfile(output_dir, 'At.txt'));
    store_sparse_vec(problem.K.s, fullfile(output_dir, 'blk.txt'));
    store_sparse_vec(size(problem.A, 2), fullfile(output_dir, 'con_num.txt'));
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