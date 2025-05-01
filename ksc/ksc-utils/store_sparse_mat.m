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