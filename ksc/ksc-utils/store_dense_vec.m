function store_dense_vec(vec, output_name)
    fileID = fopen(output_name, 'w');
    for i = 1: length(vec)
        fprintf(fileID, "%.16f\n", vec(i));
    end
    fclose(fileID);
end