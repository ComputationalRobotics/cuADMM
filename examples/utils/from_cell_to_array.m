function v = from_cell_to_array(c)
    v = [];
    for i = 1: length(c)
        v = [v; c{i}];
    end
end