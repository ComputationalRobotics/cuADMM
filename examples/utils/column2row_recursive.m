function S = column2row_recursive(S)
    % Recursively convert every numeric column vector in S to a row vector.
    flds = fieldnames(S);
    for k = 1:numel(flds)
        v = S.(flds{k});
        if isstruct(v)                %–– nested struct or struct array ––
            % Apply the function to every element in the struct array:
            S.(flds{k}) = arrayfun(@column2row_recursive, v);
        elseif isnumeric(v) && isvector(v) && size(v,2) == 1
            S.(flds{k}) = v.';        %–– flip column → row ––
        end
        % (Anything else—cell arrays, strings, matrices—stays untouched.)
    end
end