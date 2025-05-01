function msspoly_visualize(pop, x, var_str_mapping, fid)
    % pop: msspoly polynomial vector
    % x: msspoly variables: [var_1, ..., var_n]
    % var_str_mapping: [name_1, ..., name_n]
    % fid: file object
    N = length(pop);
    pop = [pop; x'*x];
    [~, degmat, coefmat] = decomp(pop);
    coefmat = coefmat'; % now the coefmat is of the size (N_term, N)
    coefmat = full(coefmat);
    degmat = full(degmat);
    pop_tex_list = [];
    for i = 1: N
        p_tex = "";
        [row, ~, val] = find(coefmat(:, i));
        for id = 1: length(row)
            term_id = row(id);
            term_tex = "";
            term_coef = val(id);
            [row_term, ~, val_term] = find(degmat(term_id, :)');
            if term_coef > 0
                if id > 1
                    term_tex = term_tex + "+";
                end
            else
                term_tex = term_tex + "-";
            end
            term_coef_abs = abs(term_coef);
            if (abs(term_coef_abs - 1) > 1e-8) || (isempty(row_term))
                term_tex = term_tex + string(term_coef_abs);
            end
            % \Pi var_id^deg_val
            for var_id = 1: length(row_term)
                deg_val = val_term(var_id); % an integer 
                var_name = var_str_mapping(row_term(var_id));
                if deg_val == 1
                    term_tex = term_tex + var_name;
                else
                    term_tex = term_tex + var_name + sprintf("^{%d}", deg_val);
                end
            end
            p_tex = p_tex + term_tex;
        end
        pop_tex_list = [pop_tex_list; p_tex];
    end
    
    % write tex codes to a markdown file
    fprintf(fid, "\n$$\n");
    fprintf(fid, "\\begin{align}\n");
    for i = 1: length(pop_tex_list)
        fprintf(fid, "   & "); 
        fprintf(fid, pop_tex_list(i)); 
        if i < length(pop_tex_list)
            fprintf(fid, " \\\\ \n");
        else
            fprintf(fid, "\n");
        end
    end
    fprintf(fid, "\\end{align}\n");
    fprintf(fid, "$$\n");
end