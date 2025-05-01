function [ SDP, info ] = sparse_sdp_relax_backup_cpp(problem)
	warning('off', 'MATLAB:structOnObject');
    assert(isfield(problem, "vars"), "problem.vars must not be empty");
    assert(isfield(problem, "objective"), "problem.objective must not be empty");
    assert(isfield(problem, "rip_predecessor"), "problem.rip_predecessor must not be empty");
    assert(isfield(problem, "relaxation_order"), "problem.relaxation_order must not be empty");
    assert(isscalar(problem.relaxation_order), "problem.relaxation_order must be a scalar");
    fprintf("Building sparse SDP relaxation of order %d.\n", problem.relaxation_order);
    problem.vars = toCell(problem.vars);
    p = length(problem.vars);
    problem.inequality = toCell(problem.inequality);
    num_psd = p + sum(cellfun(@length, problem.inequality));
    matrix = cell(num_psd, 1);
    svec = cell(num_psd, 1);
    for i = 1 : p
        [ matrix{i}, svec{i} ] = moment_variable(problem.vars{i}, problem.relaxation_order);
    end
    assert(length(problem.inequality) <= p, 'Number of inequalities (%d) is greater than number of cliques (%d)', length(problem.inequality), p);
    idx = p;
    for i = 1 : length(problem.inequality)
        for j = 1 : length(problem.inequality{i})
            idx = idx + 1;
            [ matrix{idx}, svec{idx} ] = moment_variable(problem.vars{i}, problem.relaxation_order - ceil(deg(problem.inequality{i}(j)) / 2));
            matrix{idx} = problem.inequality{i}(j) * matrix{idx};
            svec{idx} = problem.inequality{i}(j) * svec{idx};
        end
    end
    info.vmap = struct('matrix', matrix, 'svec', svec);
    problem.vars = msspoly_to_struct(problem.vars);
    if isfield(problem, "inequality"); problem.inequality = msspoly_to_struct(problem.inequality); else; problem.inequality = struct([]); end
    if isfield(problem, "equality"); problem.equality = msspoly_to_struct(problem.equality); else; problem.equality = struct([]); end
    check_msspoly = @(x) isstruct(x) && (isempty(x) || (isfield(x, "var") && isfield(x, 'sub') && isfield(x, 'coeff') && isfield(x, 'pow') && isfield(x, 'dim')));
    assert(check_msspoly(problem.vars) && check_msspoly(problem.inequality) && check_msspoly(problem.equality), "Unsupported msspoly format");
    if ~isfield(problem, 'regularization')
        problem.regularization = { struct("expression", 1, "value", 1) };
        warning('No regularization detected. Using default regularization.');
    elseif (~isa(problem.regularization, "cell"))
        problem.regularization = { problem.regularization };
    end
    flag_expression_value_mismatch = false;
    for i = 1 : length(problem.regularization)
        if (~isstruct(problem.regularization{i}))
            problem.regularization{i} = struct("expression", problem.regularization{i}, "value", []);
        elseif (~isfield(problem.regularization{i}, "value"))
            problem.regularization{i}.value = [];
        end
        problem.regularization{i}.value = double(problem.regularization{i}.value);
        problem.regularization{i}.expression = struct(msspoly(problem.regularization{i}.expression));
        if (length(problem.regularization{i}.value) < length(problem.regularization{i}.expression))
            flag_expression_value_mismatch = true;
            problem.regularization{i}.value = resize(problem.regularization{i}.value, length(problem.regularization{i}.expression), FillValue = 1);
        elseif (length(problem.regularization{i}.value) > length(problem.regularization{i}.expression))
            error('In clique %d, length of regularization values (%d) exceeds length of expressions (%d)', i, length(problem.regularization{i}.value), length(problem.regularization{i}.expression));
        end
    end
    if (flag_expression_value_mismatch)
        warning('Assuming unspecified value for regularizations be 1');
    end
    problem.regularization = cell2mat(problem.regularization);
    assert(length(problem.regularization) <= p, 'Number of regularizations (%d) is greater than number of cliques (%d)', length(problem.regularization), p);
    assert(length(problem.equality) <= p, 'Number of equalities (%d) is greater than number of cliques (%d)', length(problem.equality), p);
    assert(length(problem.objective) <= p, 'Number of objective (%d) is greater than number of cliques (%d)', length(problem.objective), p);
    assert(length(problem.rip_predecessor) <= p, 'Number of RIP predecessors (%d) mismatches number of cliques (%d)', length(problem.objective), p);
    problem.objective = struct(problem.objective(:));
	SDP = sparse_sdp_relax_cpp_core(...
		problem.vars,...
		problem.objective,...
		problem.inequality,...
		problem.equality,...
		uint64(problem.relaxation_order),...
		uint64(problem.rip_predecessor(:)),...
        problem.regularization);
end

function F_struct = msspoly_to_struct(F_msspoly)
    if ~iscell(F_msspoly); F_msspoly = { F_msspoly }; end
    if all(size(F_msspoly) ~= 1, 'all')
        warning('Detected instance of shape (%d, %d). It is suggested to reshape it into a vector.', size(F_msspoly, 1), size(F_msspoly, 2));
    end
    len = length(F_msspoly);
    for i = 1 : len
        assert(isequal(class(F_msspoly{i}), "msspoly"), "Unexpected element of class `%s'. Expected a msspoly instance.", class(F_msspoly{i}));
        F_msspoly{i} = struct(F_msspoly{i});
    end
    F_struct = cell2mat(F_msspoly);
    F_struct = F_struct(:);
end

function C = toCell(C)
    if ~iscell(C)
        assert(isequal(class(C), 'msspoly'), "Unexpected element of class `%s'. Expected a msspoly instance or cell array.", class(C));
        C = { C };
    end
    len = length(C);
    for i = 1 : len
        C{i} = C{i}(:);
    end
end