function SDP = sparse_sdp_relax_homo_matlab(problem)
	%% Implements sparse Lasserre's hierarchy
	%% Heng Yang & Xiaoyang Xu
	%% Sept. 19, 2021
	%% Oct. 13, 2021: fixed a bug in generating redundant equality constraints
	%% Oct. 11, 2023: modified to sparse

	%% problem.vars:		cell array of length p, each entry contains a clique of msspoly variables
	%% e.g. { [ x(1); x(4) ], [ x(1); x(2); x(3); x(5) ], [ x(1); x(3); x(5); x(6) ] }
	%% problem.objective:	msspoly polynomial
	%% e.g. x(2) * x(5) + x(3) * x(6) - x(2) * x(3) - x(5) * x(6) + x(1) * (-x(1) + x(2) + x(3) - x(4) + x(5) + x(6))
	%% problem.inequality:	cell array of length p, each entry contains
	%%						all polynomials that should be non-negative in the current clique
	%% e.g. { [ (6.36 - x(1)) * (x(1) - 4); (6.36 - x(4)) * (x(4) - 4) ], [ ...
	%% problem.equality:	cell array of length p, each entry contains
	%%						all polynomials that should be 0 in the current clique
	%% problem.relaxation_order:	vector that indicate the relaxation order in each clique
	%% e.g. [ 2; 2; 2 ]
	
	if ~isfield(problem,'vars') error('Please provide variables of the POP.'); end
	if ~isfield(problem,'objective'); error('Please provide objective function.'); end
	if ~isfield(problem,'equality'); problem.equality = []; end
	if ~isfield(problem,'inequality'); problem.inequality = []; end
	
	fprintf('\n======================= Sparse SDP Relaxation =======================\n')
	
	%% copy POP data and decide relaxation order
	I	   = problem.vars;
	f	   = problem.objective;
	H	   = problem.equality;
	G	   = problem.inequality;
	kappa	= problem.relaxation_order;
	assert(iscell(I), 'problem.vars should be a cell');
	assert(iscell(H), 'problem.equality should be a cell');
	assert(iscell(G), 'problem.inequality should be a cell');
	assert(length(I) == length(H) && length(H) == length(G),...
		'problem.vars, problem.equality, problem.inequality should have the same shape');
	assert(length(kappa) == 1, "relaxation order should be a scalar");
	p = length(I);
	
	%% pop is a vector that contains all expressions to decompose.
	%% In preprocessing step, we compute the expressions of moment matrices,
	%%	 localize inequility matrices, objective function and equality constraints
	%%	 to be decomposed and parsed in the next step
	%% The PSD matrices are added in the lower triangular way, similar to SDPT3 format
	pop = [];
	K.s = zeros(length(I), 1);
	fprintf('Preprocessing Moment Variables...');
	for i = 1 : p
		v = mymonomial(I{i}, kappa);
		K.s(i) = length(v);
		for j = 1 : length(v)
			pop = [ pop; v(j) * v(1 : j) ];
		end
	end
	fprintf('Done.\n');
	fprintf('Preprocessing Inequality Constraint Variables...');
	for i = 1 : p
		for j = 1 : length(G{i})
			g = G{i}(j);
			v = mymonomial(I{i}, kappa - ceil(deg(g) / 2));
			K.s(end + 1) = length(v);
			for j = 1 : length(v)
				pop = [ pop; g * v(j) * v(1 : j) ];
			end
		end
	end
	fprintf('Done.\n');
	fprintf('Preprocessing Objective & Equality Constraints...');
	pop = [ pop; sum(f(:)) ];
	num_equality = 0;
	for i = 1 : p
		for j = 1 : length(H{i})
			h = H{i}(j);
			new_poly = h * monomials(I{i}, 0 : 2 * kappa - deg(h));
			num_equality = num_equality + length(new_poly);
			pop = [ pop; new_poly ];
		end
	end
	fprintf('Done.\n');

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	z = problem.z;
	pop = [pop; z^deg(f)];
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	num_sdpt3_variables = sum(K.s .* (K.s + 1)) / 2;
	num_sedumi_variables = dot(K.s, K.s);
	[ ~, ~, coef ] = decomp(pop);
	coef_t = coef.';
	%% nterms:	number of distinct monomials
	nterms = size(coef, 2);
	%% vis(i):	whether monomial #i has appeared before
	vis = false(nterms, 1);
	%% map_*(i, :): the position of the first appearance of monomial #i (in moment matrices)
	%% e.g. if monomial #i appears first at (5, 1) of the 1st moment matrix (size: 6 * 6)
	%% then map_sedumi(i, 5) = map_sedumi(i, 25) = 0.5 and all other entries are 0
	%% and  map_sdpt3(i, 11) = sqrt(0.5)
	mapt_sedumi = spalloc(num_sedumi_variables, nterms, nterms * 2);
	mapt_sdpt3 = spalloc(num_sdpt3_variables, nterms, nterms * 2);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% A_sedumi = sparse(1, 1, 1, 1, num_sedumi_variables);
	% A_sdpt3 = sparse(1, 1, 1, 1, num_sdpt3_variables);
	p = length(I);
	coef_homo = coef(end, :); % for the last polynomial: z^d
	[~, col_homo, ~] = find(coef_homo, 1);
	n_first = K.s(1);
	zd_col = coef(:, col_homo);
	[row_first_zd, ~, ~] = find(zd_col, 1);
	count = 0;
	flag = false;
	for i = 1: n_first
		for j = 1: i
			count = count + 1;
			if count == row_first_zd
				flag = true;
				first_zd_id_sdpt3 = count;
				first_zd_id_sedumi = (i - 1) * n_first + j;
				first_zd_id_sedumi_ = (j - 1) * n_first + i;

				if i == j
					regularization_sedumi = sparse(first_zd_id_sedumi, 1, 1, num_sedumi_variables, 1);
					regularization_sdpt3 = sparse(first_zd_id_sdpt3, 1, 1, num_sdpt3_variables, 1);
				else
					regularization_sedumi = sparse(...
									[first_zd_id_sedumi, first_zd_id_sedumi_],...
                                    [1, 1],...
									[0.5, 0.5],...
									num_sedumi_variables, 1);
					regularization_sdpt3 = sparse(first_zd_id_sdpt3, 1, sqrt(0.5), num_sdpt3_variables, 1);
                end
				break;
            end
		end
		if flag == true
			break;
		end
	end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%% Each entry in a moment matrix must be a monomial
	%% If the monomial did not appear before, we log it down in map_*
	%% Otherwise, we use the information stored in map_* to give an equality constraint
	fprintf('Building Moment Constraints...');
	idx_sedumi = 1;
	idx_sdpt3 = 1;
    num_moment_constraints = K.s(1 : length(I));
    num_moment_constraints = dot(num_moment_constraints, num_moment_constraints + 1) / 2 - nterms;
	moment_sedumi = spalloc(num_sedumi_variables, num_moment_constraints, num_moment_constraints * 2);
	moment_sdpt3 = spalloc(num_sdpt3_variables, num_moment_constraints, num_moment_constraints * 2);
    idx_constraint = 1;
	for i_ = 1 : length(I)
		n = K.s(i_);
		%% Only consider the lower triangular part
		for i = 1 : n
			for j = 1 : i
				[ row, ~, v ] = find(coef_t(:, idx_sdpt3), 1);
				assert(v == 1);
				if i == j
					identity_sdpt3 = sparse(idx_sdpt3, 1, 1, num_sdpt3_variables, 1);
					identity_sedumi = sparse(idx_sedumi, 1, 1, num_sedumi_variables, 1);
				else
					idx_sedumi_ = idx_sedumi + (j - i) * (n - 1);
					identity_sdpt3 = sparse(idx_sdpt3, 1, sqrt(0.5), num_sdpt3_variables, 1);
					identity_sedumi = sparse([ idx_sedumi, idx_sedumi_ ], [ 1, 1 ], [ 0.5, 0.5 ], num_sedumi_variables, 1);
				end
				if vis(row)
					moment_sdpt3(:, idx_constraint) = mapt_sdpt3(:, row) - identity_sdpt3;
					moment_sedumi(:, idx_constraint) = mapt_sedumi(:, row) - identity_sedumi;
                    idx_constraint = idx_constraint + 1;
				else
					vis(row) = true;
					mapt_sedumi(:, row) = identity_sedumi;
					mapt_sdpt3(:, row) = identity_sdpt3;
				end
				idx_sedumi = idx_sedumi + 1;
				idx_sdpt3 = idx_sdpt3 + 1;
			end
			idx_sedumi = idx_sedumi + n - i;
		end
	end
	%% If some monomial appeared in the system has not appeared in the moment matrices
	%% Then there is some conflict between clique, objective function, constraints or relaxation order
	assert(all(vis), 'Problem has conflict');
	fprintf('Done.');

	%% Now every monomial in the system has some correspondence in some entry of some moment matrix
	%% We express each (in)equalityequality constraint and objective function
	%%	 as a linear combination of these entries to derive more equality constraints
	fprintf('Building Inequality Constraints...');
	assert(idx_sdpt3 == 1 + dot(K.s(1 : length(I)), K.s(1 : length(I)) + 1) / 2);
	ineq_sedumi = mapt_sedumi * coef_t(:, idx_sdpt3 : num_sdpt3_variables);
	ineq_sdpt3 = mapt_sdpt3 * coef_t(:, idx_sdpt3 : num_sdpt3_variables);
	idx_constraint = 1;
	for i_ = length(I) + 1 : length(K.s)
		n = K.s(i_);
		for i = 1 : n
			for j = 1 : i - 1
				idx_sedumi_ = idx_sedumi + (j - i) * (n - 1);
				ineq_sdpt3(idx_sdpt3, idx_constraint) = -sqrt(0.5);
				ineq_sedumi(idx_sedumi, idx_constraint) = -0.5;
				ineq_sedumi(idx_sedumi_, idx_constraint) = -0.5;
				idx_sdpt3 = idx_sdpt3 + 1;
				idx_sedumi = idx_sedumi + 1;
				idx_constraint = idx_constraint + 1;
			end
			ineq_sdpt3(idx_sdpt3, idx_constraint) = -1;
			ineq_sedumi(idx_sedumi, idx_constraint) = -1;
			idx_sdpt3 = idx_sdpt3 + 1;
			idx_sedumi = idx_sedumi + n - i + 1;
			idx_constraint = idx_constraint + 1;
		end
	end
	fprintf('Done.\n');

	fprintf('Building Target and Equality Constraints...');
	c = mapt_sedumi * coef_t(:, idx_sdpt3);
	SDP.sedumi.c = c;
	eq_sedumi = mapt_sedumi * coef_t(:, idx_sdpt3 + 1 : end - 1);
	At_sedumi = [ moment_sedumi, ineq_sedumi, eq_sedumi, regularization_sedumi ];
	b = sparse(num_sdpt3_variables - nterms + 1 + num_equality, 1, 1, num_sdpt3_variables - nterms + 1 + num_equality, 1);
	SDP.sedumi.b = b;
	SDP.sedumi.At = At_sedumi;
	SDP.sedumi.K = K;
	assert(size(SDP.sedumi.At, 2) == length(SDP.sedumi.b));

	%% Parse the data into the standard SDPT3 format
	C = cell(size(K.s));
	eq_sdpt3 = mapt_sdpt3 * coef_t(:, idx_sdpt3 + 1 : end - 1);
	A_sdpt3 = [ moment_sdpt3, ineq_sdpt3, eq_sdpt3, regularization_sdpt3 ].';
	assert(size(A_sdpt3, 1) == size(At_sedumi, 2));
	At = cell(size(K.s));
	blk = cell(length(K.s), 2);
	for i = 1 : length(K.s)
		blk{i, 1} = 's';
		blk{i, 2} = K.s(i);
		At{i} = A_sdpt3(:, 1 : K.s(i) * (K.s(i) + 1) / 2).';
		A_sdpt3 = A_sdpt3(:, K.s(i) * (K.s(i) + 1) / 2 + 1 : end);
		C{i} = reshape(c(1 : K.s(i) * K.s(i)), K.s(i), K.s(i));
		c = c(K.s(i) * K.s(i) + 1 : end);
	end
	SDP.sdpt3.C = C;
	SDP.sdpt3.blk = blk;
	SDP.sdpt3.b = b;
	SDP.sdpt3.At = At;
	fprintf('Done.\n');
	fprintf('====================================================================\n\n\n');
    fprintf('Summary:\n\t%d moment constraints.\n\t%d inequality constraints.\n\t%d equality constraints.\n\t%d regularization constraints', size(moment_sdpt3, 2), size(ineq_sdpt3, 2), size(eq_sdpt3, 2), size(regularization_sdpt3, 2));
end
