function [ matrix, svec ] = moment_variable(vars, deg)
    if (isempty(vars))
        vars = msspoly(zeros([0, 1]));
    end
	num_vars = uint64(length(vars));
	deg = uint64(deg);
    p = mymonomial_core(num_vars, 2 * deg);
	[ matrix_coeff, svec_coeff, num_mono ] = moment_variable_coeff_core(num_vars, deg);
    matrix = reshape(recomp(vars, p, matrix_coeff), [ num_mono, num_mono ]);
	svec = recomp(vars, p, svec_coeff);
end
