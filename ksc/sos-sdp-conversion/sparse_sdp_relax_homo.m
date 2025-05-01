function SDP = sparse_sdp_relax_homo(problem)
    warning("Function `sparse_sdp_relax_homo' is deprecated. use `sparse_sdp_relax' instead.");
	problem.regularization = problem.z ^ max(deg(problem.objective), [], 'all');
    SDP = sparse_sdp_relax(problem);
end
