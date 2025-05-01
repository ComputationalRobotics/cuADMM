% installation script for Sparse Polynomial Optimization

fprintf('\n Installing Sparse SDP Relaxor in %s:\n updating the path...', pwd);
addpath(pwd);
fprintf('\n compiling the binaries...\n');
src = { 'sparse_sdp_relax', 'mymonomial', 'monomial_id', 'moment_variable_coeff' };
for i = 1 : length(src)
	% Change -Wno-unknown-pragmas into -fopenmp to enable OpenMP Parallel Computing
	% Add [ '-D' 'RANDOM_PIVOT' ] to add randomization to the matrix
    % mex('CXXFLAGS=$CXXFLAGS -std=c++20 -Wall -Wextra -Wpedantic -fopenmp -ffast-math', [ '-D', 'NDEBUG' ], [ 'src' filesep src{i} '_core.cpp' ], [ '-I' 'include' ], [ '-l' 'gomp' ]);
    mex('CXXFLAGS=$CXXFLAGS -std=c++20 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas -ffast-math', [ '-D', 'NDEBUG' ], [ 'src' filesep src{i} '_core.cpp' ], [ '-I' 'include' ]);
    % mex('CXXFLAGS=$CXXFLAGS -std=c++20 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas -ffast-math', [ '-D', 'RANDOM_PIVOT' ], [ 'src' filesep src{i} '_core.cpp' ], [ '-I' 'include' ]);
end
fprintf('Done.\n')



