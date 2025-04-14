% Define the row indices (source nodes)
row_indices = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6];

% Define the column indices (destination nodes)
col_indices = [2, 3, 6, 3, 1, 4, 2, 5, 6, 6, 4];

% Define the weights for each edge
weights = [2, 3, 1, 4, 2, 5, 4, 3, 4, 2, 4];

% Create a sparse 6x6 matrix (undirected, so make sure to symmetrize)
Adj = sparse(row_indices, col_indices, weights, 6, 6);

% Symmetrize the matrix to reflect undirected edges
Adj = Adj + Adj';

% Display the sparse matrix
disp(full(Adj));

[ c,A,lb,ub ] = genMAXCUT( Adj, 2);