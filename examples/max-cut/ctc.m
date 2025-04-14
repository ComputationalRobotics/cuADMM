function [Amat,bvec,cvec,Kcone,info] = ctc( c, A, lb, ub, dualize)
%CTC Clique-tree conversion and dualized clique tree conversion.
% Reformulate
%   min c'x s.t. lb <= A*x <= ub, x \in SDP(n)
% to
%   max -cc'*y s.t. [ AA ]*y = [ b ]
%                   [ NN ]   = [ 0 ]
%                   y \in SDP(n1) x SDP(n2) x ... 
% and output a problem in SeDuMi format.

% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   August 8th, 2018
% Reference: R.Y. Zhang, J. Lavaei, "Sparse Semidefinite Programs with 
%            Guaranteed Near-Linear Time Complexity via Dualized Clique 
%            Tree Conversion", https://arxiv.org/abs/1710.03475
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.

max_clique_size = 60; % Quit if clique is bigger than this
real_embedding  = true; % Embed the Hermitian SDP cone into the real SDP cone?
verbose = 2;
tt = tic;

% Do we dualize?
if nargin < 5 || isempty(dualize)
    dualize = true;
end

% Defaults
if size(A,2) > size(A,1), A = A.'; end
if nargin < 4 || isempty(ub)
    ub = lb;
elseif isempty(lb)
    lb = ub;
end
lb = lb(:); ub = ub(:); c = c(:);
assert(all(lb <= ub), 'All lower-bounds must be smaller than the upper-bounds!');
m = size(A,2);

%--------------------------------------------------------------------------
% Tree decomposition
%--------------------------------------------------------------------------
% Get adjacency pattern
Adj = any([A, c],2);
n = floor(sqrt(length(Adj)));
assert(length(Adj) == n^2, 'number of rows in A must be a perfect square');
Adj = reshape(any([A, c],2), n, n);

% Compute the tree decomposition
perm = amd(Adj);
[T] = treeDecomp(Adj, perm);
if max(T.nn) > max_clique_size
    error('Cliques are way too large!');
end
if verbose > 0
	fprintf('complete\n');
    fprintf(' Num. cliques: %d \n Max cliques size: %d\n', ...
        T.ell, T.omega);
end

%--------------------------------------------------------------------------
% Generate constraints
%--------------------------------------------------------------------------
if verbose > 0
	fprintf('Beginning constraint splitting....');
end

% Allocate objective
c2 = allocate(T, c(:));

% Allocate constraints and perform splitting
A2 = allocate(T, A);
blksiz_r1 = ones(m,1);
blksiz_c1 = T.nn.^2;

% Overlap constraints
doreal = isreal(A) && isreal(c);
[ R, blksiz_r2, V] = overlaps( T, doreal);
P   = blkdiag(V{:});
blksiz_c2 = cellfun(@(X)size(X,2), V);

% Convert from blocks into sparse matrices
% A2 is transpose because we need to extract rows out of them, and
% it is much cheaper to index columns than rows.
obj = real(cell2sparse(c2, 1, blksiz_c1)*P);
R   = real(cell2sparse(R, blksiz_r2, blksiz_c2));
A2t = real(cell2sparse(A2, blksiz_r1, blksiz_c1)*P)';

if verbose > 0
	fprintf('complete\n');
end

if dualize

%--------------------------------------------------------------------------
% Dualize and output
%  lb <= A2*P*y <= ub
%  |R * P *y|   <= 0 * x0
%  eig(-P*y)   <= 0
%--------------------------------------------------------------------------

% Assemble matrices
% Isolate equality constraints
do_eq = (ub-lb) < 1e-8; % Turn this into bound
do_lb = isfinite(lb) & ~do_eq;
do_ub = isfinite(ub) & ~do_eq;

% Apply diagonal preconditioning to AtA
D = diag(sparse(1./sqrt(sum(A2t.^2,1))));
ub = D*ub; lb = D*lb; A2t = A2t*D;

% LP cone
Amat_t_lp = [-A2t(:,do_lb)'; A2t(:,do_ub)'];
rhs_lp  = [-lb(do_lb); ub(do_ub)];
m_lp    = numel(rhs_lp);

% SOCP cone
m_over  = size(R,1);
z_row   = sparse(1, size(A2t,1));
Amat_t_eq = [A2t(:,do_eq)'; R];
rhs_eq  = [0.5*(lb(do_eq)+ ub(do_eq)); zeros(m_over,1)];
m_eq    = numel(rhs_eq);

Kcone   = struct;
Kcone.l = m_lp;
Kcone.q = 1 + m_eq;

% SDP cone
if doreal
    Amat_sdp = -P;
    Kcone.s = T.nn;
else
if real_embedding
    V2 = realembed(V);
    Amat_sdp = -blkdiag(V2{:});
    Kcone.s = 2*T.nn;
else
    Amat_sdp = -P;
    Kcone.s = T.nn;
    Kcone.scomplex = 1:T.ell;
end
end
m_sdp = size(Amat_sdp,1);
rhs_sdp  = zeros(m_sdp,1);

% Put everything together
Amat = [Amat_t_lp; z_row; Amat_t_eq; Amat_sdp];
cvec = [rhs_lp;  0;     rhs_eq;  rhs_sdp];
bvec = -obj(:);

else

%--------------------------------------------------------------------------
% Dualize and output
%  lb <= A2*P*y <= ub
%  |R * P *y|   <= 0 * x0
%  eig(-P*y)   <= 0
%--------------------------------------------------------------------------

Kcone = struct;
Kcone.l = 0;
Kcone.q = 0;
Kcone.s = T.nn;
if ~doreal
    Kcone.scomplex = 1:T.ell;
    if real_embedding
        V2 = realembed(V);
        P = blkdiag(V2{:});
        Kcone.s = 2*T.nn;
    end
end

% Assemble matrices
% Isolate equality constraints
do_eq = (ub-lb) < 1e-8; % Turn this into bound
do_lb = isfinite(lb) & ~do_eq;
do_ub = isfinite(ub) & ~do_eq;

% Apply diagonal preconditioning to AtA
D = diag(sparse(1./sqrt(sum(A2t.^2,1))));
ub = D*ub; lb = D*lb; A2t = A2t*D;

% Inequality constraints
n_lp    = nnz(do_lb) + nnz(do_ub);
Amat_t_lp = [speye(n_lp); P*[-A2t(:,do_lb), A2t(:,do_ub)]];
rhs_lp  = [-lb(do_lb); ub(do_ub)];
Kcone.l = n_lp;

% Additional equality
m_eq    = nnz(do_eq); m_over = size(R,1);
Amat_t_eq = [sparse(n_lp, m_eq+m_over); P*[A2t(:,do_eq), R']];
rhs_eq  = [0.5*(lb(do_eq)+ ub(do_eq)); zeros(m_over,1)];

% Put everything together
Amat = [Amat_t_lp, Amat_t_eq];
bvec = [rhs_lp;  rhs_eq];
cvec = [sparse(n_lp,1); P*obj(:)];
    
end % if dualize

%--------------------------------------------------------------------------
% Final processing
%--------------------------------------------------------------------------

% Information needed for recovery
info.symbasis  = V;
info.treeDecomp = T;
info.perm = perm;
info.time = toc(tt);
end

function [V2] = realembed(V)
%Embedding the size-n hermitian PSD cone into the
%size-2n real symmetric PSD cone.
V2 = cell(size(V));
for j = 1:numel(V)
    V{j} = full(V{j});
    [n2, m] = size(V{j});
    V2{j} = zeros(4*n2, m);
    n = floor(sqrt(n2));
    for i = 1:m
        tmp = reshape(V{j}(:,i),n,n);
        tmp2 = [real(tmp), -imag(tmp); imag(tmp), real(tmp)];
        V2{j}(:,i) = tmp2(:);
    end
    V2{j} = sparse(V2{j});
end
end

function [A2] = allocate(T, A)
%ALLOCATE Convert sparse constraints into block compressed-row format
m = size(A,2);
A2 = cell(m,2);
for i = 1:m
    % PREPROCESS
    M = reshape(A(:,i), T.n, T.n);
    M = (M+M')/2;

    % ALLOCATION
    % Get the smallest support that covers the nonzero elements in the
    % constraint.
    % 1. Get support that covers A
    % 2. Trim nodes using leaf-removal. Traverse bottom-up, and at each
    %    each node, check if the unique elements of that node are in the
    %    support. If yes, cover that node; if not, ignore it.
    %    The unique elements are exactly M( cliques{Jk}, super{Jk} )
    Supp = unique(T.isuper(any(M,1)));
    nn = numel(Supp); 
    cover = false(1,nn);
    Mp = cell(1,nn);
    for v = 1:numel(Supp) 
        Jk = Supp(v); % This tree node
        % If A(:,Uj) is nonzero
        if nnz(M(T.clique{Jk},T.super{Jk})) > 0 
            % Must include in cover
            constr = M(T.clique{Jk}, T.clique{Jk});
            Mp{v} = constr(:)';

            % Has been convered
            cover(v) = true;
            M(T.clique{Jk}, T.clique{Jk}) = 0;
        end
    end
    A2{i,1} = Supp(cover);
    A2{i,2} = Mp(cover); 
end

end


function [ sparsemat ] = cell2sparse( cellmat, blksiz_r, blksiz_c)
%CELL2SPARSE
% cellmat is a two-column cell matrix. First cell column is the (column)
% support, second cell column are the elements.
% (compressed row format)
% WARNING: DOES NOT CHECK BLOCK SIZES!! MAKE SURE THESE ARE CORRECT

% Basic input check
assert(numel(blksiz_r) == size(cellmat,1), 'number of block-rows mismatch');

% Compute shifts
rr = [0; cumsum(blksiz_r(:))];
cc = [0; cumsum(blksiz_c(:))];

% Process nonzero elements
m = size(cellmat,1);
[ii,jj,kk] = deal(cell(m,1));
for i = 1:m
    % Set-up
    j = cellmat{i,1}; Elem = cellmat{i,2};
    numblks = numel(j); 
    assert(numel(Elem) == numblks, 'Supp Elem mismatch');
    
    % Retrive elements
    [ii_i,jj_i,kk_i] = deal(cell(1,numblks));
    for blk = 1:numblks
        if isempty(Elem{blk}), continue; end
            % Find nonzeros and shift them
            [ii_i{blk}, jj_i{blk}, kk_i{blk}] = find(Elem{blk});
            ii_i{blk} = ii_i{blk} + rr(i);
            jj_i{blk} = jj_i{blk} + cc(j(blk));
            % Force into columns
            ii_i{blk} = ii_i{blk}(:);
            jj_i{blk} = jj_i{blk}(:);
            kk_i{blk} = kk_i{blk}(:);
    end

    % Fuse them together and make sparse matrix
    ii{i} = cat(1,ii_i{:});
    jj{i} = cat(1,jj_i{:});
    kk{i} = cat(1,kk_i{:});
end
ii = cat(1,ii{:});
jj = cat(1,jj{:});
kk = cat(1,kk{:});
sparsemat = sparse(ii,jj,kk,rr(end), cc(end));
end

function [ R, blksiz_r, V] = overlaps( T, doreal)
%OVERLAPS 

% Count the number of roots in order to determine the rank of this
% projection
root = find(T.parent == 0);
nr = numel(root);
R = cell(T.ell-nr,2);

% Set up bases
V = cell(1, T.ell);
for j = 1:T.ell
    V{j} = sparse(symbasis(T.nn(j), doreal));
end

% List of edges
v = 1:T.ell; v(root) = [];
blksiz_r = zeros(T.ell-nr,1);
for i = 1:(T.ell-nr)
    vi = v(i); vpi = T.parent(vi);
    
    % Compute overlaps and basis
    [tmp, mj] = Rblk(T.clique{vpi}, T.clique{vi});
    Vo = symbasis(mj, doreal);
    
    % Form the matrix
    R{i,1} = [vi, vpi]; % Support
    R{i,2} = {Vo'*(tmp*V{vi}), ... Values
             -Vo'*(Rblk(T.clique{vi}, T.clique{vpi}) * V{vpi})};
    blksiz_r(i) = size(Vo,2);
end
end

function [mat, mj] = Rblk(I,J)
% Generate the matrix that implements the operator
%   R_IJ(X_JJ) = X_WW, W = I \cap J
    mask = find(ismember(J(:), I));
    nj = numel(J); mj = numel(mask);
    mask2 = bsxfun(@plus, mask, nj*(mask-1)');
    mat = sparse(mask2(:),1:mj^2,1,nj^2,mj^2)';
end


function chkCone(K)
nflqr = 0;
if isfield(K,'f'), nflqr = nflqr + K.f; end
if isfield(K,'l'), nflqr = nflqr + K.l; end
if isfield(K,'q'), nflqr = nflqr + K.q; end
if isfield(K,'r'), nflqr = nflqr + K.r; end
assert(nflqr == 0, 'f, l, q, r, cones not yet supported!');
assert(isfield(K,'s'), 'Must provide s cone');
assert(numel(K.s)==1, 'Must provide a only single s cone');
end
