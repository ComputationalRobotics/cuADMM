function [ c,A,lb,ub ] = genMAXCUT( Adj, k)
%genMAXCUT Given the graph, generate sparse version of the max cut problem

% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   August 8th, 2018
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.

if nargin < 2 || isempty(k)
    k = 2;
end
assert(k>1 && k == floor(k),'Meaningless choice of k');

% Get graph
n = size(Adj,1);
assert(size(Adj,2) == n);
assert(issparse(Adj));

% Make weights positive
Adj = abs(Adj)/2;
Adj = Adj + Adj';

% L = D - A
% The diagonal of A does not matter!!
Lap = diag(sum(Adj,2))- Adj;

% Diag elements
A = sparse(1:(n+1):n^2, 1:n, 1, n^2, n, n);
lb = ones(n,1); ub = ones(n,1);
c = -(k-1)/(2*k)*Lap(:);
if k == 2, return; end

% Off-diag elements
% Get graph edges
[ii,jj,~] = find(tril(Adj,-1));
ii = ii(:).'; jj = jj(:).'; 
m = numel(ii);

% Generate constraints
A = [A, sparse([(ii-1)*n + jj; (jj-1)*n + ii], repmat(1:m,2,1), 1, n^2, m, 2*m)];
lb = [lb; -2/(k-1) * ones(m,1)];
ub = [ub; inf(m,1)];
end

