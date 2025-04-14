function [V] = symbasis(n, doreal)
%SYMBASIS Basis for the set of real symmetric matrices within the linear 
% space of all n x n matrices.

% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   Feb 27, 2018
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in the home directory.

ndof = n*(n+1)/2;

% Enumerate the subscripts of a lower triangular matrix
[ii,jj] = find(sparse(tril(ones(n))));

% Convert the subscrips into indices
ii1 = ii + (jj-1)*n;
ii2 = (ii-1)*n + jj;

% Number according to DOFs
jj = (1:ndof)';

% Scaling to make the matrix unitary
kk = sqrt(0.5)*ones(ndof,2);
kk(ii1==ii2,:) = 0.5*ones(n,2);

% Output
V = sparse([ii1,ii2],[jj,jj],kk,n^2,ndof);

% Complex basis
if nargin > 1 && ~doreal
    ndof = n*(n-1)/2; %Skew symmetric part

    % Enumerate the subscripts of a lower triangular matrix
    [ii,jj] = find(sparse(tril(ones(n),-1)));

    % Convert the subscrips into indices
    ii1 = ii + (jj-1)*n;
    ii2 = (ii-1)*n + jj;

    % Number according to DOFs
    jj = (1:ndof)';

    % Scaling to make the matrix unitary
    kk = 1i*[-sqrt(0.5)*ones(ndof,1);sqrt(0.5)*ones(ndof,1)];
    
    % Antihermitian part
    V = [V,sparse([ii1,ii2],[jj,jj],kk,n^2,ndof)];
end
end