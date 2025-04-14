function [Supp2,A2,B2,rhs] = split(parent, Supp, Mp)
%SPLIT
% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   Feb 27, 2018
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in the home directory.

% Input check
% Make Supp set sorted
if any(diff(Supp) < 0)
    [Supp,idx] = sort(Supp);
    Mp = Mp(idx);
end

% Get smallest connected subtree containing support of the split constraint
% by solving a Steiner subtree problem
[Supp2, subpar] = subtree(parent, Supp);
subch   = p2ch(subpar);

% SPLITTING
% 1. Count the number of children per node, assign each child its own
%    index and auxillary variable at its parent
% 2. Iterate through every node, generate the following constraint for 
%    each node
%           sum_i u_ij + Aj o Xj - u_jk  = 0
%    at the j-th node with children i and parent k. If the node is root,
%    then
%           sum_i u_ij + Aj o Xj         = b

% helpers for allocating the constraints
% note that Supp and Supp2 are both sorted
Supp = [Supp,0]; pnt = 1;

% Begin routine
m = numel(Supp2);
A2 = cell(1,m); B2 = cell(1,m); rhs = zeros(m,1); 
for j = 1:m % every element in Supp2
    if Supp2(j) == Supp(pnt)
        A2{j} = Mp{pnt};
        pnt = pnt + 1;
    end
    
    % Construct B matrix
    chj = subch{j}; nchj = numel(chj);
    if nchj > 0
        ii = [chj(:)'; j*ones(1,nchj)]; 
        jj = repmat(1:numel(chj),2,1);
        kk = repmat([-1;1], 1, nchj);
        B2{j} = sparse(ii,jj,kk,m,nchj,2*nchj);
    end
    
    % Construct rhs
    if subpar(j) == 0
        rhs(j) = 1;
    end
end
end

function [Supp, subpar] = subtree(parent, cover)
% Given a tree a set of vertices to cover, find the smallest connected
% subtree containing these vertices
% Input.  cover  -- a list of must-keep nodes. 
%         parent -- parent vector for the tree
% Output. Supp   -- the support of the desired subtree
%         subpar -- parent structure for the nodes of the subtree

% Total complexity is 
%     \Theta ( t h log( t h ) )
% where t is the number of elements in cover and h is the height of the
% tree. First three steps are \Theta ( t h ) and the last adds the log from
% sorting (inside call to unique).

% Path to least common ancestor
% 1. Find path from every must-keep node to the root node.
nn = numel(cover);
paths = cell(1,nn);
for j = 1:nn
    v = cover(j);
    paths{j} = v;
    while parent(v) > 0
        v = parent(v);
        paths{j} = [v,paths{j}];
    end
end

% 2. Find least common ancestor
% at output, we have path{i}(1:d) = path{j}(1:d) for all i,j
len = cellfun(@numel, paths); % path lengths
minl = min(len);
anc = zeros(nn,minl);
for i = 1:nn
    anc(i,:) = paths{i}(1:minl);
end
chk = [sum(abs(diff(anc,[],1)),1),1];
d = find(chk,1)-1;
if d == 0 
    error('The nodes in cover are disconnected on the tree!'); 
end

% 3. Truncate all paths up to the most recent common ancestor.
for j = 1:nn
    paths{j} = paths{j}(end:-1:d);
end

% 4. Merge the paths into one tree and build the associated parent vector
%    The logic is the following. Given the following facts
%       paths(ia(i)) = Supp(i), paths(j) = Supp(ic(j)), and 
%       par(paths(i)) = paths(i+1) if paths(i) ~= root
%    we have
%       Supp(i) = paths(ia(i)) and Supp(i) is not a root
%    take the parent and then reverse the mapping
%       par(Supp(i)) = par(paths(ia(i))) = paths(ia(i) + 1)
%                    = Supp(k) where k = ic(ia(i) + 1)
paths = [paths{:}]; % the last element is the root
[Supp, ia, ic] = unique(paths); % see the manual
root = ic(end); n = numel(Supp);
idx = [1:(root-1),(root+1):n]; % nonroot indices
subpar = zeros(1,n);
subpar(idx) = ic(ia(idx)+1); % apply logic from above

end

function [ch, root] = p2ch(p)
% Convert the list of parent vectors into a cell array of children vectors.
n = length(p);
ch = cell(size(p));
root = [];
for i = 1:n
    par = p(i);
    if par > 0
        % Push child into parent
        ch{par} = [ch{par}, i];
    elseif par == 0
        root = [root, i];
    end
end
end
