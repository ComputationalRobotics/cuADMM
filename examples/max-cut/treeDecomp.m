function [T] = treeDecomp( Adj, perm )
%TREEDECOMP Given graph adjacency matrix, compute tree decomposition by
%eliminating with canonical ordering

% Input:
%   Adj     - Graph adjacency matrix
%   perm    - Fill-reducing ordering

% Output:
% structure T with fields:
%   cliques - maximal cliques, ordered in topological ordering
%   ell     - number of cliques
%   omega   - maximum size of any clique; clique number
%   parent  - parent structure for the supernodal elimination tree
%   child   - inverse of parent
%   super   - supernode index -> vertex index
%   isuper  - vertex index -> supernode index

% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   Feb 27, 2018
% Reference: 
% [VA] Vandenberghe, L., & Andersen, M. S. (2015). Chordal graphs and 
% semidefinite optimization. Foundations and Trends in Optimization, 
% 1(4), 241-433.
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in the home directory.

% Input checks
n = size(Adj,1);
assert(size(Adj,2) == n, 'Adjacency matrix must be square');
Adj = logical(Adj); % convert to logical
Adj = Adj | Adj'; % symmetricize
Adj(1:n+1:end) = true; % fill the diagonal

% Compute elimination tree [VA, p.41] which is also a clique tree
% decomposition. 
% Notes:
%    1. The columns of the matrix L are the cliques of the clique tree
%    2. We apply postordering as early as possible so that all subsequent
%       code can stick to the canonical ordering 1:n.
[~,~,parent,post,L] = symbfact(Adj(perm,perm),'sym','lower');
L(perm,:) = L; L = L(:,post);
parent = reorderTree(parent,post);
clique = cell(1,n);
for i = 1:n
    clique{i} = find(L(:,i));
end

% Merge redundant cliques into supernodes.
[clique, parent] = supernode(clique, parent);
ell = numel(clique);
   
% Generate a postordering via recursive DFS
% TODO: replace with nonrecursive DFS!
[ch, root] = p2ch(parent); % Get children pointers
nr = numel(root);
post = cell(1,nr);
for i = 1:nr
    post{i} = dfs(root(i));
end
post = [post{:}];
function [ord] = dfs(v) % Recursive depth-first search
    if isempty(ch{v}) % Leaf
        ord = v;
    else
        % Traverse children
        ord = cell(1,numel(ch{v}));
        for vc = 1:numel(ch{v})
            ord{vc} = dfs(ch{v}(vc));
        end
        ord = [ord{:}, v];
    end
end

% Apply the reordering to our decomposition
clique = clique(post);
parent = reorderTree(parent, post);

% Compute the supernode <-> vertices map
super = cell(1,ell);
isuper = zeros(1,n);
for v = 1:ell
    vp = parent(v);
    if vp > 0
        super{v} = setdiff(clique{v},clique{vp});
    else
        super{v} = clique{v};
    end
    isuper(super{v}) = v;
end

% Store as a structure. (Make this as portable as possible)
T = struct;
T.clique = clique;
T.n      = n; 
T.M      = nnz(tril(Adj)); 
T.ell    = ell;
T.nn     = cellfun(@numel, clique);
T.omega  = max(T.nn);
T.parent = parent;
T.child  = p2ch(parent);
T.super  = super;
T.isuper = isuper;

end

function [clique2, parent2] = supernode(clique, parent)
%SUPERNODE Maximal supernodes and supernodal elimination tree
% Vandenberghe and Andersen Algorithm 4.1

% Input checks
n = numel(parent);
assert(numel(clique) == n, 'mismatch in number of elements');
assert(all(parent > (1:n) | parent ==0), 'cliques must be given in postordering');

% Data structures
deg = cellfun(@numel,clique);
ch = p2ch(parent);

isuper = zeros(1,n);
parent2 = zeros(1,n);
repre = zeros(1,n);
ell = 0;
for v = 1:n
    % Check to see if we should make a new supernode
    makeNew = true;
    for w = ch{v}
        if deg(w) == deg(v) + 1
            makeNew = false;
            break;
        end
    end
    % If yes, create a new supernode u. 
    % If not, get supernode u to add to
    if makeNew
        ell = ell+1;
        u = ell;
        repre(u) = v;
    else
        u = isuper(w);
    end
    % Add v to supernode u
    isuper(v) = u;
    for w = ch{v}
        z = isuper(w);
        if z ~= u
            parent2(z) = u;
        end
    end
end
parent2 = parent2(1:ell);
clique2 = clique(repre(1:ell));
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

function [ parent2 ] = reorderTree( parent, v )
%RELABELTREE Reorder a given tree vector
% Let v : order -> vertex. Then define parent2 such that
%   if j = parent2(i), then v(j) = parent( v(i) )
%      0 = parent2(i), then   0  = parent( v(i) )

n = numel(parent); v = v(:)';
assert(all(sort(v) == 1:n), 'provided v is not an ordering!');
vinv(v) = 1:n; % inverse v : vertex -> order
parent2 = zeros(1,n);
for i = 1:n
    vj = parent(v(i)); 
    if vj > 0 % not a root
        parent2(i) = vinv(vj);
    end
end
end

