% This function converts the following sparse SOS program to an SDP:
% min  a'u
% s.t. f_0 + f_1u_1 + ... + f_mu_m = \sum_{i=1}^l(s_{i0} + \sum_{j\in I_i}s_{j}g_j + \sum_{j\in J_i}t_jh_j)
%      s_{10}, ..., s_{l0}, s_1, ..., s_p are SOS, t_1, ..., t_q are polynomials

% cliques: array to represent the variable clique structure
% I: array of indices of inequality constraints associated to each clique 
% J: array of indices of equality constraints associated to each clique
% order: vector of relaxation orders
% It outputs sdpt format data.

function [blk, At, C, b] = sparsesos(a, f, g, h, x, cliques, I, J, order)
polys = [f; g; h];
n = length(x);
m = length(f);
p = length(g);
q = length(h);
coe = cell(1, m+p+q);
supp = cell(1, m+p+q);
lt = zeros(1, m+p+q);
dg = zeros(1, m+p+q);
[~, tsupp, tcoe] = decomp(polys);
for k = 1:m+p+q
    [~, loc, coe{k}] = find(tcoe(k,:));
    lt(k) = length(loc);
    supp{k} = tsupp(loc,:)';
    dg(k) = deg(polys(k), x);
end
nc = length(order);

sp = [];
flb = zeros(1, nc);
fbasis = cell(1, nc);
gbasis = cell(1, nc);
glb = cell(1, nc);
hbasis = cell(1, nc);
hlb = cell(1, nc);
for i = 1:nc
    sp = [sp get_basis(n, 2*order(i), cliques{i})];
    fbasis{i} = get_basis(n, order(i), cliques{i});
    flb(i) = size(fbasis{i}, 2);
    if ~isempty(I{i})
        glb{i} = zeros(length(I{i}), 1);
        gbasis{i} = cell(1, length(I{i}));
        for k = 1:length(I{i})
            gbasis{i}{k} = get_basis(n, order(i)-ceil(dg(m+I{i}(k))/2), cliques{i});
            glb{i}(k) = size(gbasis{i}{k}, 2);
        end
    end
    if ~isempty(J{i})
        hlb{i} = zeros(length(J{i}), 1);
        hbasis{i} = cell(1, length(J{i}));
        for k = 1:length(J{i})
            hbasis{i}{k} = get_basis(n, 2*order(i)-dg(m+p+J{i}(k)), cliques{i});
            hlb{i}(k) = size(hbasis{i}{k}, 2);
        end
    end
end
sp = unique(sp', 'rows');
sp = sortrows(sp)';
lsp = size(sp, 2);

b = sparse(lsp, 1);
for i = 1:lt(1)
    locb = nbfind(sp, lsp, supp{1}(:,i), n);
    b(locb) = coe{1}(i);
end

nb = 1;
for l = 1:nc
    blk{nb,1} = 's';
    blk{nb,2} = flb(l);
    C{nb,1} = sparse(flb(l), flb(l));
    At{nb,1} = sparse(flb(l)*(flb(l)+1)/2, lsp);
    for i = 1:flb(l)
        for j = i:flb(l)
            bi = fbasis{l}(:,i) + fbasis{l}(:,j);
            locb = nbfind(sp, lsp, bi, n);
            if i == j
                At{nb,1}(j*(j+1)/2, locb) = 1;
            else
                At{nb,1}(i+j*(j-1)/2, locb) = sqrt(2);
            end
        end
    end
    nb = nb + 1;
    for k = 1:length(I{l})
        blk{nb,1} = 's';
        blk{nb,2} = glb{l}(k);
        C{nb,1} = sparse(glb{l}(k), glb{l}(k));
        At{nb,1} = sparse(glb{l}(k)*(glb{l}(k)+1)/2, lsp);
        for i = 1:glb{l}(k)
            for j = i:glb{l}(k)
                for s = 1:lt(I{l}(k)+m)
                    bi = gbasis{l}{k}(:,i) + gbasis{l}{k}(:,j) + supp{I{l}(k)+m}(:,s);
                    locb = nbfind(sp, lsp, bi, n);
                    if i == j
                        At{nb,1}(j*(j+1)/2, locb) = coe{I{l}(k)+m}(s);
                    else
                        At{nb,1}(i+j*(j-1)/2, locb) = sqrt(2)*coe{I{l}(k)+m}(s);
                    end
                end
            end
        end
        nb = nb + 1;
    end
    if ~isempty(J{l})
    blk{nb,1} = 'u';
    blk{nb,2} = sum(hlb{l});
    C{nb,1} = sparse(sum(hlb{l}),1);
    At{nb,1} = sparse(sum(hlb{l}),lsp);
    loc = 0;
    for k = 1:length(J{l})
        for i = 1:hlb{l}(k)
            for j = 1:lt(J{l}(k)+m+p)
                bi = hbasis{l}{k}(:,i) + supp{J{l}(k)+m+p}(:,j);
                locb = nbfind(sp, lsp, bi, n);
                At{nb,1}(loc+i, locb) = coe{J{l}(k)+m+p}(j);
            end
        end
        loc = loc + hlb{l}(k);
    end
    nb = nb + 1;
    end
end
blk{nb,1} = 'u';
blk{nb,2} = m-1;
C{nb,1} = sparse(m-1,1);
At{nb,1} = sparse(m-1,lsp);
for k = 1:m-1
    C{nb,1}(k) = a(k);
    for i = 1:lt(k+1)
        locb = nbfind(sp, lsp, supp{k+1}(:,i), n);
        At{nb,1}(k, locb) = -coe{k+1}(i);
    end
end
end
