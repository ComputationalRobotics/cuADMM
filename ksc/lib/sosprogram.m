% This function converts the following SOS program to an SDP:
% min  a'u
% s.t. f_0 + f_1u_1 + ... + f_mu_m = s_0 + s_1g_1 + ... + s_pg_p + t_1h_1 + ... + t_qh_q
%      s0, s_1, ..., s_p are SOS, t_1, ..., t_q are polynomials
% d is the relaxation order.
% It outputs sdpt format data.

function [blk, At, C, b] = sosprogram(a, f, g, h, x, d)
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

fbasis = get_basis(n, d);
flb = size(fbasis, 2);
if p > 0
glb = zeros(p, 1);
gbasis = cell(1, p);
for k = 1:p
    gbasis{k} = get_basis(n, d-ceil(dg(m+k)/2));
    glb(k) = size(gbasis{k}, 2);
end
end
if q > 0
hlb = zeros(q, 1);
hbasis = cell(1, q);
for k = 1:q
    hbasis{k} = get_basis(n, 2*d-dg(m+p+k));
    hlb(k) = size(hbasis{k}, 2);
end
end
sp = get_basis(n, 2*d);
lsp = size(sp, 2);

b = sparse(lsp, 1);
for i = 1:lt(1)
    locb = bfind(sp, lsp, supp{1}(:,i), n);
    b(locb) = coe{1}(i);
end

blk{1,1} = 's';
blk{1,2} = flb;
C{1,1} = sparse(flb,flb);
At{1,1} = sparse(flb*(flb+1)/2, lsp);
for i = 1:flb
    for j = i:flb
        bi = fbasis(:,i) + fbasis(:,j);
        locb = bfind(sp, lsp, bi, n);
        if i == j
            At{1,1}(j*(j+1)/2, locb) = 1;
        else
            At{1,1}(i+j*(j-1)/2, locb) = sqrt(2);
        end
    end
end

for k = 1:p
    blk{1+k,1} = 's';
    blk{1+k,2} = glb(k);
    C{1+k,1} = sparse(glb(k),glb(k));
    At{1+k,1} = sparse(glb(k)*(glb(k)+1)/2, lsp);
    for i = 1:glb(k)
        for j = i:glb(k)
            for l = 1:lt(k+m)
                bi = gbasis{k}(:,i) + gbasis{k}(:,j) + supp{k+m}(:,l);
                locb = bfind(sp, lsp, bi, n);
                if i == j
                    At{1+k,1}(j*(j+1)/2, locb) = coe{k+m}(l);
                else
                    At{1+k,1}(i+j*(j-1)/2, locb) = sqrt(2)*coe{k+m}(l);
                end
            end
        end
    end
end

blk{2+p,1} = 'u';
blk{2+p,2} = sum(hlb)+m-1;
C{2+p,1} = sparse(sum(hlb)+m-1,1);
At{2+p,1} = sparse(sum(hlb)+m-1,lsp);
loc = 0;
for k = 1:q
    for i = 1:hlb(k)
        for j = 1:lt(k+m+p)
            bi = hbasis{k}(:,i) + supp{k+m+p}(:,j);
            locb = bfind(sp, lsp, bi, n);
            At{2+p,1}(loc+i, locb) = coe{k+m+p}(j);
        end
    end
    loc = loc + hlb(k);
end
for k = 1:m-1
    C{2+p,1}(sum(hlb)+k) = a(k);
    for i = 1:lt(k+1)
        locb = bfind(sp, lsp, supp{k+1}(:,i), n);
        At{2+p,1}(loc+k, locb) = -coe{k+1}(i);
    end
end
end
