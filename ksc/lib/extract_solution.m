% Extract solutions from the moment matrix if the flatness condition is satisfied.

% Input
% moment: moment matrix
% x: polynomial variables
% v: monomial basis

% Output
% sol: a matrix whose columns consist of optimal solutions

function sol = extract_solution(moment, x, v, tol)
    if nargin < 4
        tol = 1e-3;
    end
    n = length(x);
    [U, pivot] = rref(moment', tol);
    rk = length(pivot);
    U = U(1:rk,:)';
    w = v(pivot);
    for i = 1:n
        xw = x(i)*w;
        k = [];
        for j = 1:rk
            k = [k; find(ismember(xw(j),v))];           
        end
        N{i} = U(k,:);
    end
    
    % Create random convex combination
    rands = rand(n,1);
    rands = rands/sum(rands);
    M = zeros(rk);
    for i = 1:n
        M = M + rands(i)*N{i};
    end

    [L,~] = schur(M);
    sol = zeros(n,rk);
    for i = 1:rk
        for j = 1:n
            sol(j,i) = L(:,i)'*N{j}*L(:,i);
        end
    end
end


function out = ismember(x,v)
out = zeros(length(v),1);
for i = 1:length(v)
    out(i) = isequal(x,v(i));
end
end