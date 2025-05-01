function basis = get_basis(n, d, varargin)
if nargin == 2
    var = 1:n;
else
    var = varargin{1};
end
lb = nchoosek(length(var)+d, d);
basis = zeros(n, lb);
i = 0;
t = 1;
while i < d + 1
    t = t + 1;
    if basis(var(end), t-1) == i
       if i < d
          basis(var(1), t) = i + 1;
       end
       i = i + 1;
    else
        j = 1;
        while basis(var(j), t-1) == 0
            j = j + 1;
        end
        basis(:, t) = basis(:, t-1);
        if j == 1 
            basis(var(1), t) = basis(var(1), t) - 1;
            basis(var(2), t) = basis(var(2), t) + 1;
        else
            basis(var(1), t) = basis(var(j), t) - 1;
            basis(var(j), t) = 0;
            basis(var(j+1), t) = basis(var(j+1), t) + 1;
         end
    end
end
end