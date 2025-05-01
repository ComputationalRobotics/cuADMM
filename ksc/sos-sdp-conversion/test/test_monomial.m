vars = msspoly('x', 100);
monos = mymonomial(vars, 3);
parfor i = 1 : length(monos)
    if monomial_id(vars, monos(i)) ~= i
        i
        monos(i)
        monomial_id(vars, monos(i))
        error('Error!');
    end
end