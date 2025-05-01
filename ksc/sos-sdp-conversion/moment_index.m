function id = monomial_id(vars, monomial)
    assert(isequal(class(vars), 'msspoly'), 'variable set is not msspoly.');
    assert(isequal(class(monomial), 'msspoly'), 'monomial is not msspoly.');
    assert(length(monomial) == 1, 'monomial is not a msspoly scalar.');
    id = monomial_id_core(struct(vars(:)), struct(monomial));
end