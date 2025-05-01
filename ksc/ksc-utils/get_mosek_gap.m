function gap = get_mosek_gap(res)
    dual_gap = abs(res.info.MSK_DINF_INTPNT_DUAL_FEAS);
    primal_gap = abs(res.info.MSK_DINF_INTPNT_PRIMAL_FEAS);
    dual_obj_val = res.sol.itr.dobjval;
    primal_obj_val = res.sol.itr.pobjval;

    dp_gap = abs(primal_obj_val - dual_obj_val) / (1 + abs(primal_obj_val) + abs(dual_obj_val));
    gap = max([dual_gap, primal_gap, dp_gap]);
end
