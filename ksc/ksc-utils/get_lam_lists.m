function [lam1_list, lam2_list] = get_lam_lists(moment_info)
    % moment_info should be: 
    % {[U1, D1], [U2, D2] ...}
    lam1_list = [];
    lam2_list = [];
    for i = 1: size(moment_info, 1)
        D = moment_info{i, 2};
        lam1_list = [lam1_list; D(1)];
        lam2_list = [lam2_list; D(2)];
    end
end