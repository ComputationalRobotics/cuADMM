pathinfo = dictionary();
%pathinfo("mosek") = "~/matlab-install/mosek/10.1/toolbox/r2017a";
%pathinfo("sedumi") = "~/matlab-install/sedumi";
pathinfo("utils") = "~/matlab-install/lab-code/utils";
%pathinfo("kscutils") = "../ksc-utils";
pathinfo("lib") = "..";
pathinfo("spot") = "~/matlab-install/spotless";
pathinfo("sossdp") = "../";
%pathinfo("sdpnal") = "~/matlab-install/SDPNAL+v1.0";
%pathinfo("manopt") = "~/matlab-install/manopt";
%pathinfo("stride") = "~/STRIDE";
pathinfo("sdpt3") = "~/matlab-install/SDPT3-4.0";
keys = pathinfo.keys;
for i = 1: length(keys)
    key = keys(i);
    addpath(genpath(pathinfo(key)));
end
%{x = msspoly('x', 4);%}
%{% problem.vars = { x(:) };%}
%{problem.vars = { [ x(1); x(2) ], [ x(2); x(3) ], [ x(3); x(4) ] };%}
%{problem.objective = [ [ -x(1) ], [ -x(2) ], [ -x(3) - x(4) ] ];%}
%{problem.inequality = { [ x(1) ], [ x(2) ], [ x(3); x(4) ]};%}
%{problem.equality = { [ x(1)^2 + x(2)^2 - 1 ], [ x(2)^2 + x(3)^2 - 1 ], [ x(3)^2 + x(4)^2 - 1 ] };%}
%{problem.relaxation_order = 2;%}
%{problem.rip_predecessor = 0 : 2;%}
%{SDP = sparse_sdp_relax(problem);%}
%{sol = [];%}
%{for i = 1 : length(problem.vars)%}
    %{v = subs(mymonomial(problem.vars{i}, problem.relaxation_order), problem.vars{i}, ones(size(problem.vars{i})) * sqrt(0.5));%}
    %{sol = [sol; mykron(v, v)];%}
    %{for j = 1 : length(problem.inequality{i})%}
        %{v = mymonomial(problem.vars{i}, floor(problem.relaxation_order - deg(problem.inequality{i}(j) / 2)));%}
        %{sol = [ sol; subs(problem.inequality{i}(j) * mykron(v, v), problem.vars{i}, ones(size(problem.vars{i})) * sqrt(0.5)) ];%}
    %{end%}
%{end%}

%{sol = double(sol);%}
%{disp(norm(SDP.sedumi.At.' * sol - SDP.sedumi.b, 2));%}

%prob = convert_sedumi2mosek(SDP.sedumi.At, SDP.sedumi.b, SDP.sedumi.c, SDP.sedumi.K);
%[~, res] = mosekopt('minimize info', SDP.mosek);
%[Xopt, yopt, Sopt, obj] = recover_mosek_sol_blk(res, SDP.sdpt3.blk);
% [x, y, info] = sedumi(SDP.sedumi.At, SDP.sedumi.b, SDP.sedumi.c, SDP.sedumi.K);
% disp(dot(SDP.sedumi.c, x));
% save('new.mat', 'x');
load('N=4_freq=10_kappa=2.mat');
[obj, x, ~, ~] = sdpt3(SDP.sdpt3.blk, SDP.sdpt3.At, SDP.sdpt3.C, SDP.sdpt3.b);

