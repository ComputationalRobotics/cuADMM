addpath(genpath('../../2023-optcontrolsos-codes/utils'));
addpath(genpath('~/spotless'));
addpath(genpath('~/mosek/10.1/toolbox/r2017a'));
addpath(genpath('~/sedumi'));
addpath(genpath('~/sdpt3'));
addpath(genpath('../lib'));
x = msspoly('x', 4);
z = msspoly('z', 1);
% problem.vars = { x(:) };
problem.vars = { [ z; x(1); x(2) ]; [ z; x(2); x(3) ]; [ z; x(3); x(4) ] };
problem.z = z;
problem.objective = [ [ -x(1) ]; [ -x(2) ]; [ -x(3) - x(4) ] ];
problem.inequality = { [ x(1) ]; [ x(2) ]; [ x(3); x(4) ]};
problem.equality = { [ x(1)^2 + x(2)^2 - z^2 ]; [ x(2)^2 + x(3)^2 - z^2 ]; [ x(3)^2 + x(4)^2 - z^2 ] };
problem.relaxation_order = 2;
problem.rip_predecessor = 0 : 2;
SDP = sparse_sdp_relax_homo(problem);
%{sol = [];%}
%{for i = 1 : length(problem.vars)%}
    %{v = subs(mymonomial(problem.vars{i}, problem.relaxation_order), problem.vars{i}, ones(size(problem.vars{i})) * sqrt(0.5));%}
    %{sol = [sol; mykron(v, v)];%}
%{end%}
%{for i = 1 : length(problem.inequality)%}
    %{for j = 1 : length(problem.inequality{i})%}
        %{v = mymonomial(problem.vars{i}, floor(problem.relaxation_order - deg(problem.inequality{i}(j) / 2)));%}
        %{sol = [ sol; subs(problem.inequality{i}(j) * mykron(v, v), problem.vars{i}, ones(size(problem.vars{i})) * sqrt(0.5)) ];%}
    %{end%}
%{end%}

%{sol = double(sol);%}
%{disp(SDP.sedumi.At.' * sol - SDP.sedumi.b);%}

[~, res] = mosekopt('minimize info', SDP.mosek);
%[Xopt, yopt, Sopt, obj] = recover_mosek_sol_blk(res, SDP.sdpt3.blk);
%[x, y, info] = sedumi(SDP.sedumi.At, SDP.sedumi.b, SDP.sedumi.c, SDP.sedumi.K);
%[obj, x, ~, ~] = sdpt3(SDP.sdpt3.blk, SDP.sdpt3.At, SDP.sdpt3.C, SDP.sdpt3.b);

