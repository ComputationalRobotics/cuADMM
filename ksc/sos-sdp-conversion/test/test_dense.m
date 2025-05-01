addpath(genpath('./utils'));
addpath(genpath('./sos-sdp-conversion'));
x = msspoly('x', 6);
problem.vars = x(:);
% problem.vars = { [x(1); x(4)], [x(1), x(2), x(3), x(5)], [x(1), x(3), x(5), x(6)]};
problem.objective = x(2) * x(5) + x(3) * x(6) - x(2) * x(3) - x(5) * x(6) + x(1) * (-x(1) + x(2) + x(3) - x(4) + x(5) + x(6));
problem.inequality = [];
for j = 1 : length(problem.vars)
    problem.inequality = [ problem.inequality; (6.36 - problem.vars(j)) * (problem.vars(j) - 4) ];
end
problem.equality = [];
SDP = dense_sdp_relax(problem, 2);
x = sdpt3(SDP.blk, SDP.At, SDP.C, SDP.b);