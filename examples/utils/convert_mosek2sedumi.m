function [A, b, c, K] = convert_mosek2sedumi(prob)

% Convert K
K.f = nnz(prob.blx);
K.l = length(prob.blx) - K.f;
K.s = prob.bardim;
n_sdp = length(K.s);

% Convert b
b = prob.blc;
m = length(b);

% blocks
Ainds = zeros(n_sdp + 1, 1);
for j = 1:n_sdp
    Ainds(j + 1) = Ainds(j) + K.s(j).^2;
end

% Convert c
subj = [prob.barc.subj, prob.barc.subj];
subl = [prob.barc.subl, prob.barc.subk];
subk = [prob.barc.subk, prob.barc.subl];
val = [prob.barc.val, prob.barc.val];
num = length(val);
loc = zeros(num, 1);
for v = 1:num
    loc(v) = Ainds(subj(v)) + (subk(v) - 1)*K.s(subj(v)) + subl(v);
    if subk(v) == subl(v)
        val(v) = val(v)/2;
    end
end
c = sparse(loc, ones(num, 1), val, Ainds(end), 1);
c = sparse([prob.c; c]);

% Convert A
subi = [prob.bara.subi, prob.bara.subi];
subj = [prob.bara.subj, prob.bara.subj];
subk = [prob.bara.subk, prob.bara.subl];
subl = [prob.bara.subl, prob.bara.subk];
val = [prob.bara.val, prob.bara.val];
num = length(val);
loc = zeros(num, 1);
for v = 1:num
    loc(v) = Ainds(subj(v)) + (subk(v) - 1)*K.s(subj(v)) + subl(v);
    if subk(v) == subl(v)
        val(v) = val(v)/2;
    end
end
A = sparse(loc, subi, val, Ainds(end), m);
A = sparse([prob.a'; A]);

end