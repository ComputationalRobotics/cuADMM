function tstr = mytimeADMM(ttime)
%MYTIMEAPG Summary of this function goes here
%   Detailed explanation goes here
t = seconds(ttime);
t.Format = 'hh:mm:ss';
tstr = char(t);
end

