function Aty = AtymapADMM(blk, At, y)
%% matrices are stored in vectorized format by default
numblk = size(blk, 1);
Aty = cell(numblk, 1);
for p = 1:numblk
    Atp = At{p, 1};
    if (~isempty(Atp))
        Aty{p} = Atp * y;
    end
end
end