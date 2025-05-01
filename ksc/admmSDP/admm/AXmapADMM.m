function AX = AXmapADMM(blk, At, X)
%% matrices are stored in vectorized format by default
m = size(At{1,1}, 2);
AX = zeros(m, 1);
for k = 1:size(blk, 1)
    Atk = At{k, 1};
    if (~isempty(Atk))
        AX = AX + (X{k}' * Atk)';
    end
end