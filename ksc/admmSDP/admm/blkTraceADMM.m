function tr = blkTraceADMM(blk, X, Z)
%% matrices are stored in vectorized format by default
tr = 0.0;
for k = 1:size(blk, 1)
    tr = tr + sum(sum(X{k} .* Z{k}));
end
end