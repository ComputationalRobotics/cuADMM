function r = computeMaxRank(blk, rrank)
numblk = size(blk, 1);
r = 0;
for k = 1:numblk
    kblk = blk(k, :);
    if (strcmp(kblk{1}, 's'))
        r = max(r, rrank(k));
    end
end
end