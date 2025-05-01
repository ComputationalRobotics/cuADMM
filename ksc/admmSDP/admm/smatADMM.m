function X = smatADMM(blk, x, isspX)
numblk = size(blk, 1);
if (nargin < 3)
    isspX = zeros(numblk, 1);
end
if (iscell(x))
    X = cell(numblk, 1);
    if (length(isspX) == 1)
        isspX = isspX * ones(numblk, 1);
    end
    for k = 1:numblk
        kblk = blk(k, :);
        if (strcmp(kblk{1}, 's'))
            X{k} = mexsmat(kblk, x{k}, isspX(k));
        else
            X{k} = x{k};
        end
    end
else
    if (strcmp(blk{1}, 's'))
        X = mexsmat(blk, x, isspX);
    else
        X = x;
    end
end