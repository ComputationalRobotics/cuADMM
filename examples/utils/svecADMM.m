function x = svecADMM(blk, X, isspx)
numblk = size(blk, 1);
if (iscell(X))
    if (numblk ~= size(X, 1))
        error('svec: number of rows in blk and X not equal.');
    end
    if (nargin == 2)
        isspx = ones(numblk, 1);
    else
        if (length(isspx) < size(blk, 1))
            isspx = ones(numblk, 1);
        end
    end
    x = cell(numblk, 1);
    for k = 1:numblk
        kblk = blk(k, :);
        n = sum(kblk{2});
        m = size(X, 2);
        if (strcmp(kblk{1}, 's'))
            % symmetric block
            n2 = sum(kblk{2} .* (kblk{2} + 1) / 2);
            if (isspx(k))
                x{k} = sparse(n2, m);
            else
                x{k} = zeros(n2, m);
            end
            if (~isempty(kblk{2}))
                for i = 1:m
                    if (length(kblk{2}) > 1 && ~issparse(X{k, i}))
                        x{k}(:, i) = mexsvec(kblk, sparse(X{k, i}), isspx(k));
                    else
                        x{k}(:, i) = mexsvec(kblk, X{k, i}, isspx(k));
                    end
                end
            end
        else 
            % other blocks
            if (isspx(k))
                x{k} = sparse(n, m);
            else
                x{k} = zeros(n, m);
            end
            for i = 1:m 
                x{k}(:, i) = X{k, i};
            end
        end
    end
else
    if (strcmp(blk{1}, 's'))
        if (length(blk{2}) > 1 && ~issparse(X))
            x = mexsvec(blk, sparse(X), 1);
        else
            x = mexsvec(blk, X);
        end
    else
        x = X;
    end
end

end