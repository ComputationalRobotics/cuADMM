function locb = bfind(A, l, a, n)
    low = 1;
    high = l;
    while low <= high
        mid = ceil(1/2*(low+high));
        order = comp(A(:,mid), a, n);
        if order == 0
            locb = mid;
            return
        elseif order < 0
           low = mid + 1;
        else
           high = mid - 1;
        end
    end
    locb = 0;
    return
end