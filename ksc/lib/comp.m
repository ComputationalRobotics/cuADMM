function sign = comp(a, b, n)
    sa = sum(a);
    sb = sum(b);
    if sa < sb
        sign = -1;
        return
    elseif sa > sb
        sign = 1;
        return
    end
    i = n;
    while i >= 1
          if a(i) < b(i)
             sign = -1;
             return
          elseif a(i) > b(i)
             sign = 1;
             return
          else
             i = i - 1;
          end
    end
    sign = 0;
end