function deg = get_deg_from_so2(c, s)
    deg_c = acos(c);
    deg_s = asin(s);
    if deg_s >= 0
        deg = deg_c;
    else
        deg = -deg_c;
    end
end