function arr_out = remove_close_zeros(arr_in, tolerance)
    arr_out = arr_in(abs(arr_in) < tolerance);
end