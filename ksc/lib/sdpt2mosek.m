function prob = sdpt2mosek(blk, At, C, b)

[s_At,s_b,s_c,s_K] = SDPT3data_SEDUMIdata(blk,At,C,b);


prob = convert_sedumi2mosek(s_At,...
                            s_b,...
                            s_c,...
                            s_K);
end