function mosek_to_txt(problem, output_dir)
    addpath("./utils")
    problem = column2row_recursive(problem);
    problem.blx = [];
    [A, b, c, K] = convert_mosek2sedumi(problem);
    sedumi.A = A;
    sedumi.b = b;
    sedumi.c = c;
    sedumi.K = K;
    sedumi_to_txt(sedumi, output_dir);
end