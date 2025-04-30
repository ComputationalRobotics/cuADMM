function [cuda_At, cuda_b, cuda_C, cuda_blk] = data_sdpt3_to_admmSDPcuda(sdpt3)
    cuda_At = from_cell_to_array(sdpt3.At);
    cuda_C = from_cell_to_array(svecADMM(sdpt3.blk, sdpt3.C));
    cuda_b = sdpt3.b;
    cuda_blk = zeros(size(sdpt3.blk, 1), 1);
    for i = 1: size(sdpt3.blk, 1)
        cuda_blk(i) = sdpt3.blk{i, 2};
    end
end