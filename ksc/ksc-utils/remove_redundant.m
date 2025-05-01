function SDP = remove_redundant(SDP, pool)
    if isfield(SDP, "sedumi")
        SDP.sedumi.At = SDP.sedumi.At(:, pool);
        SDP.sedumi.b = SDP.sedumi.b(pool);
    end

    for i = 1: size(SDP.sdpt3.blk, 1)
        SDP.sdpt3.At{i} = SDP.sdpt3.At{i}(:, pool);
    end
    SDP.sdpt3.b = SDP.sdpt3.b(pool);
end