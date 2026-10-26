# max-cut: MATLAB scripts that generate max-cut SDPs

`ctc.m`, `genMAXCUT.m`, `run_maxcut.m`, `split.m`, `symbasis.m`, `treeDecomp.m`, `Ybus_39.mat` and `Ybus_118.mat`
come from Richard Y. Zhang's repository `dual_ctc` (https://github.com/ryz-codes/dual_ctc, commit
7bf45c68ac941c4667d7de54343770a8cd47c930; `split.m`, `symbasis.m` and `treeDecomp.m` from its `private/`
directory). `ctc.m` and `split.m` are identical to the originals; `genMAXCUT.m` and `run_maxcut.m` carry small local
edits. They are distributed under that repository's BSD 2-Clause license, reproduced verbatim in `LICENCE` (the
file the headers of the scripts refer to; fetched on 2026-10-26). `simple_graph.m` is a local example graph.

`run_maxcut.m` builds the SDP relaxation of the max-cut problem of a power-network graph in SeDuMi form
(`A2`, `b2`, `c2`, `K2`); save it to a `.mat` file and read it with `cuadmm_exe` or `cuadmm-convert`.
