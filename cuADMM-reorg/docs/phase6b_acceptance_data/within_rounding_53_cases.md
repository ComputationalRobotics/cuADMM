cases: 53
pobj/dobj within 1e-6 (relative) at every iteration: 47; within 1e-8: 47
eta_p/eta_d/eta_g within 1e-8 (absolute) at every iteration: 48; within 1e-10: 47
sigma logs identical: 49 of 53 with logs

| kind | case | iterations | max rel. diff pobj/dobj | max abs. diff eta | first iteration above 1e-8 | sigma log |
|---|---|---|---|---|---|---|
| legacy_adaptive | trto1__legacy_s1_tol1e-4 | 20000 | 3.5e+03 | 9.3e-09 | 13921 | 155/155 changes, same |
| fixed | Large_LowRankX__fixed_s100_i800 | 800 | 2.7e-03 | 1.2e-03 | 3 | 0/0 changes, same |
| fixed | Large_LowRankX__fixed_s100_tol1e-6 | 3000 | 2.7e-03 | 1.2e-03 | 3 | 0/0 changes, same |
| residual_balancing | Large_LowRankX__balancing_s100_i800 | 800 | 2.7e-03 | 1.2e-03 | 3 | 25/25 changes, same |
| legacy_adaptive | Large_LowRankX__legacy_s100_i800 | 800 | 5.0e-04 | 2.3e-04 | 3 | 42/42 changes, same |
| sgs | Large_LowRankX__sgs_fixed_s100_i500 | 500 | 8.1e-05 | 2.5e-05 | 3 | 0/0 changes, same |
| sgs | Medium_Free__sgs_fixed_s100_i1000 | 1000 | 9.4e-11 | 5.2e-12 | None | 0/0 changes, same |
| fixed | Large_LowRankS__fixed_s0.0001_i800 | 800 | 5.1e-11 | 1.2e-11 | None | 0/0 changes, same |
| fixed | PSD_Free__fixed_s100_i1500 | 1500 | 2.1e-11 | 3.2e-13 | None | 0/0 changes, same |
| fixed | PSD_Free__fixed_s100_tol1e-6 | 1868 | 2.1e-11 | 3.2e-13 | None | 0/0 changes, same |
| residual_balancing | PSD_Free__balancing_s100_i6000 | 6000 | 2.1e-11 | 1.4e-13 | None | 135/135 changes, same |
| residual_balancing | PSD_Free__balancing_s100_tol1e-6 | 109 | 2.1e-11 | 1.4e-13 | None | 10/10 changes, same |
| fixed | Medium_Free__fixed_s100_i1500 | 1500 | 1.9e-11 | 8.1e-13 | None | 0/0 changes, same |
| fixed | Medium_Free__fixed_s100_tol1e-6 | 1316 | 1.9e-11 | 8.1e-13 | None | 0/0 changes, same |
| residual_balancing | Medium_Free__balancing_s100_i3000 | 3000 | 1.9e-11 | 8.1e-13 | None | 90/90 changes, same |
| legacy_adaptive | Medium_Free__legacy_s100_i3000 | 3000 | 1.4e-11 | 7.4e-13 | None | 92/92 changes, same |
| sgs | Nonneg_Free_PSD__hybrid100_fixed_s100_i600 | 600 | 8.3e-12 | 4.9e-13 | None | 0/0 changes, same |
| sgs | Nonneg_Free_PSD__sgs_fixed_s100_i1500 | 1500 | 8.3e-12 | 1.1e-12 | None | 0/0 changes, same |
| fixed | Large_HalfRank__fixed_s100_i800 | 800 | 6.4e-12 | 1.5e-12 | None | 0/0 changes, same |
| sgs | PSD_Nonneg__sgs_fixed_s100_i1500 | 1500 | 5.0e-12 | 1.1e-12 | None | 0/0 changes, same |
| sgs | PSD_Free__hybrid300_balancing_s100_i1500 | 1500 | 3.4e-12 | 2.5e-13 | None | 44/44 changes, same |
| sgs | PSD_Free__sgs_balancing_s100_i1500 | 1500 | 3.4e-12 | 2.5e-13 | None | 20/20 changes, same |
| sgs | PSD_Free__sgs_fixed_s100_i3000 | 3000 | 2.4e-12 | 9.0e-13 | None | 0/0 changes, same |
| sgs | PSD_Free__sgs_fixed_s100_tol1e-5 | 1121 | 2.4e-12 | 3.6e-13 | None | 0/0 changes, same |
| fixed | Nonneg_Free_PSD__fixed_s100_tol1e-6 | 3301 | 1.3e-12 | 6.0e-13 | None | 0/0 changes, same |
| fixed | Nonneg_Free_PSD__fixed_s100_i1500 | 1500 | 1.2e-12 | 5.6e-13 | None | 0/0 changes, same |
| residual_balancing | Nonneg_Free_PSD__balancing_s100_i6000 | 6000 | 1.0e-12 | 2.6e-13 | None | 119/125 changes, first different change #32 (iterations 676 vs 651) |
| fixed | Free_Free_PSD__fixed_s100_i1500 | 1500 | 9.9e-13 | 1.7e-13 | None | 0/0 changes, same |
| fixed | PSD_Nonneg__fixed_s100_i1500 | 1500 | 8.1e-13 | 3.9e-13 | None | 0/0 changes, same |
| fixed | SmallBatch_Free__fixed_s100_i1500 | 1500 | 7.6e-13 | 3.6e-13 | None | 0/0 changes, same |
| residual_balancing | trto1__balancing_s1_tol1e-4 | 20000 | 5.8e-13 | 1.7e-11 | None | 152/152 changes, same |
| legacy_adaptive | PSD_Free__legacy_s100_i6000 | 6000 | 5.5e-13 | 6.0e-14 | None | 136/136 changes, same |
| legacy_adaptive | PSD_Free__legacy_s100_tol1e-6 | 99 | 5.5e-13 | 6.0e-14 | None | 10/10 changes, same |
| fixed | Free_PSD__fixed_s100_i1500 | 1500 | 5.0e-13 | 2.4e-13 | None | 0/0 changes, same |
| fixed | PSD__fixed_s100_i1500 | 1500 | 4.4e-13 | 2.1e-13 | None | 0/0 changes, same |
| residual_balancing | PSD_Nonneg__balancing_s100_i2000 | 2000 | 4.1e-13 | 1.8e-13 | None | 71/58 changes, first different change #32 (iterations 501 vs 576) |
| fixed | PSD_Free_PSD__fixed_s100_i1500 | 1500 | 2.7e-13 | 1.3e-13 | None | 0/0 changes, same |
| sgs | trto1__sgs_fixed_s1_i2000 | 2000 | 2.4e-13 | 5.3e-13 | None | 0/0 changes, same |
| legacy_adaptive | PSD_Nonneg__legacy_s100_i2000 | 2000 | 2.3e-13 | 1.0e-13 | None | 67/64 changes, first different change #28 (iterations 476 vs 451) |
| residual_balancing | trto1__balancing_s1_i6000 | 6000 | 1.9e-13 | 1.7e-12 | None | 128/128 changes, same |
| legacy_adaptive | Nonneg_Free_PSD__legacy_s100_i6000 | 6000 | 1.8e-13 | 8.6e-14 | None | 111/117 changes, first different change #28 (iterations 426 vs 476) |
| fixed | trto1__fixed_s1_tol1e-4 | 20000 | 1.0e-13 | 7.5e-12 | None | 0/0 changes, same |
| legacy_adaptive | trto1__legacy_s1_i6000 | 6000 | 9.0e-14 | 4.7e-12 | None | 123/123 changes, same |
| fixed | trto1__fixed_s1_i3000 | 3000 | 2.9e-14 | 7.1e-13 | None | 0/0 changes, same |
| legacy_adaptive | mater-1__legacy_s1_tol1e-4 | 8117 | 2.3e-14 | 3.7e-12 | None | 141/141 changes, same |
| legacy_adaptive | mater-1__legacy_s1_i6000 | 6000 | 1.9e-14 | 3.3e-12 | None | 128/128 changes, same |
| residual_balancing | mater-1__balancing_s1_i6000 | 6000 | 1.7e-14 | 1.7e-12 | None | 130/130 changes, same |
| residual_balancing | mater-1__balancing_s1_tol1e-4 | 8850 | 1.7e-14 | 3.1e-12 | None | 145/145 changes, same |
| fixed | mater-1__fixed_s1_i3000 | 3000 | 1.6e-14 | 2.1e-12 | None | 0/0 changes, same |
| fixed | mater-1__fixed_s1_tol1e-4 | 20000 | 1.6e-14 | 5.5e-12 | None | 0/0 changes, same |
| sgs | mater-1__sgs_fixed_s1_i2000 | 2000 | 7.3e-15 | 2.0e-12 | None | 0/0 changes, same |
| fixed | Free__fixed_s100_i1500 | 1500 | 0.0e+00 | 0.0e+00 | None | 0/0 changes, same |
| sgs | Free__sgs_fixed_s100_i1500 | 1500 | 0.0e+00 | 0.0e+00 | None | 0/0 changes, same |
