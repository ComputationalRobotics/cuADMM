# Phase-4b H200 gates: results

## Runs

| gate | run | status | iterations | solve s | practical it (s) | strict it (s) | changes | tv_log2 | freeze | spikes >= 10x | max spike | E final | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | buck1_fixed_s1 | converged | 119850 | 70.1 | 69440 (40.7) | 119850 (70.1) | 0 | 0 | — | 0 | 0 | 9.98e-07 | 6566 |
| G0 | pendulum_N80_fixed_s1 | time_limit | 302334 | 900 | 230315 (686) | -1 (—) | 0 | 0 | — | 0 | 0 | 3.92e-05 | 7050 |
| G0 | trto1_fixed_s1 | converged | 60150 | 24.5 | 38474 (15.6) | 60150 (24.5) | 0 | 0 | — | 0 | 0 | 9.89e-07 | 6566 |
| G1 | buck1_bf_s1 | converged | 121900 | 67.9 | 72861 (40.8) | 121900 (67.8) | 20 | 3.74 | 191 step | 0 | 7.43 | 9.98e-07 | 6566 |
| G1 | buck1_rb_s1 | time_limit | 1005958 | 600 | -1 (—) | -1 (—) | 1074 | 1074 | — | 0 | 3.05 | 0.0969 | 6566 |
| G1 | mater-1_bf_s1 | converged | 73300 | 23.6 | 42758 (13.7) | 73300 (23.6) | 59 | 10.3 | 1301 step | 0 | 1.95 | 9.98e-07 | 6566 |
| G1 | mater-1_rb_s1 | converged | 16300 | 5.35 | 8850 (2.92) | 16300 (5.35) | 157 | 157 | — | 0 | 2.87 | 9.54e-07 | 6566 |
| G1 | mater-2_bf_s1 | converged | 163100 | 61.3 | 85171 (31.8) | 163100 (61.3) | 68 | 11.4 | 1801 step | 0 | 2.62 | 1e-06 | 6568 |
| G1 | mater-2_rb_s1 | converged | 36650 | 13.5 | 12490 (4.63) | 36650 (13.5) | 163 | 163 | — | 0 | 3.58 | 9.91e-07 | 6568 |
| G1 | shmup1_bf_s1 | time_limit | 1756546 | 1800 | -1 (—) | -1 (—) | 17 | 3.68 | 161 step | 0 | 1.23 | 0.00428 | 6568 |
| G1 | shmup1_rb_s1 | time_limit | 1756714 | 1800 | -1 (—) | -1 (—) | 1781 | 1781 | — | 0 | 4.67 | 0.00145 | 6568 |
| G1 | trto1_bf_s1 | converged | 97750 | 38.5 | 63533 (25.1) | 97750 (38.5) | 16 | 3.94 | 151 step | 0 | 3.81 | 9.76e-07 | 6566 |
| G1 | trto1_rb_s1 | time_limit | 1517982 | 600 | -1 (—) | -1 (—) | 1351 | 1351 | — | 0 | 4.02 | 0.486 | 6566 |
| G1 | vibra1_bf_s1 | converged | 80800 | 47.5 | 38172 (22.5) | 80800 (47.5) | 18 | 3.81 | 171 step | 0 | 6.89 | 9.99e-07 | 6566 |
| G1 | vibra1_rb_s1 | converged | 342300 | 200 | 179920 (105) | 342300 (200) | 472 | 472 | — | 0 | 3.86 | 9.99e-07 | 6566 |
| G1_attempt1_failed | buck1_bf_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | buck1_rb_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | mater-1_bf_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | mater-1_rb_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | mater-2_bf_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | mater-2_rb_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | shmup1_bf_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | shmup1_rb_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | trto1_bf_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | trto1_rb_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | vibra1_bf_s1 | missing | | | | | | | | | | | |
| G1_attempt1_failed | vibra1_rb_s1 | missing | | | | | | | | | | | |
| G2 | pendulum_N80_bf_s1 | time_limit | 299319 | 900 | -1 (—) | -1 (—) | 14 | 3.28 | 131 step | 0 | 0.434 | 0.000132 | 7050 |
| G2 | pendulum_N80_bf_s100 | time_limit | 307630 | 900 | 187198 (548) | -1 (—) | 54 | 11.2 | 1101 step | 0 | 2.45 | 8.73e-05 | 7050 |
| G2 | pendulum_N80_rb_s1 | time_limit | 300729 | 900 | 84390 (252) | -1 (—) | 416 | 416 | — | 0 | 2.09 | 0.000256 | 7050 |
| G2 | pendulum_N80_rb_s100 | time_limit | 304902 | 900 | 96395 (285) | -1 (—) | 421 | 421 | — | 5 | 10.3 | 0.000361 | 7050 |
| G3 | mater-3_bf_s1 | converged | 1167250 | 643 | 153451 (86.1) | 1167250 (643) | 68 | 11.6 | 1801 step | 0 | 2.4 | 1e-06 | 6574 |
| G3 | mater-3_rb_s1 | converged | 188000 | 102 | 26743 (14.4) | 188000 (102) | 324 | 324 | — | 0 | 5.83 | 9.98e-07 | 6574 |
| G3 | mater-4_bf_s1 | time_limit | 1492951 | 1800 | 330496 (398) | -1 (—) | 54 | 10.2 | 1051 step | 0 | 1.43 | 3.08e-06 | 6596 |
| G3 | mater-4_rb_s1 | converged | 652000 | 774 | 31506 (37.2) | 652000 (774) | 770 | 770 | — | 6 | 16.2 | 9.99e-07 | 6596 |
| G3 | mater-5_bf_s1 | time_limit | 809840 | 1800 | 582619 (1290) | -1 (—) | 41 | 9 | 751 step | 0 | 1.66 | 9.31e-06 | 6638 |
| G3 | mater-5_rb_s1 | time_limit | 768801 | 1800 | 61548 (143) | -1 (—) | 862 | 862 | — | 5 | 13.4 | 2e-05 | 6638 |
| G3 | mater-6_bf_s1 | time_limit | 380401 | 1800 | -1 (—) | -1 (—) | 51 | 10.2 | 951 step | 0 | 1.72 | 0.00239 | 6716 |
| G3 | mater-6_rb_s1 | time_limit | 396260 | 1800 | 88414 (403) | -1 (—) | 521 | 521 | — | 5 | 19.7 | 5.09e-05 | 6716 |

## G0 fixed-sigma reproduction

- buck1_fixed_s1: strict stage at 119850 iterations (reference 119850, 68.9 s), objective rel. diff 3.57e-11; iterations identical: yes; within 1%: yes; time to the milestone 70.1 s (solve 70.1 s): **pass**
- pendulum_N80_fixed_s1: practical 1e-4 at 230315 iterations (reference 230315, 622 s solve / 625 s end to end); iterations identical: yes; within 1%: yes; time to the milestone 686 s (solve 900 s): **pass**
- trto1_fixed_s1: strict stage at 60150 iterations (reference 60150, 23.3 s), objective rel. diff 4.49e-10; iterations identical: yes; within 1%: yes; time to the milestone 24.5 s (solve 24.5 s): **pass**

## G1 (sigma_0 = 1)

must-solve set (fixed sigma = 1 or the old rule within the cap): buck1, mater-1, mater-2, trto1, vibra1

| problem | candidate strict s (solved) | residual_balancing strict s (solved) | fixed sigma = 1 s | old rule s |
|---|---|---|---|---|
| trto1 | 38.5 (yes) | 600 (no) | 24.5 | 600 |
| buck1 | 67.8 (yes) | 600 (no) | 70.1 | 600 |
| vibra1 | 47.5 (yes) | 200 (yes) | 556 | 39.5 |
| mater-1 | 23.6 (yes) | 5.35 (yes) | 124 | 32.3 |
| mater-2 | 61.3 (yes) | 13.5 (yes) | 433 | 16.6 |
| shmup1 | 1800 (no) | 1800 (no) | 1800 | 1800 |

- G1a no problem lost: yes (lost: none)
- G1b changes stop (every run frozen or solved): yes; max total variation 11.4
- G1c no spike >= 10x: yes (0)
- G1d shifted geometric mean of the strict time: candidate 89 s, residual_balancing 183 s, fixed 225 s, old 173 s: yes

## G2 pendulum N=80

```
{
 "pendulum_N80_bf_s1": {
  "practical_iter": -1,
  "t_practical": null,
  "strict_iter": -1,
  "t_strict": null,
  "solve_time_s": 900.021038887,
  "status": "time_limit",
  "changes": 14,
  "freeze": [
   131,
   "step"
  ],
  "spikes_ge10": 0
 },
 "pendulum_N80_bf_s100": {
  "practical_iter": 187198,
  "t_practical": 548.323297432,
  "strict_iter": -1,
  "t_strict": null,
  "solve_time_s": 900.019506179,
  "status": "time_limit",
  "changes": 54,
  "freeze": [
   1101,
   "step"
  ],
  "spikes_ge10": 0
 },
 "pendulum_N80_rb_s1": {
  "practical_iter": 84390,
  "t_practical": 251.80186593,
  "strict_iter": -1,
  "t_strict": null,
  "solve_time_s": 900.021436482,
  "status": "time_limit",
  "changes": 416,
  "freeze": [
   -1,
   ""
  ],
  "spikes_ge10": 0
 },
 "pendulum_N80_rb_s100": {
  "practical_iter": 96395,
  "t_practical": 285.267699539,
  "strict_iter": -1,
  "t_strict": null,
  "solve_time_s": 900.020387335,
  "status": "time_limit",
  "changes": 421,
  "freeze": [
   -1,
   ""
  ],
  "spikes_ge10": 5
 },
 "references": {
  "old_rule_s1_end_to_end_s": 124.0,
  "old_rule_practical_iteration": 45393,
  "fixed_s1_solve_s": 622.08,
  "fixed_s1_practical_iteration": 230315
 }
}
```

## G3 mater-3..6

```
{
 "mater-3": {
  "candidate": [
   1167250,
   642.715693297,
   "converged"
  ],
  "residual_balancing": [
   188000,
   101.74942594,
   "converged"
  ],
  "campaign_fixed": [
   -1,
   1800.1
  ],
  "campaign_old": [
   231000,
   88.3
  ],
  "candidate_spikes": 0,
  "candidate_freeze": [
   1801,
   "step"
  ]
 },
 "mater-4": {
  "candidate": [
   -1,
   null,
   "time_limit"
  ],
  "residual_balancing": [
   652000,
   774.142080395,
   "converged"
  ],
  "campaign_fixed": [
   -1,
   1800.3
  ],
  "campaign_old": [
   769500,
   497.9
  ],
  "candidate_spikes": 0,
  "candidate_freeze": [
   1051,
   "step"
  ]
 },
 "mater-5": {
  "candidate": [
   -1,
   null,
   "time_limit"
  ],
  "residual_balancing": [
   -1,
   null,
   "time_limit"
  ],
  "campaign_fixed": [
   -1,
   1800.4
  ],
  "campaign_old": [
   890000,
   1039.1
  ],
  "candidate_spikes": 0,
  "candidate_freeze": [
   751,
   "step"
  ]
 },
 "mater-6": {
  "candidate": [
   -1,
   null,
   "time_limit"
  ],
  "residual_balancing": [
   -1,
   null,
   "time_limit"
  ],
  "campaign_fixed": [
   -1,
   1800.6
  ],
  "campaign_old": [
   -1,
   1800.6
  ],
  "candidate_spikes": 0,
  "candidate_freeze": [
   951,
   "step"
  ]
 }
}
```

## H200 usage

| job | name | state | elapsed | limit | node |
|---|---|---|---|---|---|
| 54683287 | p4b-G0 | COMPLETED | 0.28 h | 00:45:00 | holygpu8a12201 |
| 54688317 | p4b-G1 | COMPLETED | 0.01 h | 05:00:00 | holygpu8a12204 |
| 54688318 | p4b-G2 | COMPLETED | 1.01 h | 01:10:00 | holygpu8a12201 |
| 54688319 | p4b-G3 | COMPLETED | 2.94 h | 04:20:00 | holygpu8a12204 |
| 54689882 | p4b-G1 | COMPLETED | 1.49 h | 05:00:00 | holygpu8a12603 |

total allocated: 5.73 H200-hours (approved: at most 12.58)
