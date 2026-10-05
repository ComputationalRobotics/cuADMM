| Dataset | Sigma policy | Initial sigma | Final sigma | Sigma changes | Iterations | Time to eta 1e-4 | Time to DIMACS 1e-6 | End-to-end time | eta | Max DIMACS | Objective error | X cone violation | S cone violation | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trto1 | adaptive (legacy) | 1 | 6.25 | 5206 | 4902916 | > 1800 s | > 1800 s | 1801.3 s | 1.6e-01 | 5.7e-01 | 4.8e-01 | 6.9e-09 | 4.3e-15 | TIMEOUT |
| trto1 | fixed | 1 | 1 | 0 | 60150 | 15.0 s | 23.4 s | 24.2 s (median of 3) | 6.8e-07 | 9.9e-07 | 2.5e-07 | 1.8e-16 | 0.0e+00 | STRICT_VALIDATED |
| buck1 | adaptive (legacy) | 1 | 100 | 5652 | 3582474 | > 1800 s | > 1800 s | 1801.1 s | 6.1e-02 | 1.1e-01 | 4.8e-02 | 3.5e-10 | 7.5e-16 | TIMEOUT |
| buck1 | fixed | 1 | 1 | 0 | 119850 | 40.0 s | 68.9 s | 69.7 s (median of 3) | 2.9e-07 | 1.0e-06 | 9.6e-07 | 1.6e-15 | 2.5e-15 | STRICT_VALIDATED |
| vibra1 | adaptive (legacy) | 1 | 6.25 | 206 | 68650 | 30.5 s | 39.6 s | 40.6 s (median of 3) | 8.4e-07 | 9.2e-07 | 1.4e-08 | 4.5e-16 | 6.6e-17 | STRICT_VALIDATED |
| vibra1 | fixed | 1 | 1 | 0 | 967500 | 275.2 s | 557.0 s | 557.7 s (median of 3) | 1.0e-06 | 1.0e-06 | 5.7e-08 | 2.5e-16 | 1.1e-17 | STRICT_VALIDATED |
| mater-1 | adaptive (legacy) | 1 | 0.04 | 236 | 99850 | 2.6 s | 32.3 s | 33.2 s (median of 3) | 6.2e-07 | 8.6e-07 | 1.6e-06 | 3.6e-15 | 1.1e-14 | PRACTICAL_VALIDATED |
| mater-1 | fixed | 1 | 1 | 0 | 389800 | 74.2 s | 123.1 s | 124.0 s (median of 3) | 1.0e-06 | 1.0e-06 | 1.7e-06 | 5.4e-14 | 1.5e-14 | PRACTICAL_VALIDATED |
| shmup1 | adaptive (legacy) | 1 | 50 | 2353 | 1820426 | 1707.0 s | > 1800 s | 1801.2 s | 7.8e-05 | 7.8e-05 | 5.7e-08 | 3.2e-15 | 3.3e-18 | PRACTICAL_VALIDATED |
| shmup1 | fixed | 1 | 1 | 0 | 1762882 | > 1800 s | > 1800 s | 1801.2 s | 4.6e-02 | 4.6e-02 | 2.3e-02 | 4.3e-15 | 0.0e+00 | TIMEOUT |
| trto2 | adaptive (legacy) | 1 | 6.25 | 1321 | 1181927 | > 1800 s | > 1800 s | 1801.1 s | 1.2e-01 | 7.5e-01 | 9.9e-01 | 3.9e-12 | 2.7e-14 | TIMEOUT |
| trto2 | fixed | 1 | 1 | 0 | 1176209 | > 1800 s | > 1800 s | 1800.9 s | 2.6e-02 | 1.7e-01 | 5.4e-02 | 5.9e-15 | 9.5e-15 | TIMEOUT |
| mater-2 | adaptive (legacy) | 1 | 0.08 | 186 | 48400 | 3.4 s | 16.5 s | 17.6 s (median of 3) | 4.2e-07 | 8.3e-07 | 1.6e-06 | 5.1e-14 | 3.1e-14 | PRACTICAL_VALIDATED |
| mater-2 | fixed | 1 | 1 | 0 | 1244200 | 235.0 s | 428.9 s | 429.6 s (median of 3) | 7.5e-07 | 1.0e-06 | 9.8e-07 | 1.0e-12 | 4.5e-14 | STRICT_VALIDATED |
| buck2 | adaptive (legacy) | 1 | 50 | 1023 | 907962 | > 1800 s | > 1800 s | 1801.1 s | 1.0e-01 | 6.6e-01 | 7.4e-01 | 4.2e-06 | 4.4e-15 | TIMEOUT |
| buck2 | fixed | 1 | 1 | 0 | 891350 | > 1800 s | > 1800 s | 1801.2 s | 2.0e-04 | 3.4e-04 | 1.1e-04 | 1.7e-15 | 0.0e+00 | TIMEOUT |
| vibra2 | adaptive (legacy) | 1 | 100 | 1324 | 907431 | > 1800 s | > 1800 s | 1800.8 s | 8.3e-02 | 5.4e-01 | 9.0e-01 | 1.8e-09 | 1.5e-15 | TIMEOUT |
| vibra2 | fixed | 1 | 1 | 0 | 908190 | > 1800 s | > 1800 s | 1800.7 s | 8.0e-04 | 1.2e-03 | 2.2e-01 | 7.5e-13 | 1.2e-16 | TIMEOUT |
| mater-3 | adaptive (legacy) | 1 | 0.01 | 373 | 231000 | 3.7 s | 88.1 s | 89.1 s (median of 3) | 2.6e-07 | 9.9e-07 | 3.5e-07 | 5.9e-15 | 4.0e-14 | STRICT_VALIDATED |
| mater-3 | fixed | 1 | 1 | 0 | 4727643 | 595.7 s | > 1800 s | 1801.4 s | 3.0e-06 | 4.1e-06 | 8.8e-07 | 7.4e-13 | 5.4e-14 | PRACTICAL_VALIDATED |
| trto3 | adaptive (legacy) | 1 | 0.390625 | 639 | 521375 | > 1800 s | > 1800 s | 1801.2 s | 1.3e-01 | 1.6e+00 | 3.1e-01 | 3.3e-13 | 9.9e-15 | TIMEOUT |
| trto3 | fixed | 1 | 1 | 0 | 521616 | > 1800 s | > 1800 s | 1800.8 s | 5.2e-01 | 5.2e-01 | 7.1e-01 | 4.7e-14 | 2.4e-14 | TIMEOUT |
| mater-4 | adaptive (legacy) | 1 | 0.005 | 901 | 769500 | 25.6 s | 505.5 s | 506.6 s (median of 3) | 1.2e-07 | 6.6e-07 | 9.9e-08 | 4.0e-15 | 4.2e-14 | STRICT_VALIDATED |
| mater-4 | fixed | 1 | 1 | 0 | 2741433 | > 1800 s | > 1800 s | 1801.4 s | 1.6e-03 | 1.6e-03 | 3.3e-03 | 6.8e-13 | 5.3e-14 | TIMEOUT |
| buck3 | adaptive (legacy) | 1 | 0.0244141 | 548 | 363375 | > 1800 s | > 1800 s | 1801.4 s | 1.2e-01 | 5.2e-01 | 9.9e-01 | 7.0e-09 | 1.5e-14 | TIMEOUT |
| buck3 | fixed | 1 | 1 | 0 | 364164 | > 1800 s | > 1800 s | 1800.9 s | 9.8e-02 | 9.8e-02 | 3.1e-01 | 1.6e-13 | 1.1e-15 | TIMEOUT |
| vibra3 | adaptive (legacy) | 1 | 25 | 595 | 365924 | > 1800 s | > 1800 s | 1801.2 s | 7.6e-02 | 1.2e-01 | 9.6e-01 | 3.3e-09 | 1.3e-15 | TIMEOUT |
| vibra3 | fixed | 1 | 1 | 0 | 362386 | > 1800 s | > 1800 s | 1801.3 s | 1.1e-03 | 1.3e-02 | 2.5e-01 | 4.4e-14 | 4.2e-16 | TIMEOUT |
| mater-5 | adaptive (legacy) | 1 | 0.005 | 1012 | 890000 | 48.0 s | 1039.1 s | 1040.6 s | 2.2e-07 | 8.2e-07 | 2.4e-07 | 3.4e-15 | 5.1e-14 | STRICT_VALIDATED |
| mater-5 | fixed | 1 | 1 | 0 | 1557925 | > 1800 s | > 1800 s | 1801.5 s | 3.5e-02 | 3.5e-02 | 7.3e-02 | 6.5e-13 | 2.7e-14 | TIMEOUT |
| shmup2 | adaptive (legacy) | 1 | 0.04 | 587 | 244662 | > 1800 s | > 1800 s | 1801.2 s | 3.7e-03 | 2.8e-02 | 4.4e-01 | 1.4e-14 | 2.0e-19 | TIMEOUT |
| shmup2 | fixed | 1 | 1 | 0 | 246853 | > 1800 s | > 1800 s | 1801.3 s | 3.4e-01 | 3.4e-01 | 9.2e-01 | 4.9e-12 | 5.8e-17 | TIMEOUT |
| trto4 | adaptive (legacy) | 1 | 0.0488281 | 345 | 231000 | > 1800 s | > 1800 s | 1801.5 s | 1.5e-01 | 2.6e+00 | 9.7e-01 | 1.5e-05 | 1.4e-13 | TIMEOUT |
| trto4 | fixed | 1 | 1 | 0 | 231182 | > 1800 s | > 1800 s | 1801.5 s | 5.2e-01 | 2.0e+00 | 1.0e+00 | 1.5e-06 | 9.8e-16 | TIMEOUT |
| mater-6 | adaptive (legacy) | 1 | 0.00125 | 950 | 827123 | 188.6 s | > 1800 s | 1801.8 s | 4.4e-07 | 6.6e-06 | 1.1e-08 | 1.6e-15 | 5.7e-14 | PRACTICAL_VALIDATED |
| mater-6 | fixed | 1 | 1 | 0 | 817549 | > 1800 s | > 1800 s | 1802.3 s | 1.1e-01 | 1.1e-01 | 2.5e-01 | 4.7e-13 | 2.9e-14 | TIMEOUT |
| buck4 | adaptive (legacy) | 1 | 3.125 | 278 | 126034 | > 1800 s | > 1800 s | 1801.8 s | 1.2e-01 | 2.1e+00 | 9.8e-01 | 2.9e-07 | 6.8e-15 | TIMEOUT |
| buck4 | fixed | 1 | 1 | 0 | 125401 | > 1800 s | > 1800 s | 1801.6 s | 3.9e-01 | 3.9e-01 | 8.6e-01 | 5.6e-11 | 2.5e-14 | TIMEOUT |
| vibra4 | adaptive (legacy) | 1 | 0.00610352 | 270 | 127266 | > 1800 s | > 1800 s | 1801.8 s | 1.0e-01 | 4.3e-01 | 7.7e-01 | 2.8e-08 | 1.9e-14 | TIMEOUT |
| vibra4 | fixed | 1 | 1 | 0 | 127863 | > 1800 s | > 1800 s | 1801.7 s | 3.4e-01 | 3.4e-01 | 8.7e-01 | 7.6e-08 | 4.1e-15 | TIMEOUT |
| shmup3 | adaptive (legacy) | 1 | 100 | 293 | 91113 | > 1800 s | > 1800 s | 1802.2 s | 5.0e-02 | 5.0e-02 | 9.8e-01 | 1.5e-09 | 4.1e-17 | TIMEOUT |
| shmup3 | fixed | 1 | 1 | 0 | 90624 | > 1800 s | > 1800 s | 1802.2 s | 3.3e-01 | 3.3e-01 | 9.7e-01 | 1.2e-11 | 2.5e-17 | TIMEOUT |
| trto5 | adaptive (legacy) | 1 | 0.0488281 | 196 | 72958 | > 1800 s | > 1800 s | 1803.5 s | 1.7e-01 | 4.8e+00 | 9.7e-01 | 5.9e-11 | 2.2e-13 | TIMEOUT |
| trto5 | fixed | 1 | 1 | 0 | 74001 | > 1800 s | > 1800 s | 1804.1 s | 4.9e-01 | 1.7e+00 | 1.0e+00 | 2.5e-08 | 2.1e-15 | TIMEOUT |
| shmup4 | adaptive (legacy) | 1 | 100 | 217 | 51773 | > 1800 s | > 1800 s | 1805.0 s | 2.0e-01 | 2.0e-01 | 9.9e-01 (ref. uncertain) | 1.2e-09 | 1.4e-17 | TIMEOUT |
| shmup4 | fixed | 1 | 1 | 0 | 59175 | > 1800 s | > 1800 s | 1805.3 s | 3.7e-01 | 3.7e-01 | 9.8e-01 (ref. uncertain) | 6.2e-12 | 1.3e-17 | TIMEOUT |
| buck5 | adaptive (legacy) | 1 | 0.01 | 179 | 37497 | > 1800 s | > 1800 s | 1805.4 s | 1.9e-01 | 1.6e+00 | 9.9e-01 | 3.2e-05 | 4.1e-15 | TIMEOUT |
| buck5 | fixed | 1 | 1 | 0 | 37569 | > 1800 s | > 1800 s | 1805.7 s | 3.4e-01 | 9.5e-01 | 9.8e-01 | 2.2e-09 | 3.5e-15 | TIMEOUT |
| vibra5 | adaptive (legacy) | 1 | 0.02 | 189 | 38091 | > 1800 s | > 1800 s | 1805.9 s | 2.4e-01 | 6.7e-01 | 9.7e-01 | 5.2e-06 | 4.0e-15 | TIMEOUT |
| vibra5 | fixed | 1 | 1 | 0 | 38250 | > 1800 s | > 1800 s | 1805.6 s | 3.1e-01 | 4.4e-01 | 9.2e-01 | 7.0e-08 | 6.6e-15 | TIMEOUT |
| shmup5 | adaptive (legacy) | 1 | 50 | 151 | 18613 | > 1800 s | > 1800 s | 1822.1 s | 3.1e-01 | 3.1e-01 | 9.7e-01 (ref. uncertain) | 5.3e-09 | 5.3e-17 | TIMEOUT |
| shmup5 | fixed | 1 | 1 | 0 | 54504 | > 1800 s | > 1800 s | 1821.8 s | 3.7e-01 | 3.7e-01 | 9.9e-01 (ref. uncertain) | 4.5e-12 | 1.9e-17 | TIMEOUT |
