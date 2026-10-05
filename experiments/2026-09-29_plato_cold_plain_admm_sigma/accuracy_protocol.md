# Accuracy protocol: PLATO / Kocvara sparse structural-optimization SDPs, cold-start plain ADMM

Sources, downloaded 2026-09-29 18:08–18:11 EDT; copies and SHA-256 sums are in `data/`:

- the PLATO page https://plato.asu.edu/ftp/sparse_sdp.html (H. D. Mittelmann, "Several SDP-codes on sparse and other SDP problems", 25 Apr 2026);
- the DIMACS error definitions it cites, https://plato.asu.edu/dimacs/node3.html;
- the collection page https://web.mat.bham.ac.uk/kocvara/pennon/problems.html (M. Kocvara, 2 Feb 2017);
- the solver logs https://plato.asu.edu/ftp/sparse_logs/.

## 1. Requirements explicitly stated by the PLATO benchmark

The PLATO sparse-SDP page states the following:

- **No stopping tolerance is mandated.** Each code runs with its own stopping rule.
- **Time limit:** "Given are total CPU seconds; maxtime = 40000s" (AMD Ryzen 9 5900X, 12 cores, 128 GB).
- **Error measures:** "For all codes error measures are given as defined in the 7th DIMACS Challenge benchmark paper, Math Prog 95, 407-430 (2003), plato.asu.edu/dimacs/node3.html."
- **Classification**, with no pass/fail threshold other than this:
  - **"a"**: "at least one DIMACS error > 1e-4". The run is still counted as solved, with reduced accuracy.
  - **"f"**: "fail (at least one DIMACS error > 1e-2; counted as 40,000s)".
  - **"m"**: "memory exceeded (counted as 40,000s)".
- **Objectives:** the page publishes no reference objectives.

The Kocvara collection page gives one reference objective per problem, "computed by PENSDP and, in most cases, confirmed by SDPT3 and MOSEK. In the large-scale problems, the two/three codes may differ in the 5th-6th digit. In this case, we give the PENSDP value." These values carry 5 to 8 significant digits, and trto1/2/3/5 are marked "exact value". The page defines no tolerance.

**Summary.** PLATO specifies the DIMACS measures, a 40,000 s CPU limit, and the thresholds 1e-4 ("a") and 1e-2 ("f"). It does not require 1e-4 as a stopping tolerance, and it defines no objective-digit requirement. In particular, 1e-4 is not an official PLATO success criterion. It only marks reduced accuracy.

## 2. Historical accuracy achieved by published solvers

Source: `data/historical_dimacs.csv`, parsed from the PLATO logs by `scripts/parse_mittelmann_logs.py`. The table gives max |DIMACS error| over the six measures, for the 8 Kocvara problems on the PLATO page:

| problem | COPT | CSDP | MOSEK | SDPA | SDPT3 | SeDuMi | HDSDP | cuLoRADS* |
|---|---|---|---|---|---|---|---|---|
| buck5 | 1.2e-3 | 3.9e-6 | 3.7e-6 | 2.1e-4 | 7.5e-5 | 2.0e-5 | 1.0 | 2.1e-5 |
| mater-6 | 8.9e-6 | 1.6e-8 | 3.7e-9 | 3.0e-9 | 2.7e-9 | 3.3e-6 | 5.3e-9 | n/a (fail) |
| shmup4 | 4.5e-6 | 1.7e-5 | 1.4e-7 | 1.0e-5 | 1.3e-7 | 2.5e-8 | 1.2e-3 | 9.0e-6 |
| shmup5 | 2.4e-6 | 5.6e-5 | 4.4e-7 | 7.9e-5 | 5.1e-7 | 0.49 | 3.3e-5 | 4.4e-5 |
| trto4 | 6.1e-6 | 3.0e-4 | 1.2e-5 | 1.5e-3 | 1.3e-4 | 4.7e-8 | 5.7e-6 | 2.5e-3 |
| trto5 | 1.5e-6 | 8.6e-4 | 1.8e-5 | 8.5e-4 | 9.5e-4 | 2.9e-6 | 1.4e-7 | 4.4e-6 |
| vibra4 | 9.5e-5 | 6.0e-6 | 1.0e-6 | 5.6e-4 | 1.7e-5 | 3.3e-9 | 2.5e-5 | 5.6e-4 |
| vibra5 | 2.3e-4 | 1.2e-4 | 1.9e-5 | 8.1e-3 | 3.3e-4 | 1.1e-5 | 0.15 | 3.9e-3 |

\* cuLoRADS reports only err1, err3 and err5.

What the table shows:

- **Reaching 1e-6 is rare.** No solver reaches max |DIMACS| ≤ 1e-6 on every problem. MOSEK exceeds 1e-6 on 5 of the 8, where the worst measure is usually err6 or err3.
- **1e-4 is common but not universal.** Errors of 1e-4 to 1e-2 ("a" on the page) are common for buck, trto and vibra.

## 3. Our cuADMM experimental acceptance criteria (chosen here, not PLATO requirements)

All quantities are recomputed from the returned X, y, S on the original (unscaled) data by the external validator. The problem is in cuADMM's orientation, min ⟨C, X⟩ s.t. A(X) = b, X ∈ K. Here K is a product of PSD blocks and nonnegative orthants ('l' blocks). The SDPA data map to it as C = −F0, A_i = F_i, b = c, and SDPA objective = −⟨C, X⟩ (see `conversion_validation.md`).

### DIMACS measures

These follow node3.html, with z = S (cuADMM maintains S):

- err1 = ‖A(X) − b‖₂ / (1 + ‖b‖∞)
- err2 = max(0, −λmin,K(X)) / (1 + ‖b‖∞)
- err3 = ‖A*(y) + S − C‖_K / (1 + ‖C‖∞)
- err4 = max(0, −λmin,K(S)) / (1 + ‖C‖∞)
- err5 = (⟨C, X⟩ − bᵀy) / (1 + |⟨C, X⟩| + |bᵀy|), signed
- err6 = ⟨X, S⟩ / (1 + |⟨C, X⟩| + |bᵀy|)

Conventions:

- ‖·‖_K is the DIMACS norm on the space of X: the sum over PSD blocks of the Frobenius norms, plus the 2-norm of the nonnegative part.
- ‖·‖∞ is DIMACS's "‖·‖₁", the absolute value of the largest component. For C it is taken over the matrix entries, so the √2 of svec is undone.
- λmin,K is the minimum over PSD-block eigenvalues and over nonnegative components.
- err6 is reported always. DIMACS defines it only when err2 = err4 = 0; the flag `err6_defined` records whether that holds.
- max_abs_DIMACS = max_i |err_i| over all six.
- As a diagnostic, we also report the variant with z = C − A*(y), for which err3 = 0 and err4 is the cone violation of C − A*(y).

### cuADMM normalized measures

- eta_p = ‖A(X) − b‖₂/(1 + ‖b‖₂)
- eta_d = ‖A*(y) + S − C‖₂/(1 + ‖C‖₂)
- eta_g = |⟨C, X⟩ − bᵀy|/(1 + |⟨C, X⟩| + |bᵀy|)
- eta = max(eta_p, eta_d, eta_g)

Normalized cone violations:

- X: max over blocks of max(0, −λmin(X_b))/(1 + ‖X_b‖_F).
- S: max over blocks of max(0, −λmin(S_b))/(1 + ‖C‖₂).
- Z = C − A*(y): the same normalization as S.

### Relative objective error

Relative objective error = |obj − ref|/(1 + |ref|), where obj = −⟨C, X⟩ is the objective in the SDPA convention and ref is Kocvara's published value.

The published values have limited precision (5–8 significant digits), so the error is also reported against the resolution of the reference. A reference given to d digits cannot confirm agreement below half a unit in its last digit.

### The two declared targets

**Practical cuADMM target (PRACTICAL_VALIDATED).** A run meets it when all of the following hold:

- eta ≤ 1e-4;
- max(X, S, Z normalized cone violation) ≤ 1e-4;
- all values are finite.

This is the stopping target: the run stops successfully the first time the external validator confirms it.

**Strict benchmark target (STRICT_VALIDATED).** A run meets it when all of the following hold:

- max_abs_DIMACS ≤ 1e-6;
- relative objective error ≤ 1e-6;
- no material PSD-cone violation, meaning err2 ≤ 1e-6 and err4 ≤ 1e-6, which max_abs_DIMACS already implies;
- all values are finite.

After the practical target is met, the run continues toward the strict target within the same time limit. When the published reference has fewer digits than a 1e-6 check needs, the objective part is reported as "undecidable at the reference precision". The run is then STRICT_VALIDATED only on DIMACS, and the table says so.

### Other statuses

| status | condition |
|---|---|
| NOT_VALIDATED | neither target reached, and the run stopped for another reason: non-finite values, or a crash before the limit |
| TIMEOUT | the practical target not reached within the time limit |
| OUT_OF_MEMORY | host or GPU memory exhausted |
| CONVERSION_FAILED | the instance fails the conversion checks and is not benchmarked |

For TIMEOUT runs:

- the best and the final iterate are both validated and reported;
- time-to-tolerance is reported as "> limit";
- nothing is extrapolated.

### Threshold times

For eta and for max_abs_DIMACS, we record the first validation snapshot at or below 1e-3, 1e-4, 1e-5 and 1e-6, with its iteration and solve-clock time. The solver validates periodically, so these times are exact to the validation interval.

### Relation to PLATO's classification

- Our practical target uses cuADMM's eta (2-norm normalizations), not the DIMACS ∞-norm normalizations, so it is not PLATO's "a" threshold.
- A run can satisfy eta ≤ 1e-4 and still have max_abs_DIMACS > 1e-4, or the reverse.
- Every run therefore reports the PLATO-style classification as well:
  - clean: max_abs_DIMACS ≤ 1e-4;
  - "a": 1e-4 < max_abs_DIMACS ≤ 1e-2;
  - "f": max_abs_DIMACS > 1e-2.

## 4. Reference objectives used for the relative objective error

The references come from `data/reference_objectives.csv`, produced by `scripts/reference_objectives.py`.

**Scale corrections.** The published PENSDP values do not always match the scale of the `.dat-s` files. Independent solutions of the same files show pure powers of ten:

- trto1: ×10³;
- trto2 and trto4–5: ×10⁴;
- buck1: ×10.

The independent solutions are:

- **Clarabel**, an interior-point solver run through CVXPY on the converted data. It was used for trto1, trto2, buck1, buck2, vibra1, vibra2, shmup1, mater-1 and mater-2. Its value agrees with the published one to 6e-8 to 1.5e-6 where no factor is needed. This is also the independent confirmation of the conversion's orientation and sign.
- **MOSEK's primal objective** in Mittelmann's PLATO logs, for the 8 benchmarked instances. MOSEK was not run here.

For trto3 the factor 10⁴ is inferred from its family, because trto2, trto4 and trto5 all need it.

**The reference.** The reference is the published value times that factor.

**Its uncertainty.**

- For "exact value" entries it is 0.
- Otherwise it is the resolution of the published digits.
- Where the PLATO MOSEK solution is itself reliable (max |DIMACS| ≤ 1e-6), the uncertainty is raised to the difference between that MOSEK value and the published one. The Kocvara page notes differences between codes in the 5th–6th digit.

**Consequence.** shmup4 (2.0e-6) and shmup5 (1.8e-6) have a reference uncertainty above 1e-6. For them, the objective part of the strict target cannot be decided at 1e-6. The report then gives STRICT_VALIDATED only on the DIMACS criterion, with the note "objective reference uncertain (x)". All other 24 references are decidable at 1e-6.
