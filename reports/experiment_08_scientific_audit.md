# Experiment 8 Scientific Audit

## 1. Objective
Perform a rigorous mathematical and scientific audit of the interpretive claims made following Experiment 8 regarding PLD sensitivities, peak-anchoring behavior, redundancy, and causality.

## 2. Data and Physiological Grid
- **Verified Source of Selected PLDs:** The analysis strictly utilized `[1.525, 2.525, 3.025]` as the selected subset and `[2.025, 3.525, 4.025]` as discarded, exactly matching the authoritative Experiment 4 json output. No retroactive tampering occurred.
- **Verified Grid:** The data frame contains exactly 216 rows corresponding to the 6 (CBF) × 6 (ATT) grid evaluated across all 6 baseline PLDs.

## 3. Mathematical Calculations
- Finite difference steps were verified ($\Delta CBF = \pm 1.0$, $\Delta ATT = \pm 0.05$). The resulting derivatives logically scale with analytical expectations (CBF scaling signal linearly; ATT shifting the decay onset).

## 4. Derivative Validation (Sensitivities)
- **Claim:** "PLD 2.525s provides the highest numerical sensitivity to CBF."
  - **Audit Result: INCORRECT / REQUIRES CORRECTION.** 
  - **Evidence:** The raw grid data shows PLD 1.525s holds the maximum absolute CBF sensitivity in 50% of the cases (18/36) and has the highest *average* absolute sensitivity overall (5.56 vs 3.72 for 2.525s). Early signals suffer the least $T_1$ exponential attenuation, preserving the pure CBF scalar.
- **Claim:** "PLD 1.525s provides the highest numerical sensitivity to ATT."
  - **Audit Result: DIRECTLY DEMONSTRATED.**
  - **Evidence:** 1.525s dominates the ATT sensitivity in 30/36 cases, maintaining a mean of 166.49.

## 5. Peak-Location Validation
- **Claim:** "PLD 2.525s anchoring the global kinetic peak."
  - **Audit Result: REQUIRES CAUTIOUS WORDING.**
  - **Evidence:** The exact continuous ASL peak shifts dramatically with ATT. For 18/36 conditions (short ATTs), the peak actually occurs at $\leq 1.5$ seconds. The mean absolute distance to the peak is actually *smaller* for 1.525s (0.66s) than for 2.525s (0.85s). The selected set does not statically map to "Arrival, Peak, Decay"; rather, it is spaced to guarantee at least one measurement is always near the peak regardless of physiological state.

## 6. Selected vs Discarded Comparison
- **Claim:** "Discarded PLDs (3.525s, 4.025s) are redundant."
  - **Audit Result: DIRECTLY DEMONSTRATED.**
  - **Evidence:** The audit correlation matrix (`pld_correlation_matrix.csv`) mathematically proves redundancy. The signal at 3.525s correlates $0.999$ with 4.025s across the grid. They provide mathematically identical decay variance. Discarded PLD 2.025s correlates strongly ($0.963$) with 2.525s. The chosen set `[1.525, 2.525, 3.025]` maximizes orthogonal spacing across the correlation matrix.

## 7. Causality Limitations
- **Claim Check:** "The results strongly suggest that the authoritative 3-PLD configuration minimizes DNN validation error specifically because it samples mutually complementary, non-redundant kinetic information."
  - **Audit Result: SUPPORTED.** 
  - The phrasing correctly identifies the interpretation ("strongly suggest") without falsely claiming the DNN actively computed the partial derivatives. Experiment 4 minimized MAE; Experiment 8 characterizes the physical properties of the chosen subset.

## 8. Experiment 7 → Experiment 8 Scientific Refinement
**Previous Interpretation (Exp 7):** PLD 2.525s is the kinetic peak, and 1.525s is the arrival phase.
**New Evidence (Exp 8):** Peak locations are highly volatile based on ATT. 
**Corrected Interpretation:** The DNN did not select a static "Peak PLD" because a static peak does not exist. It selected a wide, uncorrelated basis set (`1.525, 2.525, 3.025`) capable of triangulating the shifting curve shape across all possible ATTs.

## 9. Final Scientifically Defensible Interpretation
The exhaustive 3-PLD selection identified a subset that mathematically avoids highly correlated (redundant) measurements. By preserving 1.525s, it retains the maximum average sensitivity to both ATT and CBF. By preserving 2.525s and 3.025s, it retains sufficient temporal spacing to bound the shifting kinetic peak and initial decay across the tested physiological domain, while safely discarding the highly correlated ($r=0.999$), low-sensitivity deep $T_1$ tails.

## 10. Reproducibility Information
- **Audit Notebook:** `notebooks/08b_physiological_reference_pld_audit.ipynb`
- **Generated CSVs:** `peak_location_by_condition.csv`, `pld_correlation_matrix.csv`, `region_statistics.csv`.
