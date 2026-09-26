# Permanent Experiment History

This document serves as the chronological research record for the project, tracking the evolution of methodology, bug fixes, and scientific findings.

---

### Experiment 1: Initial Baseline (Pre-Audit)
* **Research question:** Can a DNN estimate CBF and ATT using a heuristic 3-PLD subset (`[0,1,3]`) compared to a full 6-PLD acquisition?
* **Notebook:** `notebooks/archive/pre_audit/compare_10000_samples.ipynb`
* **Configuration:** 4,000 training samples. SNR=10. `[0,1,3]` heuristic selection.
* **Result:** Demonstrated feasibility but showed a ~20% CBF MAE degradation for 3-PLD.
* **Status:** **SUPERSEDED.** An audit discovered the `paper_signal` simulation divided by $\lambda_{blood}$ twice, unphysically scaling the signal.

### Experiment 2: Corrected Clean Baseline
* **Research question:** Can the DNN learn the corrected Buxton kinetic model (single $\lambda$)?
* **Notebook:** `notebooks/02_clean_baselines.ipynb` (formerly `experiment_clean.ipynb`)
* **Configuration:** 4,000 training, 10,000 test.
* **Result:** Successfully trained on corrected math, but the 4k sample size was suspected to be underfitting the DNN capacity.
* **Status:** **SUPERSEDED** by larger training sizes.

### Experiment 3: Training Size Convergence
* **Research question:** Is the observed 3-PLD performance degradation caused by insufficient training data?
* **Notebook:** `notebooks/05_training_size_convergence.ipynb`
* **Configuration:** Sweep from 4k to 1M samples. Same corrected architecture and validation sets.
* **Result:** 3-PLD plateaus at ~50k samples. The performance gap actually *widens* with more data, proving a fundamental information-theoretic bottleneck. 100,000 samples was identified as the primary operational scale.
* **Status:** **AUTHORITATIVE.**

### Experiment 4: Exhaustive 3-PLD Selection
* **Research question:** Which of the 20 possible 3-PLD combinations yields the lowest Validation CBF MAE? Was the historical `[0,1,3]` choice optimal?
* **Notebook:** `notebooks/04_exhaustive_3pld_selection.ipynb`
* **Configuration:** 100,000 samples. Evaluated all $C(6,3)$ combinations. Test set strictly walled-off.
* **Result:** The historical `[0,1,3]` was NOT optimal (ranked 2nd). The mathematically verified selection is **`[1.525, 2.525, 3.025]`** (indices `[0,2,3]`).
* **Status:** **AUTHORITATIVE.**

### Experiment 5: Final 3-PLD vs 6-PLD Test
* **Research question:** What is the true degradation penalty when using the strictly selected `[0,2,3]` PLDs?
* **Notebook:** `notebooks/04_exhaustive_3pld_selection.ipynb` (Conducted at the end).
* **Result:** CBF MAE degradation was reduced from ~20% (heuristic) to ~13.0% on the untouched test set.
* **Status:** **AUTHORITATIVE.**

### Experiment 6: SNR Robustness
* **Research question:** Does the `[0,2,3]` configuration generalize across noise conditions without retraining?
* **Notebook:** `notebooks/06_snr_robustness.ipynb`
* **Correction Note:** Discovered a scalar SNR bug that randomized noise. Fixed by passing numpy arrays.
* **Result:** The relative CBF MAE degradation remains remarkably stable (~9-12%) across SNR 10 to 80.
* **Status:** **AUTHORITATIVE.**

### Experiment 7: Signal Curve Analysis
* **Research question:** Where on the analytical ASL signal curve are the selected 3 PLDs located, and what information do they retain?
* **Notebook:** `notebooks/07_selected_pld_signal_curve_analysis.ipynb`
* **Result:** Visually and quantitatively demonstrated that the selected PLDs capture the arrival slope (1.525s), kinetic peak (2.525s), and initial decay (3.025s), discarding redundant deep-decay tails.
* **Status:** **AUTHORITATIVE.**

### Experiment 8: Physiological Reference Analysis
* **Research question:** Do the selected 3 PLDs sample complementary signal regions consistently across different physiological CBF and ATT conditions? How sensitive is each PLD locally?
* **Notebook:** `notebooks/08_physiological_reference_pld_analysis.ipynb`
* **Status:** **CURRENT / AUTHORITATIVE.**
