# Experiment Registry

This registry tracks all completed, authoritative, and superseded experiments in the repository to ensure no historical research is lost and to distinguish current authoritative methodology from older heuristic attempts.

| Exp | Name | Research Question | Notebook | Status |
|---|---|---|---|---|
| **1** | **Initial Baseline (Heuristic)** | Can a DNN estimate CBF/ATT using heuristic 3-PLD vs 6-PLD? | `archive/pre_audit/compare_10000_samples.ipynb` | Historical / Superseded (Physics Bug) |
| **2** | **Clean Baseline** | Can the DNN learn the corrected physical Buxton model? | `notebooks/02_clean_baselines.ipynb` | Historical (Data Size too small) |
| **3** | **Training Size Convergence** | Is the 3-PLD degradation caused by insufficient training data? | `notebooks/05_training_size_convergence.ipynb` | **Authoritative** |
| **4** | **Exhaustive 3-PLD Selection** | Which of the 20 possible 3-PLD combinations performs best? | `notebooks/04_exhaustive_3pld_selection.ipynb` | **Authoritative** |
| **5** | **Final 3-PLD vs 6-PLD** | How does the mathematically selected 3-PLD compare to 6-PLD? | `notebooks/04_exhaustive_3pld_selection.ipynb` | **Authoritative** |
| **6** | **SNR Robustness** | Does the selected 3-PLD model generalize across noise conditions? | `notebooks/06_snr_robustness.ipynb` | **Authoritative** |
| **7** | **Signal Curve Analysis** | Where do the selected PLDs lie on the kinetic signal curve? | `notebooks/07_selected_pld_signal_curve_analysis.ipynb` | **Authoritative** |
| **8** | **Physiological Reference Analysis** | Does PLD location remain informative across CBF/ATT physiological space? | `notebooks/08_physiological_reference_pld_analysis.ipynb` | **Authoritative** (NEW) |
