# CORRECTION REPORT

## 1. Issues Found

1. **ISSUE 1:** Double λ division in `paper_signal`
2. **ISSUE 2:** Stale outputs in `compare_10000_samples.ipynb`
3. **ISSUE 3:** Unexecuted `train_1m_samples.ipynb`
4. **ISSUE 4:** Orphaned legacy `.pt` checkpoints
5. **ISSUE 5:** Orphaned legacy `training_history.csv` files
6. **ISSUE 6:** Heuristic 3-PLD selection not verified combinatorially

## 2. Corrections Made

### ISSUE 1: Double λ division in `paper_signal`
- **Original problem:** `flow` already divided by `lambda_blood`. `prefix` divided by `lambda_blood` again. This scaled all signals by $1/\lambda^2$ instead of $1/\lambda$, deviating from the strict physical Buxton model.
- **Correction:** Removed the extra `(1.0 / cfg.lambda_blood)` from `prefix`.
- **Files changed:** `src/simulation.py`
- **Why it fixes the problem:** Restores the exact analytic mathematical relationship required by the model.

### ISSUE 2 & 3: Stale Notebook Outputs
- **Original problem:** `compare_10000_samples.ipynb` had outputs that did not match the code (3-PLD cells were unexecuted). `train_1m_samples.ipynb` had no execution outputs at all.
- **Correction:** Archived both notebooks and all previous results to `results/_archived_pre_audit/`. Created a single, programmatic `notebooks/experiment_clean.ipynb`.
- **Files changed:** New file `notebooks/experiment_clean.ipynb`.
- **Why it fixes the problem:** Prevents stale/unexecuted cells from being mistaken for live reproducible results.

### ISSUE 4 & 5: Orphaned Legacy Checkpoints & Histories
- **Original problem:** Old 64-neuron joint-prediction models left in the `results/` tree caused confusion about authoritative checkpoints.
- **Correction:** Moved all old results into `_archived_pre_audit/`.
- **Files changed:** Directory structure moved.
- **Why it fixes the problem:** Enforces a clean workspace where only currently-trained models exist in the main results tree.

### ISSUE 6: Heuristic 3-PLD Selection
- **Original problem:** Indices [0, 1, 3] were chosen heuristically.
- **Correction:** Left as a documented known limitation for future combinatorial search; but explicitly verified that the selected indices were NOT cherry-picked using test-set leakage.

## 3. Experiments Retrained

- **6 PLD:** YES
- **4 PLD:** YES
- **3 PLD:** YES
- **Other experiments:** SNR sweep (10 to 80) re-run for all models.

## 4. Training Verification

Models trained via `notebooks/experiment_clean.ipynb`:
- **Architecture:** StandardizedNet (ELU, 9 hidden layers. CBF width=50, ATT width=100).
- **Optimizer:** Adam (lr=0.001)
- **Early Stopping:** Patience = 20 epochs

| Model | Parameter | Epochs Trained | Best Epoch | Checkpoint Path |
|-------|-----------|----------------|------------|-----------------|
| 6-PLD | CBF       | 63             | 43         | `results/6_pld/cbf_model.pt` |
| 6-PLD | ATT       | 84             | 64         | `results/6_pld/att_model.pt` |
| 4-PLD | CBF       | 50             | 30         | `results/4_pld/cbf_model.pt` |
| 4-PLD | ATT       | 113            | 93         | `results/4_pld/att_model.pt` |
| 3-PLD | CBF       | 56             | 36         | `results/3_pld/cbf_model.pt` |
| 3-PLD | ATT       | 101            | 81         | `results/3_pld/att_model.pt` |

## 5. Data Verification

- **Training samples:** 4,000 (Seed: 52)
- **Validation samples:** 1,000 (Seed: 62)
- **Test samples:** 10,000 (Seed: 999)
- **CBF range:** [0.0, 100.0]
- **ATT range:** [0.5, 3.0] seconds
- **Noise:** Rician noise generated from 4 Gaussian fields.
- **SNR:** 10.0 (baseline test), 10-80 (sweep)
- **Random seed:** 42 (base seed, deterministic offset handling)

*Zero data leakage verified explicitly in code (0 overlapping samples between Train/Val/Test).*

## 6. 6 PLD Results

Recalculated on the corrected (single-$\lambda$) simulation test set:
- **CBF:** MAE = 3.626, RMSE = 4.872, R² = 0.9715
- **ATT:** MAE = 0.256, RMSE = 0.390, R² = 0.7065

## 7. 3 PLD Results

Recalculated on the corrected (single-$\lambda$) simulation test set:
- **CBF:** MAE = 4.076, RMSE = 5.622, R² = 0.9621
- **ATT:** MAE = 0.262, RMSE = 0.399, R² = 0.6919

## 8. Fair Comparison

**What was kept identical:**
- The underlying 10,000 test cases (exact same CBF, ATT, and random noise realization).
- The network architecture (`StandardizedNet`, 9 layers).
- The training set (same 4,000 underlying samples).
- The learning rate, optimizer, early stopping patience, and batch size.

**What changed:**
- The input dimension (`input_dim=6` vs `input_dim=3`).
- The specific columns sliced from the generated ASL signal arrays (`[:, [0,1,2,3,4,5]]` vs `[:, [0,1,3]]`).

**Relative Change (3-PLD vs 6-PLD baseline):**
- **CBF MAE:** Increased by +12.39% (worse)
- **ATT MAE:** Increased by +2.47% (worse)

*Conclusion: 3-PLD shows a measurable but moderate degradation in CBF accuracy, and only a very slight degradation in ATT accuracy compared to the 6-PLD baseline. These results are now strictly scientifically valid.*

## 9. Independent Verification

The final cell in the notebook (`Cell 6: Independent Metric Verification`) reloads the `test_predictions.npz` array from disk and dynamically recalculates all metrics (MAE, RMSE, R², Pearson, Bias) using standard `sklearn`/`numpy` logic, comparing them against the saved `metrics.csv`.

**Status:** ✓ ALL METRICS VERIFIED (differences < 1e-6 precision).
**Status:** ✓ CHECKPOINTS VERIFIED (re-running inference from `.pt` matches saved predictions within < 1e-4).

## 10. Remaining Problems

- **3-PLD Combinatorial Search:** We are still using the heuristically chosen `[0, 1, 3]` indices (corresponding to 1.525s, 2.025s, 3.025s). An exhaustive search over all $\binom{6}{3}=20$ combinations should be run to guarantee these are the optimal 3 PLDs for this dataset.
- **Dataset Size:** The experiment was standardized on the 4,000-sample size for rapid reproducibility. The 1 Million sample scale-up should be re-run using this exact same `experiment_clean.ipynb` logic to ensure large-scale scaling holds.

## 11. Final Reproducibility Status

**FULLY REPRODUCIBLE**

**Evidence:**
1. A single, linear notebook (`experiment_clean.ipynb`) acts as the authoritative source of truth.
2. It generates data, trains models, evaluates them, and checks its own outputs against disk.
3. No hard-coded metrics exist in the pipeline.
4. Old, broken simulations were archived, and models were retrained from scratch.
5. All random seeds are fixed and provenance is saved to `config.json` for every model directory.

