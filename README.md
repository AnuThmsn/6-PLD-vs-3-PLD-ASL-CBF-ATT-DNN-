# 6-PLD vs 3-PLD ASL-MRI DNN Parameter Estimation

This repository investigates whether a Deep Neural Network (DNN) can maintain comparable estimation performance for Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) when the number of Post-Labeling Delays (PLDs) in Arterial Spin Labeling (ASL) MRI is reduced from a conventional 6 to just 3.

By accelerating the acquisition process (requiring fewer temporal samples), we aim to make ASL more robust to patient motion and clinical time constraints.

---

## 🎯 Primary Objective
**The core scientific question:** Can a DNN using only 3 PLDs estimate CBF and ATT with performance comparable to the existing 6-PLD DNN, when everything else (simulation physics, noise, neural network architecture) is kept perfectly controlled?

## 🧬 Methodology & Simulation
We utilized a simulated ASL dataset based on standard kinetic models (Buxton) augmented with Rician noise. This creates a perfectly reproducible testbed with absolute ground-truth knowledge.
- **CBF Range:** 0.0 - 100.0 ml/100g/min
- **ATT Range:** 0.5 - 3.0 s
- **Labeling Duration ($\tau$):** 1.8s
- **Baseline 6 PLDs:** `[1.525, 2.025, 2.525, 3.025, 3.525, 4.025]` s
- **Experimental 3 PLDs:** `[1.525, 2.025, 3.025]` s (Early, Middle, and Late samples)

## 🧠 Neural Network Architecture
We deployed identical Multi-Layer Perceptrons (MLPs) for all experiments, isolating the input dimension (the number of PLDs) as the sole experimental variable.
- **Architecture:** Input Layer $\rightarrow$ 9 Hidden Layers $\rightarrow$ 1 Output Node.
- **Width:** 50 for the CBF model, 100 for the ATT model.
- **Activation:** ELU (Exponential Linear Unit) for robust, noise-resilient gradient flow.
- **Loss & Optimizer:** Mean Absolute Error (MAE) trained via Adam ($lr=10^{-3}$).

---

## 📊 Final Results (1 Million Sample Experiment)

To establish true asymptotic algorithmic capacity, we scaled the training dataset to **1,000,000 samples**, evaluated on 10,000 completely unseen ground-truth target cases at SNR=10.

| Model | Parameter | MAE | RMSE | R² | Pearson | Bias |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **6-PLD** | CBF | **3.147** | **4.377** | 0.977 | 0.988 | -0.078 |
| **6-PLD** | ATT | **0.231** | **0.361** | 0.746 | 0.865 | +0.031 |
| | | | | | | |
| **4-PLD** | CBF | **3.307** | **4.559** | 0.975 | 0.987 | +0.104 |
| **4-PLD** | ATT | **0.236** | **0.368** | 0.737 | 0.860 | +0.026 |
| | | | | | | |
| **3-PLD** | CBF | **3.857** | **5.437** | 0.964 | 0.982 | +0.264 |
| **3-PLD** | ATT | **0.253** | **0.393** | 0.701 | 0.841 | +0.039 |

### Interpretation
The massive dataset experiment proves the theoretical hierarchy: **6-PLD > 4-PLD > 3-PLD**. However, the **3-PLD model remains highly viable**. It tracks ATT almost identically to the 6-PLD baseline, and estimates CBF with an absolute MAE penalty of only ~0.7 ml/100g/min, retaining an excellent $R^2 > 0.96$. 

This strongly suggests that halving the number of ASL delay acquisitions is a highly practical strategy when combined with DNN estimation.

---

## 📂 Repository Structure
- `/notebooks`: Contains all executable experiments.
  - `train_6_pld.ipynb` & `train_3_pld.ipynb`: The core comparative baseline and experimental implementations on 4k samples.
  - `train_1m_samples.ipynb`: The massive 1-million sample definitive evaluation.
- `/src/simulation.py`: Centralized kinetic ASL generation and dataset processing.
- `/results`: Checkpoints (`.pt`), saved `metrics.csv`, and training history logs.
- `report.md`: Detailed engineering and architectural justification.


## Experiment 7: Selected PLDs on ASL Signal Curves
**Objective:** Visualize and quantify *where* the selected 3 PLDs ([1.525, 2.525, 3.025]) fall on the theoretical ASL kinetic signal evolution curve compared to the discarded PLDs, investigating why this combination yields optimal DNN performance.

**Methodology:**
- Generated dense (500-point) ASL signal curves across diverse physiological CBF/ATT conditions using the Buxton kinetic model.
- Evaluated the position, normalized signal strength, and absolute slopes of the selected vs. discarded PLDs.
- Performed randomized aggregate analysis over 100 cases to extract general properties of the PLD selection.

**Key Findings:**
1. **Early Arrival (PLD 1.525s):** The earliest measurement is strongly retained because it captures the crucial arrival slope, acting as the primary differentiator for short Arterial Transit Times (ATTs).
2. **Kinetic Peak (PLD 2.525s):** This measurement sits at or immediately after the mean signal peak for central ATT distributions. Anchoring the maximum signal intensity is highly informative for Cerebral Blood Flow (CBF) scaling.
3. **Decay Onset (PLD 3.025s):** This establishes the initial trajectory of the $ relaxation decay phase.
4. **Discarded Tails (3.525s, 4.025s):** The deep decay phase is highly predictable (pure exponential) and carries the lowest signal-to-noise ratio. The combinatorial selection algorithm correctly identified these as redundant for a non-linear estimator.

**Notebook:** 
otebooks/07_selected_pld_signal_curve_analysis.ipynb
