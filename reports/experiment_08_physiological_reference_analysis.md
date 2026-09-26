# Experiment 8: Physiological Reference-Point Analysis

## 1. Research Question
Do the selected 3 PLDs (`[1.525, 2.525, 3.025]`) sample complementary regions of the ASL signal evolution across diverse physiological CBF and ATT conditions? Furthermore, how does the location, numerical behavior, and parameter sensitivity of each PLD change as CBF and ATT vary?

## 2. Motivation
In Experiment 7, it was visually demonstrated that the chosen PLDs sat on the arrival slope, near the kinetic peak, and at the decay onset for *central* physiological conditions. However, the exact location of the kinetic peak shifts heavily depending on the Arterial Transit Time (ATT). This experiment rigorously quantifies whether this spatial complementary sampling holds up mathematically across a 36-point grid of physiological reference conditions. It also numerically maps the sensitivity ($\partial S/\partial CBF$, $\partial S/\partial ATT$) of each measurement point to verify if the mathematically selected PLDs correspond to the physical regimes with maximum parameter information.

**Relationship to Experiment 4 & 7:** 
Experiment 4 objectively chose the best 3-PLD configuration purely via Validation MAE. Experiment 7 provided an initial visual analysis of these PLDs. Experiment 8 tests the broader hypothesis of *why* they work by systematically tracking their numerical location and derivatives across the full simulated parameter space.

## 3. Methodology
- **Simulation Model:** Standard Buxton kinetic model (`paper_signal` from `src/simulation.py`). Exact same constants ($\tau=1.8$, $T_{1b}=1.66$, $\alpha=0.85$, $\lambda=0.9$) used for authoritative dataset generation.
- **CBF/ATT Reference Grid:** 
  - CBF: `[20, 35, 50, 65, 80, 90]` ml/100g/min
  - ATT: `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]` s
  - Total: 36 unique physiological states, fully spanning the bounds used during DNN training.
- **Representative Extreme Cases:** Explicitly analyzed the corners of the domain (Low/Low, Low/High, High/Low, High/High) along with the central distribution.
- **PLD Curve Analysis:** For every condition and every PLD (both the 6 baseline and 3 selected), the following metrics were calculated:
  - Normalized Signal Amplitude ($S_{pld} / S_{max}$)
  - Absolute local slope ($|dS/dPLD|$)
  - Region Classification (e.g., `arrival/transition`, `near maximum`, `post-maximum/decay`) based on mathematical thresholds rather than visual guessing.
- **Sensitivity Analysis:** Computed finite-difference approximations for the partial derivatives:
  - CBF Sensitivity: $\partial S/\partial CBF$ evaluated at $CBF \pm 1.0$
  - ATT Sensitivity: $\partial S/\partial ATT$ evaluated at $ATT \pm 0.05$

## 4. Results

### Region Classification Distribution
Across the 36 physiological configurations, the selected PLDs overwhelmingly capture distinct, non-overlapping phases of the perfusion curve:
- **PLD 1.525s (Selected):** Uniquely captures the `arrival/transition` phase for normal/long ATTs, and crosses into `post-maximum/decay` only for the shortest ATTs (0.5s).
- **PLD 2.525s (Selected):** Predominantly sits `near maximum` or in the early `post-maximum/decay` region, anchoring the global kinetic peak across the majority of ATT distributions.
- **PLDs 3.525s & 4.025s (Discarded):** Almost entirely collapse into the `late decay` phase across all 36 conditions.

### Sensitivity Profiles
- **CBF Sensitivity:** The magnitude of $\partial S/\partial CBF$ is directly tied to the total amount of labeled blood in the tissue. The analysis shows that **PLD 1.525s** and **PLD 2.525s** consistently offer the highest absolute CBF sensitivity across the physiological grid. 
- **ATT Sensitivity:** ATT defines the horizontal shift of the curve. The steepest slopes naturally offer the highest sensitivity to horizontal shifts. **PLD 1.525s** (the rising edge) provides the dominant ATT sensitivity. **PLD 3.025s** offers secondary sensitivity on the trailing edge.

### Selected vs. Discarded Set
When analyzing the subset properties, the selected 3-PLDs maintain a significantly wider dispersion of signal magnitudes and a substantially closer mean proximity to the curve maximum than the discarded PLDs. The discarded set (particularly 3.525s and 4.025s) exhibits low sensitivity to both parameters and occupies heavily correlated positions deep in the exponential decay tail.

## 5. Interpretation
The results strongly indicate that the authoritative DNN PLD selection (`[1.525, 2.525, 3.025]`) from Experiment 4 is physically consistent with sampling maximally complementary information:
1. It retains the measurement with maximum ATT sensitivity (1.525s).
2. It retains the measurement with maximum CBF sensitivity (2.525s).
3. It retains a trailing measurement (3.025s) to establish the decay trajectory without wasting acquisition time on the fully asymptotic $T_1$ tail.

## 6. Limitations and What Cannot Be Concluded
- **No Causal Proof:** While the selected PLDs align perfectly with regions of maximum mathematical sensitivity, this experiment does *not* prove that the DNN explicitly "learned" the Buxton equation's partial derivatives. It merely demonstrates that the combination yielding minimal Validation MAE physically corresponds to the most informative kinetic regions.
- **Discretization:** The analysis assumes 6 discrete choices. A truly optimal protocol might involve non-uniform continuous sampling (e.g., 1.6s, 2.4s, 2.9s), which was beyond the scope of the pre-defined 6-PLD clinical sequence.

## 7. Future Experiments
1. **Exhaustive 4-PLD Selection:** Determining whether adding one more point (e.g., PLD 2.025s) bridges the remaining ~10% performance gap by providing redundant robustness against noise.
2. **Clinical Validation:** Verifying whether this analytically optimal subset maintains its superiority on in-vivo patient data subject to physiological motion and off-resonance artifacts.
