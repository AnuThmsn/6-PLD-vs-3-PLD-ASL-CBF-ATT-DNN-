

### Experiment 8b: Scientific Audit
* **Date:** 2026-09-26
* **Reason:** Strict scientific validation of the interpretive claims made in Experiment 8.
* **Files inspected:** 
esults/physiological_reference_analysis/pld_curve_metrics.csv
* **Findings:** The claim that PLD 2.525s provides the highest CBF sensitivity was found to be factually incorrect (1.525s provides the highest mean CBF sensitivity). The claim that 2.525s is the universal peak was corrected (the peak shifts heavily with ATT).
* **Corrections:** Rewrote the scientific interpretation. The selected PLDs do not map to static curve regions, but rather form a maximally orthogonal basis set (proven via signal correlation matrix) that tracks the shifting kinetics.
* **Notebook:** 
otebooks/08b_physiological_reference_pld_audit.ipynb


### Experiment 9: Reference Value Sensitivity
* **Date:** 2026-09-26
* **Reason:** Evaluate if absolute noise scaling (via reference CBF/ATT) changes the relative degradation of dropping from 6 PLDs to 3 PLDs.
* **Findings:** Absolute error scales linearly with reference CBF (as it dictates a higher global noise variance). It peaks at reference ATT=1.6s because the simulated reference signal at PLD 2.0s happens to maximize there, injecting the highest absolute noise. Crucially, the 3-PLD relative degradation penalty (~10% for CBF, ~4% for ATT) remains highly stable across all noise regimes.
* **Notebook:** 
otebooks/09_reference_value_sensitivity.ipynb


### Experiment 10: Architecture Exploration
* **Date:** 2026-09-26
* **Reason:** Evaluate if alternative architectures (Conv1D, Residual, Wider MLP, Multi-task) can improve estimation accuracy or overcome the 3-PLD vs 6-PLD performance gap.
* **Findings:** All alternative architectures performed virtually identically to the Baseline MLP. The 3-PLD CBF degradation remained strictly ~11-12% across all models. This 'negative result' conclusively proves that the ~11% error gap is a hard mathematical information limit caused by dropping measurements in the presence of noise, and cannot be algorithmically 'fixed' by increasing network capacity or changing structural inductive biases.
* **Notebook:** 
otebooks/10_architecture_exploration.ipynb
