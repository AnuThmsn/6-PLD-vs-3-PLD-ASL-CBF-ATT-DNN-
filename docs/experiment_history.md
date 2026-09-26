

### Experiment 8b: Scientific Audit
* **Date:** 2026-09-26
* **Reason:** Strict scientific validation of the interpretive claims made in Experiment 8.
* **Files inspected:** 
esults/physiological_reference_analysis/pld_curve_metrics.csv
* **Findings:** The claim that PLD 2.525s provides the highest CBF sensitivity was found to be factually incorrect (1.525s provides the highest mean CBF sensitivity). The claim that 2.525s is the universal peak was corrected (the peak shifts heavily with ATT).
* **Corrections:** Rewrote the scientific interpretation. The selected PLDs do not map to static curve regions, but rather form a maximally orthogonal basis set (proven via signal correlation matrix) that tracks the shifting kinetics.
* **Notebook:** 
otebooks/08b_physiological_reference_pld_audit.ipynb
