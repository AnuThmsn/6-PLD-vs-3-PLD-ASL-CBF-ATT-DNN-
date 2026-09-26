# Experiment 13: Literature-Grounded Architecture Sweep (Phase B)

## Research Question
Given that generic sequence models (TCN, GRU, Transformers) failed to break the parameter estimation floor, can architectures specifically grounded in ASL and IVIM literature—namely Deep Standardized MLPs, PLD-aware sequences, and Physics-Informed Neural Networks—reduce the absolute parameter error or mitigate the 3-PLD degradation penalty?

## Literature Basis
1. **Mastropietro et al. (2022) / Ishida et al. (2024)**: Emphasize the importance of standardization and evaluated fully-connected models of varying depth. We test Deep MLPs with 3, 5, 7, and 9 hidden layers.
2. **Kim et al. (2023) / Toronto ASL**: Explored hierarchical processing for missing PLDs. We abstract this to a PLD-Aware Attention model that dynamically processes $(PLD_i, S_i)$ tuples.
3. **Ishida et al. (2024)**: Validated a PINN approach where parameter estimates are forced through the physical forward model to reconstruct the signal. We test this with a multi-objective loss function controlled by $\lambda_{phys}$.

## Methodology
- **Training Data**: 100,000 synthetic samples.
- **Signal**: Buxton single-compartment model.
- **Evaluation**: 6-PLD vs 3-PLD `[1.525, 2.525, 3.025]`.

## Results
| Experiment | Model | 6-PLD CBF | 6-PLD ATT | 3-PLD CBF | 3-PLD ATT | 3-PLD CBF Penalty | 3-PLD ATT Penalty |
|---|---|---|---|---|---|---|---|
| A | DeepMLP-3 | 4.39 | 0.372 | 4.93 | 0.389 | +12.2% | +4.8% |
| A | DeepMLP-5 | 4.38 | 0.372 | 4.92 | 0.390 | +12.2% | +4.8% |
| A | DeepMLP-7 | 4.40 | 0.368 | 4.94 | 0.390 | +12.4% | +5.9% |
| A | DeepMLP-9 | 4.38 | 0.369 | 4.92 | 0.389 | +12.5% | +5.4% |
| B | PLDAware Attn | 4.48 | 0.374 | 5.06 | 0.393 | +12.8% | +5.0% |
| C | PINN ($\lambda=0.01$) | 4.85 | 0.385 | 5.32 | 0.407 | +9.6% | +5.5% |
| C | PINN ($\lambda=0.1$) | 6.02 | 0.487 | 6.12 | 0.487 | **+1.7%** | **0.0%** |
| C | PINN ($\lambda=0.5$) | 6.26 | 0.494 | 6.20 | 0.493 | **-0.9%** | **-0.1%** |

## Interpretation
1. **Deeper MLPs**: Increasing depth to 9 layers did not improve performance. The models remain completely bounded by the ~4.38 CBF / 0.37s ATT limit established in Phase A.
2. **PLD-Aware Attention**: Encoding PLD timing explicitly did not provide an advantage, indicating the baseline MLPs already perfectly learn the implicit temporal mapping of the 6 fixed inputs.
3. **Physics-Informed (PINN)**: The PINN successfully eradicated the 3-PLD penalty (0%). However, absolute error degraded significantly (CBF > 6.0). This occurs because the physics loss forces the parameter predictions to reconstruct the *noisy* ASL observation. In highly degenerate, low-SNR regimes, reconstructing the noise actively pulls the parameter estimates away from the true underlying clean parameters. 

## Conclusion
The literature-grounded architecture sweep conclusively supports the findings of Phase A: the primary limitation is information content and identifiability, not architectural capacity. Physics-informed regularization can eliminate structural degradation (the 3-PLD penalty) but is highly sensitive to noise overfitting in ill-posed settings.
