# 6-PLD versus 4-PLD ASL DNN experiment

The training notebooks now follow the provided paper-matched reference notebook: PLDs `[1.525, 2.025, 2.525, 3.025]` seconds, labeling duration `1.8` seconds, CBF range `[0, 100]`, ATT range `[0.5, 3.0]` seconds, dense noise-SD levels, and Rician `mc + ml` generation. The 6-PLD notebook uses those four reference PLDs plus `[3.525, 4.025]` seconds as an explicit extension because the reference notebook defines four PLDs.

The simplified project structure is:

```text
src/simulation.py
notebooks/train_6_pld.ipynb
notebooks/train_4_pld.ipynb
notebooks/compare_10000_samples.ipynb
```

Run the notebooks in this order:

1. Run `notebooks/train_6_pld.ipynb`.
2. Run `notebooks/train_4_pld.ipynb`.
3. Run `notebooks/compare_10000_samples.ipynb`.

Only the paper-matched physics and noise simulation is shared in `src/simulation.py`. The separate CBF and ATT DNN definitions and training loops are visible directly in each training notebook. Both notebooks use 9 ELU hidden layers, MAE loss, Adam, gradient clipping, a maximum of 200 epochs, and early stopping patience of 20.

The comparison notebook generates exactly 10,000 new test samples and evaluates both saved models on the identical signals and targets. Results are saved under `results/model_6_pld/`, `results/model_4_pld/`, and the comparison CSV/plot files directly under `results/`.
