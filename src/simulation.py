"""Paper-matched ASL simulation shared by the 4-PLD and 6-PLD notebooks."""
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

@dataclass
class SimulationConfig:
    seed: int = 42
    cbf_range: tuple = (0.0, 100.0)
    att_range_ms: tuple = (500.0, 3000.0)
    att_range_s: tuple = (0.5, 3.0)
    tau_s: float = 1.8
    scale: float = 100000.0
    alpha: float = 0.85
    beta: float = 0.75
    lambda_blood: float = 0.9
    t1_tissue_s: float = 1.2
    t1_blood_s: float = 1.66
    plds_4_s: tuple = (1.525, 2.025, 2.525, 3.025)
    # The reference notebook defines four PLDs. These two late points are the
    # explicit extension used only for the requested six-input comparison.
    plds_6_s: tuple = (1.525, 2.025, 2.525, 3.025, 3.525, 4.025)
    plds_4_indices: tuple = (0, 1, 2, 3)
    reference_pld_s: float = 2.0
    reference_ld_s: float = 1.8
    reference_cbf: float = 50.0
    reference_att_s: float = 1.6

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def paper_signal(cbf, att_s, plds_s, cfg: SimulationConfig):
    cbf = np.asarray(cbf, dtype=np.float64)
    att = np.asarray(att_s, dtype=np.float64)
    plds = np.asarray(plds_s, dtype=np.float64)[None, :]
    delta = att[:, None]
    flow = (cbf / (6000.0 * cfg.lambda_blood))[:, None]
    # Buxton: ΔM = 2·α·β·T1t·(M0/λ)·f·exp(...)
    # flow already contains cbf/(6000·λ), so prefix should NOT divide by λ again.
    # Previous version had (1.0/λ)*flow which caused double-division (λ² bug).
    prefix = 2.0 * cfg.alpha * cfg.beta * cfg.t1_tissue_s * flow
    first = np.exp(-np.maximum(plds - delta, 0.0) / cfg.t1_tissue_s)
    second = np.exp(-np.maximum(cfg.tau_s + plds - delta, 0.0) / cfg.t1_tissue_s)
    return cfg.scale * prefix * np.exp(-delta / cfg.t1_blood_s) * (first - second)

def reference_signal(cfg: SimulationConfig) -> float:
    return float(abs(paper_signal([cfg.reference_cbf], [cfg.reference_att_s], [cfg.reference_pld_s], cfg)[0, 0]))

def noisy_signals(cbf, att_s, snr, cfg: SimulationConfig, rng=None, n_noise_levels=100):
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    clean = paper_signal(cbf, att_s, cfg.plds_6_s, cfg)
    sd_max = reference_signal(cfg) / 5.0
    if np.isscalar(snr):
        sd_levels = np.linspace(0.0, sd_max, n_noise_levels)
        sd = sd_levels[rng.integers(0, n_noise_levels, len(clean))]
    else:
        sd = reference_signal(cfg) / np.asarray(snr)
    sd = sd[:, None]
    e1 = rng.normal(0.0, 1.0, clean.shape) * sd
    e2 = rng.normal(0.0, 1.0, clean.shape) * sd
    e3 = rng.normal(0.0, 1.0, clean.shape) * sd
    e4 = rng.normal(0.0, 1.0, clean.shape) * sd
    mc = np.sqrt((clean + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((clean + e3) ** 2 + e4 ** 2)
    return (mc + ml).astype(np.float32)

def generate_dataset(n_samples: int, cfg=None, seed=None, snr=10.0):
    cfg = SimulationConfig() if cfg is None else cfg
    seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    cbf = rng.uniform(*cfg.cbf_range, n_samples).astype(np.float32)
    att = rng.uniform(*cfg.att_range_s, n_samples).astype(np.float32)
    if np.isscalar(snr):
        snr_values = np.full(n_samples, snr, dtype=np.float32)
    else:
        snr_values = np.asarray(snr, dtype=np.float32)
    signals = noisy_signals(cbf, att, snr_values if not np.isscalar(snr) else snr, cfg, np.random.default_rng(seed + 1000))
    targets = np.column_stack((cbf, att)).astype(np.float32)
    return signals, targets, snr_values

def select_inputs(signals, cfg: SimulationConfig, mode: str):
    if mode == "6_pld":
        return signals
    if mode == "4_pld":
        return signals[:, cfg.plds_4_indices]
    raise ValueError("mode must be '6_pld' or '4_pld'")

def split_dataset(n_train=4000, n_validation=1000, cfg=None):
    cfg = SimulationConfig() if cfg is None else cfg
    train = generate_dataset(n_train, cfg, cfg.seed + 10, snr=10.0)
    validation = generate_dataset(n_validation, cfg, cfg.seed + 20, snr=10.0)
    return train, validation
