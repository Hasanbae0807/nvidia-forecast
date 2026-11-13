# Calibrated Persistence Blend

Entry for make.fun's **polymarket-nvidia-ceo-forecast-01**: forecast the Yes price of
every market in Polymarket's "Next CEO of NVIDIA?" event, minute by minute for the
240 minutes after each checkpoint, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

`main.py` is the whole entry: Python 3.13, standard library only, deterministic
(a pure function of the case it is given), ~0.2 s per case.

## Method

- **Median = persistence.** The median forecast is the checkpoint price. For an
  illiquid event market like this one, the last price is the best point forecast;
  carrying historical drift into the median measurably hurt in backtests.
- **Spread = a 60/40 blend of two per-market estimates** from the case's own history:
  - **A (60%) — level-dependent sigma.** The MAD of the market's 60-minute changes
    over its last 12 hours, each change normalised by `sqrt(p(1-p))` (a binary price
    near 0 or 1 cannot move like one at 0.5), scaled back to the checkpoint price's
    level and shrunk geometrically (weight 0.6) toward a global 0.0020/sqrt(min)
    prior. Quantiles are the normal's z * sigma * sqrt(h).
  - **B (40%) — empirical change quantiles.** The distribution of the market's own
    h-minute changes over its last 24 hours, at log-spaced horizons with log-h
    interpolation, recentred to zero median change. This keeps the real shape
    (jump tails, asymmetry) without importing drift.
- Component rows are clamped to [0, 1] and monotonised before blending; output is
  rounded to 6 decimals.

## Backtest (mean pinball loss, percentage points; lower is better)

Scored with a local scorer that reproduces the house baseline's published practice
figures exactly (loss 0.363589, median error 0.767503, coverage 0.949642), on the
kit's six practice cases and on five extra checkpoints cut from the same fetched
price history (2026-11-11, a far more volatile regime):

| Model | Practice (kit p00–p05) | Stress (5 extra checkpoints) |
|---|---|---|
| House baseline (global sigma 0.0025) | 0.363589 | 1.526201 |
| Zero-width persistence | 0.383752 | — |
| Best single model (level-dependent sigma) | 0.268586 | 1.599057 (loses to baseline) |
| **This entry (blend, actual outputs)** | **0.282897** | **1.497599** |

The practice-optimal single model was rejected: it loses to the baseline in the
volatile regime (regime overfit). The shipped blend beats the baseline in both
regimes — about −22% on practice, −2% on the stress set — with 90%-interval
coverage of 0.877 on practice.

MIT licensed (see LICENSE).
