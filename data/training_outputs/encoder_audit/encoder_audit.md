# Encoder feature audit
- samples per checkpoint: **8000** (same fixed patch indices for every run)
- checkpoint file: `best_model.pt`
- device: `cuda`
- total runtime: 656.5s

**Diagnostic formulae**
- `σ̄` = mean over feature dims of per-dim std (`Z.std(0, ddof=1).mean()`).
- `ρ̄` = mean `|ρ_ij|` over strict upper triangle of `corrcoef(Z)`; dims with std<1e-6 are masked.
- `rank` = `exp(-Σ p log p)` with `p = σ²/Σσ²` and `σ` the singular values of mean-centred `Z` (RankMe; Garrido et al. 2023).

## variant: `default`
| init | enc σ̄ | enc ρ̄ | enc rank | proj σ̄ | proj ρ̄ | proj rank |
|---|---|---|---|---|---|---|
| scratch | 0.0606 | 0.1090 | 6.32 | 0.3288 | 0.2215 | 49.12 |
| moby | 0.0478 | 0.1469 | 4.88 | 0.2862 | 0.2307 | 38.25 |
| timm_imagenet | 0.0447 | 0.1374 | 4.59 | 0.2863 | 0.2365 | 36.17 |

## variant: `fourier_vicreg_on`
| init | enc σ̄ | enc ρ̄ | enc rank | proj σ̄ | proj ρ̄ | proj rank |
|---|---|---|---|---|---|---|
| scratch | 0.5844 | 0.1832 | 16.76 | 0.8325 | 0.0294 | 753.36 |
| moby | 0.4640 | 0.1272 | 40.03 | 0.9297 | 0.0251 | 993.39 |
| timm_imagenet | 0.5010 | 0.1284 | 38.67 | 0.9328 | 0.0279 | 864.86 |

## variant: `fourier_vicreg_weak`
| init | enc σ̄ | enc ρ̄ | enc rank | proj σ̄ | proj ρ̄ | proj rank |
|---|---|---|---|---|---|---|
| scratch | 0.6413 | 0.1806 | 17.39 | 0.8366 | 0.0271 | 795.48 |
| moby | 0.4240 | 0.1351 | 36.99 | 0.8861 | 0.0279 | 831.00 |
| timm_imagenet | 0.4751 | 0.1360 | 35.60 | 0.9117 | 0.0299 | 771.72 |

## variant: `no_fourier_vicreg_off`
| init | enc σ̄ | enc ρ̄ | enc rank | proj σ̄ | proj ρ̄ | proj rank |
|---|---|---|---|---|---|---|
| scratch | 0.0596 | 0.1049 | 6.54 | 0.3268 | 0.2182 | 53.13 |
| moby | 0.0423 | 0.1580 | 5.60 | 0.2514 | 0.2227 | 43.41 |
| timm_imagenet | 0.0494 | 0.1576 | 3.96 | 0.2967 | 0.2589 | 27.25 |

## variant: `weak_long`
| init | enc σ̄ | enc ρ̄ | enc rank | proj σ̄ | proj ρ̄ | proj rank |
|---|---|---|---|---|---|---|
| scratch | 0.6058 | 0.1723 | 18.21 | 0.8582 | 0.0260 | 911.65 |
