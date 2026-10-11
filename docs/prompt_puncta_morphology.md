# Prompt: implement the puncta morphology analysis

Copy everything below the line into a new Claude Code session opened on this
repository. The work is split into 5 steps. The agent stops after each step,
shows the result, and waits for "continue" before starting the next one.

---

You are working in the `anokhver/thesis` repository (Python package
`src/synaptic_ssl`). Implement a per-punctum morphology analysis that writes
**one CSV with one row per punctum**. Every histogram, density value and
group comparison must be computable from that CSV alone. The design is in
`docs/plan_puncta_morphology.md`; read it first.

## Ground rules

- Work in **5 steps**, in order. After each step: run its checks, show me the
  output, commit with a short message, and **stop and wait for me to say
  "continue"**. Do not start the next step on your own.
- Match the style of the surrounding code: module docstrings, type hints,
  `from __future__ import annotations`, small pure functions, `logging`
  rather than `print` in library code.
- Reuse existing functions. Do not copy them, and do not change their
  behaviour:
  - `synaptic_ssl.pseudolabels.puncta_log`: `detect_puncta_channel`,
    `resolve_cfg`, `DEFAULT_PUNCTA_CFG_PRE`, `DEFAULT_PUNCTA_CFG_POST`,
    `PunctaCfg`
  - `synaptic_ssl.pseudolabels.puncta_common`: `restrict_puncta_to_near`
  - `synaptic_ssl.clustering.groups`: `load_group_rules`,
    `classify_image_by_patterns`
  - `scripts/pseudolabels/puncta_from_mip.py`: copy its CLI conventions
    (`--input_dir`, `--dendrite_dir`, `--pre_channel` default 0,
    `--post_channel` default 1, `--near_dilate_px`, `--cfg_pre`,
    `--cfg_post`, `--no_auto_floors`) and its dendrite-mask loading
    (`<stem>_dend.npy`, or stitched `<stem>_r??_c??_dend.npy` tiles).
- Inputs are full-MIP `.npy` files of shape `(C, H, W)`. The dendrite masks
  already exist (produced by `scripts/pseudolabels/dendrite_from_mip.py` from
  structural channel 2). Do not recompute them.
- Puncta centres come **only from LoG** (`detect_puncta_channel`, the `kept`
  output). Do not use the segmentation model.
- If something in the repo contradicts this prompt, stop and ask me instead
  of guessing.

## Step 1: ellipse from moments (pure function + tests)

Create `src/synaptic_ssl/analysis/__init__.py` and
`src/synaptic_ssl/analysis/puncta_morphology.py` containing:

```python
def ellipse_from_moments(
    image: np.ndarray,          # 2D, background-subtracted (tophat) channel
    row: float, col: float,     # LoG centre
    sigma: float,               # LoG scale
    *, k: float = 3.0,          # window half-size = ceil(k * sigma), min 2 px
) -> dict | None:
```

Algorithm:
1. Crop the window `[row ± h, col ± h]` with `h = max(2, ceil(k*sigma))`.
   Return `None` if the window would cross the image border.
2. Background = median of the 1-px ring along the window edge. Weights
   `w = clip(window - background, 0, None)`. Return `None` if
   `m00 = w.sum() <= 1e-9`.
3. Centroid `ȳ = Σw·y / m00`, `x̄ = Σw·x / m00`, in full-image coordinates.
4. Central moments `μ20 = Σw(x−x̄)²/m00`, `μ02 = Σw(y−ȳ)²/m00`,
   `μ11 = Σw(x−x̄)(y−ȳ)/m00`.
5. Covariance `[[μ20, μ11], [μ11, μ02]]`, eigenvalues `λ1 ≥ λ2 ≥ 0` (clip
   tiny negatives to 0; use `np.linalg.eigh`).
6. Return a dict with:
   `x, y` (centroid, px), `a_px = 2√λ1`, `b_px = 2√λ2`,
   `radius_px = √(a_px·b_px)`, `eccentricity = √(1 − λ2/λ1)` (0 if λ1 = 0),
   `theta_deg = degrees(½·atan2(2μ11, μ20 − μ02))`,
   `intensity = m00`, `lambda1, lambda2`.

Tests in `tests/test_puncta_morphology.py` (pytest, synthetic images only):
- An isotropic Gaussian with σ = 2 gives eccentricity < 0.05, `a ≈ b ≈ 4 px`
  (±10 %), and centroid error < 0.1 px for a sub-pixel centre.
- An anisotropic Gaussian with σx = 3, σy = 1.5, rotated 30°, gives
  `a ≈ 6`, `b ≈ 3` (±10 %), `theta ≈ 30°` (±3°), and
  `eccentricity ≈ √(1 − 0.25)` (±0.03).
- A constant background offset added to the image does not change the
  result.
- A centre next to the border, or an all-zero image, returns `None`.

Check: `pytest tests/test_puncta_morphology.py -q` passes. Show the output.

## Step 2: analysis of one image → per-punctum DataFrame

In the same module add:

```python
def analyse_image(
    npy_path: Path,
    dend_mask: np.ndarray,           # bool (H, W)
    *,
    pre_channel: int, post_channel: int,
    cfg_pre: PunctaCfg, cfg_post: PunctaCfg,
    near_dilate_px: int,
    px_size_um: float,
    auto_floors: bool = True,
    k: float = 3.0,
    group_rules: dict | None = None,
) -> pd.DataFrame:
```

For each `role, channel_index` in `[("pre", pre_channel), ("post", post_channel)]`:
1. `img = full[channel_index].astype(float32)`. Run
   `detect_puncta_channel(img, cfg, auto_floors=...)` and take `kept`.
2. `near = dilation(dend_mask, disk(near_dilate_px))` (or `dend_mask` itself
   if the value is 0). Keep the puncta with `restrict_puncta_to_near(kept, near)`.
   Only the **dendrite** mask is used here; the soma is not.
3. `dendrite_area_um2 = near.sum() * px_size_um**2`. Density uses the **same
   mask** that gated the puncta, so the counts and the area match.
4. Apply the same white tophat that `detect_puncta_channel` uses
   (`cfg.intensity_tophat_radius`). Call `ellipse_from_moments` on that image
   for every kept punctum and drop the `None` results. Log how many were
   dropped.
5. Build one row per punctum, with exactly these columns in this order:

```
file_name;group_name;role;channel_index;dendrite_channel_index;x;y;radius_um;a_um;b_um;eccentricity;theta_deg;intensity;sigma_log;dendrite_area_um2;px_size_um;near_dilate_px
```

- `file_name` = `npy_path.name`.
- `group_name` = `classify_image_by_patterns(...)` with the rules from
  `configs/clustering/group_patterns.json`, or `""` if nothing matches.
- `role` is the literal string `pre` or `post`.
- `channel_index` is the integer index into `(C, H, W)` that the punctum was
  detected in.
- `dendrite_channel_index` is the structural channel the dendrite mask was
  built from (CLI argument, default 2).
- Lengths in µm are `*_px * px_size_um`. `x`, `y` stay in pixels.
- `sigma_log` is the LoG scale (kept for traceability).

An image with zero puncta in a role still produces no rows for that role,
but its area must not be lost. Also return (or log) a per-image record of
`n_pre`, `n_post` and `dendrite_area_um2` so that I can see the case.

Check: run `analyse_image` on one real MIP (ask me for the path if none is
under `data/`). Print `df.head()`, the number of rows per role, the
pre/post density, and the min/median/max of `radius_um` and `eccentricity`.
Save a PNG that overlays the fitted ellipses (red = pre, green = post) on a
crop of the image, and show it to me.

## Step 3: CLI script → CSV

Create `scripts/analysis/puncta_morphology.py`, modelled on
`scripts/pseudolabels/puncta_from_mip.py`:

- Arguments: `--input_dir`, `--dendrite_dir`, `--output_csv`,
  `--pre_channel 0`, `--post_channel 1`, `--dendrite_channel 2`,
  `--near_dilate_px` (same default as `puncta_from_mip.py`),
  `--px_size_um 0.10685428060522417` (107 nm/px, the value used across the
  repo), `--k 3.0`, `--cfg_pre`, `--cfg_post`, `--no_auto_floors`,
  `--group_patterns configs/clustering/group_patterns.json`, `--workers 0`.
- Loop over `*.npy` in `--input_dir` and skip `*_pre/_post/_soma/_dend.npy`.
  Load the dendrite mask, call `analyse_image`, and concatenate the results.
- Write the CSV with `sep=";"`, `index=False`, UTF-8, and the column order
  from step 2.
- Also write `<output_csv stem>_images.csv` (`;`-separated, one row per image
  × role): `file_name;group_name;role;channel_index;n_puncta;dendrite_area_um2`.
  This file is needed only so that images with 0 puncta still count in the
  density statistics.
- Log a skip (do not crash) for an image whose dendrite mask is missing.

Check: run it on a small folder (2–3 images). Show the CSV header, 5 rows,
and the row count per `file_name` × `role`. Verify that
`df.groupby(["file_name","role"]).size() / dendrite_area_um2` gives the same
density as step 2.

## Step 4: notebook that reads only the CSV

Create `notebooks/analysis/puncta_histograms.ipynb`. It must **not** import
any image-processing code. It only uses `pandas`, `numpy`, `matplotlib` and
`scipy.stats`.

1. Load the two CSVs (`sep=";"`). Rebuild density per (`file_name`, `role`)
   from `n_puncta / dendrite_area_um2` in the `_images.csv`.
2. A `time_filter` cell where I choose the subset by `file_name` or
   `group_name` substring (for example `"24h"` vs `"1h pulse"`). Do not
   hard-code the grouping anywhere else.
3. For `radius_um`, `a_um`, `b_um`, `eccentricity` and density:
   - one histogram figure per quantity, with pre and post overlaid and the
     chosen subsets as separate panels;
   - the same bin edges in all panels (computed once from the full CSV);
   - a smooth density curve (`scipy.stats.gaussian_kde`) drawn over each
     histogram, so the continuous distribution is visible independent of the
     bin width;
   - a text box with n, min, max, 5th / 50th / 95th percentile.
4. One summary table: per subset × role, n puncta, number of images, and the
   median radius, eccentricity and density.
5. Save every figure as PNG and SVG into `notebooks/analysis/figures/`.

Check: run the notebook top to bottom (`jupyter nbconvert --execute`) on the
CSV from step 3 and show me the figures.

## Step 5: README + final check

- Add a short "Puncta morphology" section to `README.md`: what it computes,
  the command line, the CSV columns (one line each), and the fact that the
  plots come only from the CSV.
- Run `pytest -q` for the new tests, and run the script once more from
  scratch.
- Report what was built, which files changed, any puncta or images that were
  dropped and why, and anything you were unsure about.
