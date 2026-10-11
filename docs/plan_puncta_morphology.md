# Plan: puncta density and shape analysis

Goal: for every image, detect the PRE and POST puncta with LoG, measure each
punctum's shape with an ellipse fitted from moments, and write one CSV with
one row per punctum. Every histogram and density value is computed from that
CSV alone.

Existing building blocks:

| Need | Already in repo |
|---|---|
| Puncta centres + LoG scale | `pseudolabels/puncta_log.py` → `detect_puncta_channel()` returns `(row, col, sigma)` |
| Dendrite mask | `pseudolabels/dendrite_frangi.py` → `run_dendrite_on_image()` |
| Restrict puncta to dendrite | `pseudolabels/puncta_common.py` → `restrict_puncta_to_near()` |
| Group name from filename | `clustering/groups.py` + `configs/clustering/group_patterns.json` |
| Pixel size (µm/px) | `utils_metadata/extractors.py` (microscope metadata) |

New code:
- `src/synaptic_ssl/analysis/puncta_morphology.py`: writes the CSV.
- `scripts/analysis/puncta_morphology.py`: CLI that loops over the images.
- `notebooks/analysis/puncta_histograms.ipynb`: reads **only** the CSV and
  draws the plots.

---

## 1. Puncta per dendrite area

For each image and each channel (PRE, POST; the channel indices are CLI
arguments `--pre_channel` / `--post_channel`):

1. Build the dendrite mask (`run_dendrite_on_image`).
2. Detect puncta with LoG (`detect_puncta_channel`) and keep only those on
   the dendrite (`restrict_puncta_to_near`).
3. `dendrite_area_um2 = mask.sum() * px_size_um**2`.
4. `density_per_um2 = n_puncta / dendrite_area_um2`.

`dendrite_area_um2` is written on every row of the CSV, so the density can be
recomputed from the CSV: group by `file_name` and `role`, count the rows,
and divide by `dendrite_area_um2`.

## 2. Histograms of the value ranges

All plots are made from the CSV:

- radius, eccentricity, `a`, `b`: one value per row;
- density: one value per (`file_name`, `role`), computed as in step 1.

Each plot overlays PRE and POST, uses the same bin edges for every group, and
lists min / max and the 5th–95th percentile.

## 3. 24 h and 1 h

The script does not decide which group an image belongs to. It writes
`file_name` (and `group_name` from `classify_image_by_patterns()`, which
already contains `24h` or `1h pulse` when the filename has it) into the CSV.
The 24 h / 1 h split is then a filter on those columns in the notebook, so it
can be changed without re-running the analysis.

## 4. Ellipse from moments

LoG gives only the centre. The shape is measured in a local window around it:

1. Crop a window of half-size `k * sigma` (for example `k = 3`) around
   `(row, col)` from the background-subtracted (tophat) image.
2. Weights `w(x, y)` = intensity minus the local background (annulus median,
   as in `_score_one_puncta_zscore`), clipped at 0.
3. Moments:
   - `m00 = Σw`
   - centroid `x̄ = Σw·x / m00`, `ȳ = Σw·y / m00` (refined centre)
   - `μ20 = Σw(x−x̄)² / m00`, `μ02 = Σw(y−ȳ)² / m00`,
     `μ11 = Σw(x−x̄)(y−ȳ) / m00`
4. Covariance `C = [[μ20, μ11], [μ11, μ02]]`, eigenvalues `λ1 ≥ λ2`:
   - semi-axes `a = 2·√λ1`, `b = 2·√λ2`
   - orientation `θ = ½·atan2(2μ11, μ20 − μ02)`
   - radius `r = √(a·b)`
   - eccentricity `e = √(1 − λ2/λ1)` (0 = circle, close to 1 = elongated)
5. Continuous distribution: a smooth density curve drawn over each histogram,
   computed from the radius and eccentricity columns of the CSV
   (`scipy.stats.gaussian_kde`). It is the same data as the histogram, but
   the curve does not depend on the chosen bin width.
6. Drop puncta whose window touches the image border or has `m00 ≈ 0`. On a
   few images, overlay the ellipses on the raw image to check them by eye.

## 5. CSV

One row per punctum, `;`-separated:

```
file_name;group_name;role;channel_index;x;y;radius_um;a_um;b_um;eccentricity;theta_deg;intensity;dendrite_area_um2
```

- `role` is `pre` or `post`, written out explicitly on every row.
- `channel_index` is the index of the channel in the image stack that the
  punctum was detected in (the `--pre_channel` / `--post_channel` value, as in
  `scripts/pseudolabels/puncta_from_mip.py`), so a row can be traced back to
  the raw data even if the channel order differs between experiments.
- `x`, `y` are the refined moment centroid in px (full-image coordinates).
- `radius_um`, `a_um`, `b_um` are converted with the pixel size from the
  metadata.

---

## Implementation order

1. `ellipse_from_moments(image, row, col, sigma, k)` → dict. Unit-test it on
   synthetic Gaussians with known σx, σy and rotation.
2. `analyse_image(path, cfg)` → per-punctum DataFrame (steps 1, 4).
3. CLI script: loop over the MIPs and write the CSV (step 5).
4. Notebook: read the CSV, filter 24 h / 1 h, draw histograms and density
   curves (steps 2, 3).
