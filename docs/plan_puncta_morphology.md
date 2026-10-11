# Plan: puncta density and shape analysis

Goal: for every image, count the PRE and POST puncta, normalise the count by
dendrite area, describe each punctum's shape with an ellipse fitted from
moments, and export everything to one table, split into the 24 h and 1 h
groups.

Existing building blocks:

| Need | Already in repo |
|---|---|
| Puncta centres + LoG scale | `pseudolabels/puncta_log.py` → `detect_puncta_channel()` returns `(row, col, sigma)` |
| Model-based centres (alternative) | `segmentation/colocalisation.py` → `_prob_to_points()` on the 2-ch probability map |
| Dendrite mask | `pseudolabels/dendrite_frangi.py` → `run_dendrite_on_image()` |
| Restrict puncta to dendrite | `pseudolabels/puncta_common.py` → `restrict_puncta_to_near()` |
| Group / time-point from filename | `clustering/groups.py` + `configs/clustering/group_patterns.json` (`24h`, `1h pulse`) |
| Pixel size (µm/px) | `utils_metadata/extractors.py` (microscope metadata) |

New module: `src/synaptic_ssl/analysis/puncta_morphology.py`, plus a CLI
script `scripts/analysis/puncta_morphology.py`.

---

## 1. Number of puncta per dendrite area

For each image and each channel (PRE, POST):

1. Build the dendrite mask (`run_dendrite_on_image`).
2. Detect puncta (LoG or model peaks) and keep only those on or near the
   dendrite (`restrict_puncta_to_near`).
3. `area_px = mask.sum()`; `area_um2 = area_px * px_size_um**2`.
4. `density = n_puncta / area_um2` (puncta / µm²). Also report a per-length
   variant (`n / skeleton_length_um`), because dendrite width differs between
   images and per-length density is the usual metric in the literature.

Output: one row per image × channel (`file_name, group, channel, n,
area_um2, density`).

## 2. Histogram of the ranges

Plot the distributions of the per-punctum quantities from step 4 (radius,
eccentricity, semi-axes) and of the per-image density from step 1:

- one histogram per quantity, PRE and POST overlaid;
- shared bin edges across groups so the groups can be compared;
- print the range (min / max / 5th–95th percentile) beside each histogram.

> Open question: "ranges" here is read as the value ranges of radius and
> eccentricity. If it means something else (for example the PRE–POST
> distance range), only this section changes.

## 3. Separate 24 h and 1 h

- Parse each filename with `classify_image_by_patterns()`; the label already
  holds the time qualifier (`| 24h`, `| 1h pulse`).
- Add a `time_group` column with values `24h`, `1h`, and `other` (48h, 72h,
  or unlabelled; these are kept in the CSV but left out of the comparison
  plots).
- Make every plot and summary table from steps 1, 2 and 4 once per
  `time_group`, and inside each time group per treatment.
- Test the difference between groups with Mann–Whitney U on the per-image
  medians (not per-punctum values, which are not independent).

## 4. Ellipse from moments (instead of the LoG radius)

The LoG `sigma` is used only to find the centre. The shape is then measured
in a local window around that centre:

1. Crop a window of half-size `k * sigma` (for example `k = 3`) around
   `(row, col)` from the background-subtracted (tophat) image.
2. Weights `w(x, y)` = intensity minus the local background (annulus median,
   as in `_score_one_puncta_zscore`), clipped at 0. Optionally restrict them
   to the connected component that contains the centre.
3. Moments:
   - `m00 = Σw`
   - centroid `x̄ = Σw·x / m00`, `ȳ = Σw·y / m00` (refined centre)
   - central second moments `μ20 = Σw(x−x̄)² / m00`, `μ02 = Σw(y−ȳ)² / m00`,
     `μ11 = Σw(x−x̄)(y−ȳ) / m00`
4. Covariance `C = [[μ20, μ11], [μ11, μ02]]`; eigenvalues `λ1 ≥ λ2` and
   eigenvectors give the axes:
   - semi-major axis `a = 2·√λ1`, semi-minor axis `b = 2·√λ2` (the 2σ
     ellipse)
   - orientation `θ = ½·atan2(2μ11, μ20 − μ02)`
   - equivalent radius `r = √(a·b)` (in px, then × `px_size_um` for µm)
   - eccentricity `e = √(1 − λ2/λ1)` (0 = circle, close to 1 = elongated)
5. Continuous distribution: instead of only histograms, fit a KDE
   (`scipy.stats.gaussian_kde`) to `r` and `e` per group, and optionally a
   parametric fit (log-normal for `r`, beta for `e`). Compare the groups with
   a KS test.
6. Sanity checks: drop puncta whose window touches the image border or has
   `m00 ≈ 0`. On a few images, overlay the ellipses on the raw image to check
   them by eye (reuse `pseudolabels/viz.py`).

## 5. Output table

One CSV row per punctum, `;`-separated:

```
file_name;group_name;time_group;channel;x;y;radius_um;a_um;b_um;eccentricity;theta_deg;intensity;density_per_um2
```

- `channel` ∈ {`pre`, `post`}. One row per punctum, with the channel in its
  own column, is easier to filter than separate `pre`/`post` columns. If the
  layout should be one row per **matched synapse**, add
  `pre_x;pre_y;post_x;post_y` using the PRE↔POST matching in
  `colocalisation.py`.
- `x`, `y` = refined moment centroid (px, full-image coordinates).
- Also write a per-image summary CSV: `file_name;group_name;time_group;
  n_pre;n_post;dendrite_area_um2;density_pre;density_post;median_radius_pre;
  median_radius_post;median_ecc_pre;median_ecc_post`.

---

## Implementation order

1. `ellipse_from_moments(image, row, col, sigma, k)` → dict. Unit-test it on
   synthetic Gaussians with known σx, σy and rotation.
2. `analyse_image(path, cfg)` → per-punctum DataFrame plus per-image summary
   (steps 1, 4, 5).
3. CLI script that loops over the MIPs, adds `group_name` / `time_group`, and
   writes both CSVs.
4. Notebook `notebooks/analysis/puncta_morphology.ipynb`: histograms, KDEs,
   24 h vs 1 h comparison (steps 2, 3).

## Open questions for the supervisor

- Which detector should give the centres: LoG pseudolabels or the trained
  model's peaks?
- Should density be normalised by area (as asked) or by length, or both?
- Does "1 h" mean the `1h pulse` washout images, or a separate 1 h exposure
  set?
- Should the table have one row per punctum or one row per matched synapse
  (PRE + POST)?
