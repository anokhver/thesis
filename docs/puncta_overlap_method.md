# Puncta overlap method

## Scope

This document describes the procedure used to generate PRE/POST puncta overlap masks from the fluorescence MIPs in this repository. The implementation is in [`scripts/pseudolabels/puncta_from_mip.py`](../scripts/pseudolabels/puncta_from_mip.py), [`src/synaptic_ssl/pseudolabels/puncta_log.py`](../src/synaptic_ssl/pseudolabels/puncta_log.py), and [`src/synaptic_ssl/pseudolabels/puncta_common.py`](../src/synaptic_ssl/pseudolabels/puncta_common.py).

## Input data and channels

The input is a float32 NumPy array with shape `(C, H, W)` and values in `[0, 1]`. For the dataset used to calibrate the defaults:

| Channel | Index | Interpretation | Use |
|---|---:|---|---|
| PRE | 0 | Presynaptic fluorescence | Detect presynaptic puncta,|
| POST | 1 | Postsynaptic fluorescence | Detect postsynaptic puncta|
| Structural | 2 | Soma/dendrite signal | Build the structural gate; it is not used as a puncta channel |

The channel order and the value `107 nm/pixel` are recorded in the puncta-detection notebook.The production script exposes `--pre_channel` and `--post_channel`; its defaults are 0 and 1.

The working images are maximum-intensity projections (MIPs), so all geometry below is projected 2-D geometry. The method does not use a z-voxel size and should not be described as a 3-D overlap measurement.

## Pixel-size conversion and patch size

The working calibration is:

\[
  s = 107\ \mathrm{nm/pixel} = 0.107\ \mathrm{\mu m/pixel}.
\]

For a length `L` in micrometres, the corresponding image length is:

\[
  L_{px} = \frac{L_{\mu m}}{0.107}.
\]

For a length `L` in nanometres:

\[
  L_{px} = \frac{L_{nm}}{107}.
\]

The training/inference tiles are 128 x 128 pixels. Their physical field of view is therefore:

\[
  128 \times 0.107 = 13.696\ \mathrm{\mu m}
\]

per side, or approximately 13.7 x 13.7 micrometres. Patches are only storage/training tiles: in patch mode, the script stitches the image and structural masks, detects puncta at full-image scale, and slices the resulting masks back into 128 x 128 tiles. This avoids detecting objects independently at every tile boundary.

## Biological size priors and what they mean

The detector uses different scale ranges for the two fluorescence channels:

| Signal | Internal working prior | Pixel conversion at 107 nm/pixel | LoG sigma range | Rendered disk diameter |
|---|---:|---:|---:|---:|
| PRE bouton-like signal | 500-1000 nm diameter | 4.7-9.3 px | 1.8-3.4 px | approximately 5.1-9.6 px |
| POST PSD-like signal | 200-500 nm diameter | 1.9-4.7 px | 1.0-1.8 px | approximately 2.8-5.1 px |

The conversions are direct arithmetic. The rendered diameter follows the repository's LoG convention:

\[
  r_{px} = \sqrt{2}\,\sigma,
  \qquad
  d_{px} = 2\sqrt{2}\,\sigma.
\]

Thus, for example, the largest PRE scale gives:

\[
  d_{px} = 2\sqrt{2}(3.4) \approx 9.62\ \mathrm{px},
  \qquad
  d_{\mu m} = 9.62(0.107) \approx 1.03\ \mathrm{\mu m}.
\]

The biological nm ranges (500–1000 nm for PRE, 200–500 nm for POST) are sourced from **Harris & Stevens (1989)**, who measured these dimensions via serial electron microscopy of rat CA1 dendritic spines. These EM measurements provide the biological foundation for the detector scales.

However, the observed pixel ranges (5–10 px for PRE, 2.7–5 px for POST at 107 nm/pixel) reflect the convolution of these EM priors with an optical point-spread function (~200 nm ≈ 1.9 px). The observed pixel ranges are **internal engineering measurements from this repository's calibration**, not values independently established from the microscope or elsewhere. The LoG sigma ranges (1.8–3.4 px for PRE, 1.0–1.8 px for POST) are derived from these observed ranges and are also internal engineering choices.

The repository comments additionally describe an approximately 200 nm PSF and say that the EM priors were convolved with it. That is an **assumption in the current calibration**, not a value independently established here from the microscope's measured point-spread function. It should be reported as an approximate optical-resolution allowance unless a microscope-specific PSF measurement is available.

**Source code documentation:**
The exact EM-to-pixel mapping is documented in [`src/synaptic_ssl/pseudolabels/puncta_log.py`](../src/synaptic_ssl/pseudolabels/puncta_log.py) lines 100–101 with the Harris & Stevens reference. The per-channel configurations are calibrated in [`notebooks/pseudolabels/puncta_detection.ipynb`](../notebooks/pseudolabels/puncta_detection.ipynb) on representative dataset images.

## Per-channel detection

PRE and POST are detected independently; no channel is used to threshold the other channel.

1. Optionally apply a white top-hat filter to flatten slow background variation.
2. Run scale-space Laplacian-of-Gaussian (LoG) blob detection.
3. Estimate local background in an annulus around each candidate and calculate a robust local contrast score.
4. Keep candidates above the channel-specific z-score and absolute-intensity floors.
5. Keep only candidate centres inside the structural near-mask.
6. Render each remaining candidate as a binary disk.
7. Apply the optional connected-component size/shape filter.

The calibrated LoG settings are:

| Parameter | PRE | POST | Rationale |
|---|---:|---:|---|
| `log_min_sigma` | 1.8 px | 1.0 px | Lower end of the respective working size prior |
| `log_max_sigma` | 3.4 px | 1.8 px | Upper end of the respective working size prior |
| `log_num_sigma` | 5 | 4 | Scale samples across each interval |
| `log_overlap` | 0.5 | 0.5 | LoG candidate de-duplication in `skimage.feature.blob_log` |
| top-hat radius | 10 px | 5 px | Approximately `2*sqrt(2)*max_sigma` |
| z-score threshold | 4.0 | 6.0 | More conservative threshold for smaller POST spots |
| z-score annulus | 5/10 px | 3/6 px | Inner/outer radii scaled to the spot size |
| shape area range | 20-75 px^2 | 6-22 px^2 | Detector-scale mask prior, not a physical synapse-area measurement |

The local score is a parametric approximation:

\[
  z = \frac{\mu_{in} - \mu_{bg}}{\sigma_{bg}}.
\]

The implementation uses the 25th percentile and MAD-based scale for the annular background in the calibrated defaults. This is motivated by the difficulty of estimating background in crowded fluorescence fields. It is not the same statistical test as SynQuant: SynQuant uses an order-statistics/Wilcoxon-based significance procedure and false-discovery control. SynQuant is therefore a methodological precedent for local statistical testing, not a source for the numerical z-score thresholds used here.

## Structural near-gate

The structural mask is:

\[
  M_{struct} = M_{soma} \cup M_{dendrite}.
\]

It is dilated with a disk of radius 4 pixels:

\[
  M_{near} = \operatorname{dilate}(M_{struct}, r=4\ \mathrm{px}).
\]

At 107 nm/pixel, the dilation radius is:

\[
  4 \times 107 = 428\ \mathrm{nm} = 0.428\ \mathrm{\mu m}.
\]

A punctum is retained when its rounded centre pixel lies in `M_near`. This is a spatial plausibility filter that suppresses bright puncta in regions without a detected soma or dendrite. The 4-pixel value is a repository-defined margin, not a literature constant.

## How overlap is calculated

After the independent detections and structural gate, each channel is represented by a binary mask:

- `pre_mask`: union of all rendered PRE disks;
- `post_mask`: union of all rendered POST disks.

The masks must have identical `(H, W)` shape. The code labels connected components in each mask using 8-connectivity. For every PRE component `P_i` and POST component `Q_j`, it computes the component centroids and equivalent-circle radii:

\[
  r(P_i) = \sqrt{\frac{|P_i|}{\pi}},
  \qquad
  r(Q_j) = \sqrt{\frac{|Q_j|}{\pi}}.
\]

A component pair is called overlapping when the centroid distance is less than the sum of these radii:

\[
  \|c(P_i) - c(Q_j)\| < r(P_i) + r(Q_j).
\]

The final overlap mask is the union `P_i | Q_j` for every qualifying pair. This preserves a qualifying pair even when the rasterized masks do not share a pixel. The output also records the total number of PRE components, total number of POST components, and number of qualifying PRE/POST pairs.

### Worked interpretation

If a PRE component has 50 pixels and a POST component has 30 pixels, their equivalent radii are:

\[
  r_{PRE} = \sqrt{50/\pi},
  \qquad
  r_{POST} = \sqrt{30/\pi}.
\]

The pair qualifies when the centroid distance is less than $r_{PRE} + r_{POST}$. This is a **circular proximity rule**, not the symmetric intersection-over-union (IoU):

\[
  IoU = \frac{|P_i \cap Q_j|}{|P_i \cup Q_j|}.
\]

The current implementation uses the centre-distance circular approximation described for SynBot and its predecessor Puncta Analyzer (Savage et al., 2024). Its use here is an operational image-analysis criterion, not proof of molecular contact. The former smaller-component rule remains available in the helper as an explicit legacy option, but is no longer the default.

## Why these design choices are defensible

- **Separate channel detection:** PRE and POST fluorescence have different expected scales and intensities, so a single detector scale would privilege one signal. SynQuant also treats multiple staining channels as separate evidence while addressing heterogeneous brightness and antibody specificity.
- **LoG scale space:** LoG detection is appropriate for approximately blob-like fluorescence puncta and makes the size prior explicit through `sigma`. The repository uses the scale-normalized blob detector available through scikit-image.
- **Structural gate:** antibody signal can occur outside synapses and neurites. Restricting candidate centres to the soma/dendrite neighbourhood reduces anatomically implausible detections, but it can also remove true puncta when the structural mask is incomplete.
- **Near-complete smaller-object overlap:** PRE boutons and POST PSD-like signals need not have identical apparent footprint. Requiring 80% of the smaller connected component to be shared is therefore less restrictive than demanding equal areas, while still rejecting weak edge contacts.
- **Full-image detection before tiling:** detecting after stitching avoids artificial objects and missed detections at patch borders.
- **Circular proximity overlap:** the centre-distance criterion follows the circular approximation described for SynBot and Puncta Analyzer (Savage et al., 2024), while the equivalent-circle radius is computed from each connected-component area.

## What is and is not supported by the literature

1. **Supported biological rationale:** Harris and Stevens (1989) report that spine-head dimensions correlate with PSD area and presynaptic vesicle number in rat CA1 serial-EM reconstructions. This supports using synaptic geometry as a prior.
2. **Supported size ranges:** The 500–1000 nm PRE-bouton and 200–500 nm POST-PSD diameter ranges are from Harris & Stevens (1989), EM measurements on rat CA1 synapses. These ranges provide the biological foundation for the detector scales.
3. **Not literature-derived (EM → pixel mapping):** The observed pixel ranges (5–10 px PRE, 2.7–5 px POST) and inferred LoG sigma ranges (1.8–3.4 px PRE, 1.0–1.8 px POST) depend on the ~200 nm PSF assumption, which is a repository calibration choice. This PSF value is not independently established from the microscope's measured point-spread function and should be reported as an approximate optical-resolution allowance.
4. **Supported statistical precedent:** Wang et al. (2020) describe SynQuant, an automatic multi-channel synapse detector that uses a probability-principled order-statistics procedure and addresses heterogeneous antibody brightness, nonspecific staining, and noise. The present annular z-score is an approximation inspired by that general problem, not a reproduction of SynQuant's test.
5. **Supported spot-detection context:** Dominguez Mantes et al. (2025) describe Spotiflow for fluorescence spot detection. The repository contains an optional Spotiflow pipeline, but the overlap method documented here uses the LoG path unless explicitly run with the Spotiflow script.
6. **Not literature-derived (pipeline settings):** 107 nm/pixel, the channel indices, 128-pixel tile size, 4-pixel structural dilation, z-score thresholds, and 0.80 component-overlap threshold are dataset/pipeline settings. They require reporting as implementation parameters and, ideally, sensitivity analysis.
7. **Important limitation:** the fluorescence MIPs are diffraction-limited projections. Colocalisation in these masks is evidence of spatial agreement at the image resolution, not proof of molecular contact or ultrastructural synaptic connectivity.

## References

- Harris, K. M., & Stevens, J. K. (1989). Dendritic spines of CA1 pyramidal cells in the rat hippocampus: serial electron microscopy with reference to their biophysical characteristics. *The Journal of Neuroscience, 9*(8), 2982-2997. https://doi.org/10.1523/JNEUROSCI.09-08-02982.1989
- Wang, Y., Wang, C., Ranefall, P., Broussard, G. J., Wang, Y., Shi, G., Lyu, B., Wu, C.-T., Wang, Y., Tian, L., & Yu, G. (2020). SynQuant: an automatic tool to quantify synapses from microscopy images. *Bioinformatics, 36*(5), 1599-1606. https://doi.org/10.1093/bioinformatics/btz760
- Savage et al. (2024). SynBot and its predecessor Puncta Analyzer. bioRxiv preprint, version 4. https://www.biorxiv.org/content/10.1101/2023.06.26.546578v4.full
- Dominguez Mantes, A., et al. (2025). Spotiflow: accurate and efficient spot detection for fluorescence microscopy with deep stereographic flow regression. *Nature Methods, 22*, 1495-1504. https://doi.org/10.1038/s41592-025-02662-x
- Smal, I., Loog, M., Niessen, W. J., & Meijering, E. (2010). Quantitative comparison of spot detection methods in fluorescence microscopy. *IEEE Transactions on Medical Imaging, 29*(2), 282-301. https://doi.org/10.1109/TMI.2009.2025127

## Reproducibility checklist

Before presenting a result, record:

- input dataset/session and MIP dimensions;
- channel indices and marker identities;
- measured or metadata-derived pixel size;
- whether LoG or Spotiflow was used;
- exact PRE/POST detector configurations;
- structural-mask source and `near_dilate_px`;
- whether the shape filter was enabled;
- the overlap rule (centroid distance `<` sum of equivalent-circle radii; legacy rule, if used: `min(|P_i|, |Q_j|)`, 0.80);
- counts of PRE components, POST components, and qualifying pairs;
- a visual overlay of PRE, POST, and overlap masks on representative images;
- sensitivity of the overlap count to at least nearby thresholds, such as 0.70, 0.80, and 0.90.
