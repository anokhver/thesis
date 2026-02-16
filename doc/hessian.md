# Hessian-Based Filtering for Neural Structure Enhancement

## Overview

This module implements **Hessian-based image filtering** for enhancing neural structures — dendrites, axons, and soma — in confocal microscopy images. The filtering serves as a preprocessing and feature-extraction step within a larger self-supervised segmentation pipeline.

The dataset comprises over a thousand 3D confocal microscope images of neural cell cultures treated with various substances. The goal is to measure neuroplasticity effects by quantifying synaptic structures. Hessian filtering highlights tubular and blob-like structures in noisy microscopy data, producing enhanced images and binary masks that can be used as pseudo-labels for downstream deep learning segmentation — reducing dependence on expensive manual annotations.

Two filtering approaches are implemented:

1. **Multi-scale Frangi vesselness filter** — the classical approach from Frangi et al. (1998), applied at multiple Gaussian scales.
2. **DDeep3M+ adaptive filter** — an adaptive method that selects the smoothing scale per-pixel using the Euclidean distance transform, with Lindeberg scale normalization.

---

## Theoretical Background

### The Hessian Matrix

The **Hessian matrix** of a 2D image $I(x, y)$ is the matrix of second-order partial derivatives:

$$
H = \begin{bmatrix} \dfrac{\partial^2 I}{\partial x^2} & \dfrac{\partial^2 I}{\partial x \, \partial y} \\[6pt] \dfrac{\partial^2 I}{\partial y \, \partial x} & \dfrac{\partial^2 I}{\partial y^2} \end{bmatrix}
$$

The Hessian captures **second-order intensity variations** — i.e., the curvature of the image intensity surface. Intuitively, it encodes the *acceleration of change in all directions* at every pixel. A large negative second derivative along a direction means the intensity is at a ridge (peak) along that axis; a value near zero means the intensity is approximately flat.

In practice, computing raw second derivatives on discrete, noisy images is unstable. Instead, the image is first convolved with **Gaussian derivative filters**:

$$
I_{xx} = G_{\sigma,xx} * I, \quad I_{xy} = G_{\sigma,xy} * I, \quad I_{yy} = G_{\sigma,yy} * I
$$

where $G_{\sigma}$ is a Gaussian kernel with standard deviation $\sigma$. The parameter $\sigma$ controls the scale of structures that are enhanced — it acts as a *radius of influence* determining how far away neighboring pixels contribute to each derivative estimate.

### Eigenvalue Analysis

For a 2D symmetric Hessian, the **eigenvalues** $\lambda_1, \lambda_2$ represent the principal curvatures of the intensity surface. They can be computed analytically for a $2 \times 2$ symmetric matrix:

$$
\lambda = \frac{\text{trace}(H) \pm \sqrt{\text{trace}(H)^2 - 4 \cdot \det(H)}}{2}
$$

where $\text{trace}(H) = I_{xx} + I_{yy}$ and $\det(H) = I_{xx} I_{yy} - I_{xy}^2$.

By convention, eigenvalues are sorted so that $|\lambda_1| \leq |\lambda_2|$. The pattern of eigenvalue magnitudes and signs encodes the local geometry:

| Structure   | $\lambda_1$ | $\lambda_2$ | $\lambda_3$ (3D) | Geometric Interpretation                       |
|-------------|-------------|-------------|-------------------|-------------------------------------------------|
| Background  | $\approx 0$ | $\approx 0$ | $\approx 0$       | No curvature (flat / noise)                     |
| Neurite     | $\approx 0$ | $\ll 0$     | $\ll 0$           | Tubular (1 flat + 2 curved directions)          |
| Soma        | $\ll 0$     | $\ll 0$     | $\ll 0$           | Blob-like (curved in all directions)            |
| Plate/Sheet | $\approx 0$ | $\approx 0$ | $\ll 0$           | Planar (curved in only 1 direction)             |

For **bright structures on a dark background**, negative eigenvalues indicate concavity — a bright ridge or blob. Positive eigenvalues would indicate a dark valley, which is typically suppressed.

### Frangi Vesselness Filter

> **Reference:** Frangi, A. F., et al. (1998). *"Multiscale vessel enhancement filtering."* MICCAI.  
> See [`papers/BFb0056195.pdf`](../papers/BFb0056195.pdf)

The **Frangi vesselness function** in 2D combines the eigenvalues into a single scalar response that is high for elongated tubular structures and low for blobs or background:

$$
\mathcal{V}(\sigma) = \exp\!\left(-\frac{R_B^2}{2\beta^2}\right) \cdot \left(1 - \exp\!\left(-\frac{S^2}{2c^2}\right)\right)
$$

where:

- $R_B = \dfrac{|\lambda_1|}{|\lambda_2|}$ — the **blobness ratio**. Measures deviation from a purely tubular shape. For an ideal tube, $\lambda_1 \approx 0$ so $R_B \approx 0$. For a blob, $\lambda_1 \approx \lambda_2$ so $R_B \approx 1$.

- $S = \sqrt{\lambda_1^2 + \lambda_2^2}$ — the **structureness**. The Frobenius norm of the eigenvalues, distinguishing actual structures from flat background. Low $S$ = background noise.

- $\beta$ — controls sensitivity to $R_B$ (blob suppression). Lower $\beta$ = more aggressively suppress blob-like responses.

- $c$ — controls sensitivity to $S$ (background suppression). Should be tuned to the intensity/noise range of the image. Too small → overly sensitive to noise; too large → suppresses weak structures.

The filter is set to **zero** wherever the eigenvalue signs do not match the expected structure polarity. For bright tubular structures on a dark background, $\lambda_2$ must be negative; otherwise the response is zeroed out.

**Intuition:** The first exponential term selects elongated (non-blob) structures via $R_B$. The second term rejects featureless background via $S$. Together, they produce a response that peaks on tubular ridges.

### Multi-Scale Analysis

Neural structures span a wide range of widths — from thin axons (sub-micron) to thick dendrites (several microns). A single $\sigma$ cannot capture all of them simultaneously:

- **Small $\sigma$**: detects thin axons (sharp features survive smoothing), but is sensitive to noise.
- **Large $\sigma$**: detects thick dendrites (fine noise is smoothed away), but blurs thin structures.

**Lindeberg's scale-space theory** provides the solution: multiply the Hessian by $\sigma^2$ for **scale normalization**, making the filter response comparable across different scales:

$$
H_{\text{norm}}(\sigma) = \sigma^2 \cdot H(\sigma)
$$

The **multi-scale approach** then computes the vesselness at each scale $\sigma_i$ and takes the **maximum response** across all scales:

$$
\mathcal{V}_{\text{final}} = \max_{\sigma \in \{\sigma_1, \dots, \sigma_n\}} \mathcal{V}(\sigma)
$$

This ensures that each structure is detected at its optimal scale.

### Adaptive Scale Selection (DDeep3M+ Approach)

Instead of evaluating a fixed set of scales and taking the maximum, the **DDeep3M+ method** adapts $\sigma$ per-pixel based on the local structure size:

1. **Binarize** the image using Otsu's threshold → foreground mask.
2. **Euclidean Distance Transform (DT)**: each foreground pixel receives its distance to the nearest background pixel. This estimates the local half-width of the structure.
3. **Normalize** the DT to $[1, 256]$, then compute $\sigma = \text{clip}(\log_2(\text{DT}_{\text{norm}}),\, 1,\, 8)$.
4. **Discretize** $\sigma$ to integer scales and compute the Hessian for each unique scale.
5. Each pixel receives the filter response from **its own adaptive scale**.

**Soma hole-filling:** The Frangi filter produces ring-like responses around soma (round structures, where $R_B \approx 1$). High DT values indicate soma centers. These regions are filled with the maximum surrounding response to avoid hollow soma masks.

**Advantage:** Automatically adapts to varying structure widths within a single image, without requiring a predefined set of scales.

---

## Accuracy Evaluation Methods

### Common Metrics

- **Sensitivity (True Positive Rate / Recall):** Proportion of actual structures correctly detected.
  $$\text{Sensitivity} = \frac{TP}{TP + FN}$$

- **Specificity (True Negative Rate):** Proportion of background correctly classified.
  $$\text{Specificity} = \frac{TN}{TN + FP}$$

- **Precision (Positive Predictive Value):** Fraction of detected structures that are truly structures.
  $$\text{Precision} = \frac{TP}{TP + FP}$$

- **Dice Coefficient (F1 Score):** Overlap between predicted and ground truth segmentation.
  $$\text{Dice} = \frac{2 \, |A \cap B|}{|A| + |B|}$$

- **Jaccard Index (IoU):** Intersection over union.
  $$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

- **ROC Curves:** Plot sensitivity vs. $(1 - \text{specificity})$ at varying thresholds. The **Area Under Curve (AUC)** provides a threshold-independent performance measure.

### Ground Truth and Benchmarks

- Common retinal vessel benchmarks: **DRIVE**, **STARE**, **CHASE_DB1** datasets.
- Challenge datasets: **Cell Segmentation Challenge**, **BigNeuron** project.

### Validation Approaches in the Literature

- **Frangi et al.** validated on synthetic images with known geometry and clinical angiography data.
- **Retinal vessel papers** use pixel-wise comparison against expert segmentations, reporting sensitivity / specificity / AUC.
- **Neuroscience papers** compare against manual tracings of neuronal processes.
- **Parameter sensitivity analysis**: systematic variation of $\beta$, $c$, and $\sigma$ ranges to demonstrate robustness.
- **Multi-scale vs. single-scale comparison**: quantifying the improvement from multi-scale integration.
- **Comparative studies**: Frangi vs. Sato vs. Meijering filters, each with different eigenvalue formulations.

### Leveraging Filtering for Downstream Segmentation

- **Preprocessing**: Hessian filtering enhances structures before thresholding or ML-based segmentation.
- **Feature channels**: Frangi response, eigenvalue maps, and scale maps as additional input features to neural networks.
- **Self-supervised learning**: Using Frangi-generated masks as **pseudo-labels** for training when manual annotations are unavailable.
- **Attention guidance**: Using vesselness maps to guide attention mechanisms in deep networks.
- **Post-processing**: Combining deep learning predictions with Hessian-based structure analysis for refinement.

---

## References

1. Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998). *"Multiscale vessel enhancement filtering."* In MICCAI, LNCS 1496, pp. 130–137. — [`papers/BFb0056195.pdf`](../papers/BFb0056195.pdf)

2. Lindeberg, T. (1994). *"Scale-Space Theory in Computer Vision."* Springer.

3. Sato, Y., Nakajima, S., Shiraga, N., et al. (1998). *"Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medical images."* Medical Image Analysis, 2(2), 143–168.

4. Meijering, E., Jacob, M., Sarria, J.-C. F., Steiner, P., Hirling, H., & Unser, M. (2004). *"Design and validation of a tool for neurite tracing and analysis in fluorescence microscopy images."* Cytometry Part A, 58A(2), 167–176.

5. Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). *"Cellpose: a generalist algorithm for cellular segmentation."* Nature Methods, 18, 100–106. — [`papers/stringer2020.pdf`](../papers/stringer2020.pdf)

6. DDeep3M / DDeep3M+ methodology — deep learning enhanced microscopy segmentation with adaptive Hessian-based preprocessing and distance transform-guided sigma selection.

7. Hessian application papers:
   - [`electronics-12-04159-v3.pdf`](../papers/hessian/electronics-12-04159-v3.pdf) — Electronics journal paper on Hessian methods
   - [`sensors-22-04956-v3.pdf`](../papers/hessian/sensors-22-04956-v3.pdf) — Sensors journal paper on Hessian-based approaches
   - [`s11760-024-03724-x.pdf`](../papers/hessian/s11760-024-03724-x.pdf) — Signal, Image and Video Processing paper
   - [`Research_on_Image_Quality_Enhancement_Algorithm_Us.pdf`](../papers/hessian/Research_on_Image_Quality_Enhancement_Algorithm_Us.pdf) — Image quality enhancement using Hessian
   - [`TSP_JNM_27060.pdf`](../papers/hessian/TSP_JNM_27060.pdf) — Hessian-based filtering in biomedical imaging
   - [`application-of-a-hessian-based-image-processing-method-for-enhanced-visualization-of-nanoscale-rubber-cross-linked.pdf`](../papers/hessian/application-of-a-hessian-based-image-processing-method-for-enhanced-visualization-of-nanoscale-rubber-cross-linked.pdf) — Hessian for nanoscale visualization

8. Additional neuroscience / segmentation references:
   - [`fnana-14-00038.pdf`](../papers/fnana-14-00038.pdf) — Frontiers in Neuroanatomy
   - [`elife-99848-v1.pdf`](../papers/elife-99848-v1.pdf) — eLife neuroscience paper
   - [`srep11576.pdf`](../papers/srep11576.pdf) — Scientific Reports
   - [`The_Multi-modality_Cell _Segmentation_Challenge.pdf`](../papers/The_Multi-modality_Cell%20_Segmentation_Challenge.pdf) — Multi-modality Cell Segmentation Challenge
