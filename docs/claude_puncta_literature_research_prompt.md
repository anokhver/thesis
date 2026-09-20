# Claude prompt: literature-grounded puncta size and overlap criteria

Copy the prompt below into Claude with web search enabled.

---

You are a microscopy-methods literature researcher helping prepare a defensible neuroscience thesis method. Search the peer-reviewed literature and primary sources on the web. Do not assume that a value currently used in code is literature-supported. Your task is to determine which numerical ranges and overlap criteria are justified for detecting and declaring colocalisation of presynaptic (PRE) and postsynaptic (POST) puncta in fluorescence microscopy images.

## Research question

Find literature-supported ranges for:

1. The apparent diameter, area, or equivalent radius of presynaptic puncta/boutons corresponding to markers such as synapsin, synaptophysin, bassoon, or related presynaptic proteins.
2. The apparent diameter, area, or equivalent radius of postsynaptic puncta/PSDs corresponding to markers such as PSD-95, Homer, Shank, or related postsynaptic proteins.
3. Appropriate object-based criteria for deciding whether a PRE and POST punctum overlap or represent a putative synapse/colocalised pair.
4. How these values change with imaging modality, optical resolution, marker, preparation, tissue, magnification, pixel size, z sampling, and 2D projection versus 3D analysis.

The final recommendation must distinguish:

- biological ultrastructural dimensions measured by electron microscopy;
- apparent immunofluorescence punctum dimensions after optical blur and antibody labeling;
- detector parameters such as LoG sigma or rendered-mask diameter;
- colocalisation/overlap criteria used for analysis.

Do not convert an EM diameter directly into a fluorescence spot diameter without explaining the point-spread function, labeling geometry, sampling, and projection effects.

## Repository and acquisition context

These are the facts currently available from the repository. Treat them as experimental context, not as proof that the literature uses the same values:

- The input is fluorescence microscopy from multichannel z-stack acquisitions.
- Raw acquisition files include `.vsi` and `.oex` files.
- Representative metadata report 3 channels named `C488`, `C561`, and `C640`.
- Representative raw image dimensions are 2304 x 2304 pixels, with either 23 or 43 z-planes in the sessions inspected.
- Recorded lateral pixel size is approximately 0.106854 µm/pixel, reported in the working method as 107 nm/pixel.
- The acquisition metadata report a 60x objective and numerical aperture approximately 1.42. One code comment says 60x, NA 1.4, and 488 nm excitation; verify exact channel wavelengths and microscope settings from metadata before presenting them as definitive.
- Metadata identify a CSU/Free objective/model field and a Hamamatsu ORCA-Fusion multi-camera detector. Do not infer the complete optical modality from these fields alone; determine whether the acquisition was spinning-disk confocal, widefield, or another modality, and label uncertainty if the evidence is insufficient.
- The channel emission metadata include approximately 508, 580, and 670 nm for the three channels; excitation metadata are incomplete or null in the inspected OME records. Do not invent excitation wavelengths.
- The analysis uses 2D maximum-intensity projections (MIPs) of the z-stacks. The detector therefore measures projected 2D mask geometry, not true 3D object volume or 3D molecular contact.
- The working channel mapping is: channel 0 = PRE/presynaptic fluorescence, channel 1 = POST/postsynaptic fluorescence, channel 2 = structural soma/dendrite signal. Verify whether the biological marker identity behind each channel is documented elsewhere before claiming a specific marker.
- Images are stored as float32 arrays normalized to [0, 1] after preprocessing. The source acquisition metadata include uint16 data and 16-bit acquisition fields. Keep raw bit depth separate from normalized analysis values.
- The standard training/storage tile is 128 x 128 pixels, approximately 13.7 x 13.7 µm at 107 nm/pixel. This tile size is a computational choice, not a biological punctum size.

## Current implementation to audit

The code currently uses these internal working values:

- PRE working prior: 500-1000 nm diameter.
- POST working prior: 200-500 nm diameter.
- PRE LoG sigma: 1.8-3.4 pixels.
- POST LoG sigma: 1.0-1.8 pixels.
- Approximate PSF allowance in comments: 200 nm, or about 1.9 pixels at 107 nm/pixel.
- Rendered binary-disk diameter follows `d = 2 sqrt(2) sigma`, giving approximately 5.1-9.6 pixels for PRE and 2.8-5.1 pixels for POST.
- Candidate centres are restricted to a structural soma/dendrite mask dilated by 4 pixels.
- PRE and POST masks are labeled with 8-connectivity.
- For every PRE component `P_i` and POST component `Q_j`, the implementation computes equivalent-circle radii `r = sqrt(area / pi)` and centroid distance.
- A pair is currently called overlapping when `distance(centroid_PRE, centroid_POST) < radius_PRE + radius_POST`, following the circular approximation described for SynBot/Puncta Analyzer (Savage et al. 2024).
- The output overlap mask is the union of qualifying component pairs, so a pair can qualify even when rasterized pixels do not directly intersect.
- The previous rule, `|P_i intersection Q_j| / min(|P_i|, |Q_j|) >= 0.80`, is retained only as an explicit legacy mode.
- The current visualization uses lime for PRE contours and cyan for POST contours.

Audit every one of these values. Say explicitly whether each value is:

1. directly supported by a primary paper;
2. indirectly motivated by a paper but adapted by this project;
3. derived from the acquisition's measured pixel size or optical model; or
4. an internal engineering choice requiring calibration or sensitivity analysis.

## Search strategy

Prioritize, in this order:

1. Primary quantitative studies measuring synaptic puncta or synapse-marker spot dimensions with fluorescence microscopy.
2. Studies that compare fluorescence puncta to EM or super-resolution measurements in the same preparation.
3. Validated synapse-detection and colocalisation methods, including object-based methods.
4. Methodological reviews only when they lead to the primary measurement or clearly summarize a consensus.
5. General spot-detection papers for detector scale and PSF limitations.

Search combinations of terms such as:

- presynaptic puncta diameter synapsin fluorescence microscopy
- synaptophysin bouton size immunofluorescence nanometer
- bassoon puncta size confocal hippocampus
- postsynaptic density PSD-95 puncta diameter fluorescence microscopy
- Homer PSD puncta size confocal
- synaptic puncta colocalization object based overlap threshold
- synapse detection PRE POST mask intersection over union
- synaptic colocalization Manders object based distance criterion
- fluorescence puncta size versus electron microscopy PSD bouton
- spinning disk confocal synapse puncta pixel size z-stack

Include relevant literature on SynQuant, Spotiflow, spot-detection benchmarks, and established colocalisation methods, but do not treat these papers as evidence for a numerical value unless they actually report or justify that value.

## Requirements for each source

For every candidate source, report:

- full citation and DOI or stable publisher/PubMed link;
- whether it is primary research, a methods paper, or a review;
- tissue/preparation and species;
- marker identity and whether it is pre- or postsynaptic;
- imaging modality, objective, NA, excitation/emission if reported, pixel size, z-step, and whether the image is a stack, single plane, MIP, or deconvolved image;
- the measured quantity: diameter, area, radius, FWHM, centroid distance, nearest-neighbor distance, or another definition;
- the numerical result, including units and uncertainty or distribution when available;
- whether the number is biological structure size or apparent fluorescence size;
- the exact page, figure, table, supplement, or quoted passage supporting the number;
- limitations and whether the result is transferable to the acquisition described above.

Never cite a paper merely because it discusses synapses generally. A source must support the specific numerical claim being made. If the paper does not provide an exact range, say so.

## Special audit of the Harris & Stevens citation

The current code comments cite Harris & Stevens (1989), *Dendritic spines of CA1 pyramidal cells in the rat hippocampus: serial electron microscopy with reference to their biophysical characteristics*, The Journal of Neuroscience 9(8):2982-2997, DOI 10.1523/JNEUROSCI.09-08-02982.1989.

Check the original paper directly and answer:

- Does it actually report a 500-1000 nm presynaptic bouton diameter range?
- Does it actually report a 200-500 nm PSD diameter range?
- Does it measure diameters, areas, lengths, or another geometry?
- Are those values for boutons and PSDs, or are they an extrapolation from spine-head/PSD measurements?
- If the current code overstates what the paper supports, provide corrected wording and identify a better primary source.

Do not preserve the citation just because it already appears in the repository.

## Overlap and colocalisation analysis

Compare at least these criteria:

- intersection over union (IoU);
- intersection divided by the smaller object area, as used here;
- fraction of PRE covered by POST;
- fraction of POST covered by PRE;
- centroid-to-centroid distance normalized by object size;
- nearest-boundary or nearest-centre distance;
- Manders coefficients or pixel-intensity colocalisation, explaining why they may be inappropriate for object pairing;
- 3D object overlap or distance criteria where available.

For each criterion, explain what it means biologically and mathematically, whether it is symmetric, how it behaves when PRE and POST have different apparent sizes, and what false positives/false negatives it creates. Find published thresholds where possible, but do not generalize a threshold from a different modality or marker without qualification.

Specifically evaluate whether the implemented circular rule is defensible for projected 2D PRE/POST masks, including the use of connected-component centroids and equivalent-circle radii rather than original detector-blob centres and radii. Compare it with the previous rule, `intersection / min(area_PRE, area_POST) >= 0.80`; if no literature supports a numerical threshold directly, state that clearly and recommend empirical validation using manual annotation, null-model controls, and sensitivity analysis.

Also discuss a key implementation issue: connected components are formed from the union of rendered disks before pairing. Explain how touching puncta, merged objects, MIP projection, and different PRE/POST spot sizes can bias component overlap statistics.

## Required final output

Produce a citation-ready report with these sections:

1. **Executive conclusion**: the most defensible literature-supported PRE and POST apparent-size ranges for this imaging context, with confidence and caveats.
2. **Acquisition comparability**: a table comparing each important paper with this experiment's 60x, NA ~1.42, 107 nm/pixel, multichannel z-stack, MIP workflow.
3. **Evidence table for punctum sizes**: separate EM/ultrastructural dimensions from fluorescence measurements; include marker, modality, measurement definition, range, and exact citation location.
4. **Evidence table for overlap criteria**: formula, threshold, object/pixel basis, symmetry, source, and relevance.
5. **Audit of the current implementation**: classify every current numeric value as literature-derived, measured metadata, derived optical conversion, or internal engineering choice.
6. **Corrected recommendation**: propose PRE and POST detector ranges in physical units and pixels, with a transparent conversion to LoG sigma and an explanation of PSF assumptions.
7. **Recommended overlap rule**: recommend a primary criterion and sensitivity-analysis range, or explain why no universal threshold exists.
8. **What must be measured locally**: list PSF calibration, antibody/marker identity, z-step, acquisition modality, deconvolution, segmentation quality, and manual ground truth requirements.
9. **Citation-ready methods paragraph**: write a concise paragraph suitable for a thesis, with citations attached only to claims the sources actually support.
10. **BibTeX entries** for every cited source.

## Strict evidence rules

- Use primary sources and provide DOI or stable links.
- Quote or precisely locate the passage supporting every numerical range and threshold.
- Do not call an EM measurement an immunofluorescence diameter.
- Do not call a detector sigma a biological diameter.
- Do not treat the 200 nm PSF assumption as a measured property of this microscope.
- Do not invent missing acquisition details such as excitation wavelengths, z-step, pinhole, laser power, exposure time, deconvolution, or staining protocol.
- Label all extrapolations and internal choices explicitly.
- If evidence is absent or contradictory, say “no direct evidence found” and recommend a validation experiment rather than manufacturing a consensus.
- Prefer a narrower defensible claim over a broad unsupported range.
