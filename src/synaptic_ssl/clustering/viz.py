"""Visualisations for the clustering pipeline notebook."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def plot_singular_value_spectrum(S, ax=None, mark_target_var: float = 0.95):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    s2 = (S ** 2) / (S ** 2).sum().clip(min=1e-12)
    cum = np.cumsum(s2)
    ax.plot(np.arange(1, len(cum) + 1), cum, "o-", ms=3)
    ax.axhline(mark_target_var, color="red", ls="--",
               label=f"{mark_target_var*100:.0f}% var")
    d_target = int(np.searchsorted(cum, mark_target_var) + 1)
    ax.axvline(d_target, color="red", ls=":", alpha=0.6)
    ax.set_xlabel("rank")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"Singular-value spectrum  (d@{int(mark_target_var*100)}%var = {d_target})")
    ax.grid(alpha=0.3); ax.legend()
    return ax


def plot_resolution_stability(summary: dict, full: dict, ax=None):
    """Plot bootstrap stability across resolutions.

    ``summary[r] = (mean_ari, std_ari, mean_nmi, std_nmi)`` (legacy
    2-tuple is also accepted, in which case only ARI is plotted);
    ``full[r] = labels``. ARI (Lange et al. 2004; Hubert & Arabie 1985)
    is the primary curve used for resolution selection; NMI (Strehl &
    Ghosh 2002) is overlaid as a complementary clustering-similarity
    measure that is invariant to cluster relabelling.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    rs = sorted(summary.keys())
    has_nmi = len(next(iter(summary.values()))) >= 4
    means_ari = [summary[r][0] for r in rs]
    stds_ari  = [summary[r][1] for r in rs]
    ks    = [len(set(full[r])) for r in rs]
    ax.errorbar(rs, means_ari, yerr=stds_ari, fmt="o-", capsize=3, color="C0",
                label="bootstrap ARI")
    if has_nmi:
        means_nmi = [summary[r][2] for r in rs]
        stds_nmi  = [summary[r][3] for r in rs]
        ax.errorbar(rs, means_nmi, yerr=stds_nmi, fmt="^--", capsize=3,
                    color="C2", alpha=0.7, label="bootstrap NMI")
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("stability vs full partition")
    ax.tick_params(axis="y")
    ax2 = ax.twinx()
    ax2.plot(rs, ks, "s--", color="C3", alpha=0.7, label="K (# clusters)")
    ax2.set_ylabel("# clusters", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title("Bootstrap resolution stability")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    return ax


def plot_2d_clusters(Z2: np.ndarray, labels: np.ndarray, *,
                     ax=None, title: str = "UMAP-2  (visualisation only)",
                     point_size: float = 3.0):
    """Scatter coloured by cluster id. Noise (-1) shown as grey."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    uniq = sorted(set(int(c) for c in labels))
    cmap = plt.get_cmap("tab20")
    noise_drawn = False
    for i, c in enumerate(uniq):
        m = labels == c
        if c == -1:
            label = "noise" if not noise_drawn else None
            ax.scatter(Z2[m, 0], Z2[m, 1], s=point_size, color="lightgray",
                       alpha=0.5, label=label)
            noise_drawn = True
            continue
        ax.scatter(Z2[m, 0], Z2[m, 1], s=point_size,
                   color=cmap(i % 20), alpha=0.8, label=f"c{c}")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    if len(uniq) <= 20:
        ax.legend(loc="best", fontsize=7, markerscale=2)
    return ax


def plot_3d_clusters(
    Z3: np.ndarray,
    labels: np.ndarray,
    *,
    fig=None,
    title: str = "UMAP-3D  (visualisation only)",
    point_size: float = 2.0,
    elev: float = 20.0,
    azim: float = 30.0,
    rasterized: bool = True,
):
    """3D scatter coloured by cluster id. Noise (-1) shown as grey.

    Static rendering at a single viewing angle (``elev`` / ``azim``).
    Returns the ``Axes3D`` so the caller can save / close the figure.
    Pass ``fig`` to render into an existing figure.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d proj

    if fig is None:
        fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    uniq = sorted(set(int(c) for c in labels))
    cmap = plt.get_cmap("tab20")
    noise_drawn = False
    for i, c in enumerate(uniq):
        m = labels == c
        if c == -1:
            label = "noise" if not noise_drawn else None
            ax.scatter(
                Z3[m, 0], Z3[m, 1], Z3[m, 2],
                s=point_size, color="lightgray", alpha=0.4,
                label=label, rasterized=rasterized, linewidths=0,
            )
            noise_drawn = True
            continue
        ax.scatter(
            Z3[m, 0], Z3[m, 1], Z3[m, 2],
            s=point_size, color=cmap(i % 20), alpha=0.7,
            label=f"c{c}", rasterized=rasterized, linewidths=0,
        )
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    if len(uniq) <= 20:
        ax.legend(loc="best", fontsize=7, markerscale=2)
    return ax


def plot_3d_groups(
    Z3: np.ndarray,
    source_images: np.ndarray,
    group_map: dict[str, str],
    *,
    fig=None,
    title: str = "UMAP-3D coloured by group",
    point_size: float = 1.5,
    alpha: float = 0.35,
    rng_seed: int = 0,
    max_points_per_group: int | None = 20000,
    elev: float = 20.0,
    azim: float = 30.0,
    rasterized: bool = True,
):
    """3D scatter of patches coloured by biological group.

    Mirrors :func:`plot_2d_groups`'s single-panel mode: shuffles per-group
    plotting order, optionally subsamples each group to
    ``max_points_per_group`` for the scatter only, renders unmapped
    patches as a light-grey backdrop.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d proj

    rng = np.random.default_rng(rng_seed)
    src = np.asarray([str(s) for s in source_images])
    group_arr = np.asarray([group_map.get(s) for s in src], dtype=object)
    groups = sorted({g for g in group_arr.tolist() if g is not None})

    palette: list = []
    for cmap_name, n_take in (("tab10", 10), ("tab20", 20), ("Set3", 12)):
        cmap = plt.get_cmap(cmap_name)
        for i in range(n_take):
            palette.append(cmap(i))
    while len(palette) < len(groups):
        palette.extend(palette)
    colour = {g: palette[i] for i, g in enumerate(groups)}

    def _subsample(mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if max_points_per_group is not None and idx.size > max_points_per_group:
            idx = rng.choice(idx, size=max_points_per_group, replace=False)
        return idx

    if fig is None:
        fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    unmapped = group_arr == None  # noqa: E711
    if unmapped.any():
        idx = _subsample(unmapped)
        ax.scatter(
            Z3[idx, 0], Z3[idx, 1], Z3[idx, 2],
            s=point_size * 0.6, color="lightgray", alpha=0.10,
            rasterized=rasterized, linewidths=0, label="unmapped",
        )

    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)
    handles: dict = {}
    for g in shuffled_groups:
        idx = _subsample(group_arr == g)
        ax.scatter(
            Z3[idx, 0], Z3[idx, 1], Z3[idx, 2],
            s=point_size, color=colour[g], alpha=alpha,
            rasterized=rasterized, linewidths=0,
        )
        handles[g] = plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=colour[g], markeredgecolor="none",
            markersize=6, label=g,
        )

    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(
        handles=[handles[g] for g in groups],
        loc="best", fontsize=8, framealpha=0.85,
    )
    return ax


def plot_cluster_grid(
    raw_dataset, medoids: dict, *, channel: int = 0, ncols: int | None = None,
    figsize_per_cell=(1.5, 1.5),
):
    """Row of example patches per cluster (medoid + samples).

    ``raw_dataset`` is indexable, returning (C, H, W) tensor or array.
    ``medoids`` is the dict from :func:`cluster_medoids`.
    ``channel`` selects which channel (or composite max-projection) is shown.
    """
    import matplotlib.pyplot as plt
    cluster_ids = sorted(medoids.keys())
    n_per = max(len(medoids[c]["samples"]) for c in cluster_ids)
    if ncols is None:
        ncols = n_per
    nrows = len(cluster_ids)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_cell[0] * ncols,
                                      figsize_per_cell[1] * nrows))
    axes = np.atleast_2d(axes)

    for r, c in enumerate(cluster_ids):
        info = medoids[c]
        for col in range(ncols):
            ax = axes[r, col]
            if col < len(info["samples"]):
                idx = info["samples"][col]
                patch = raw_dataset[idx]
                if hasattr(patch, "numpy"):
                    patch = patch.numpy()
                if channel == "composite" or channel < 0:
                    img = patch.max(0)
                else:
                    img = patch[channel]
                ax.imshow(img, cmap="gray")
                if col == 0:
                    ax.set_ylabel(f"c{c}\nN={info['size']}", fontsize=8)
                if col == 0 and r == 0:
                    ax.set_title("medoid", fontsize=8)
                elif r == 0:
                    ax.set_title(f"sample {col}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("white" if col >= len(info["samples"]) else "black")
    fig.suptitle("Cluster medoids and samples", y=1.0)
    fig.tight_layout()
    return fig


def plot_image_cluster_heatmap(
    freq: np.ndarray, image_names: np.ndarray, cluster_ids: np.ndarray,
    ax=None, max_images: int = 60, sort_by_dominant_cluster: bool = True,
):
    """Heatmap of image × cluster frequencies.

    Rows can be reordered so visually similar images sit adjacent.
    """
    import matplotlib.pyplot as plt
    if sort_by_dominant_cluster:
        dom = np.argmax(freq, axis=1)
        order = np.lexsort((-freq.max(axis=1), dom))
        freq, image_names = freq[order], image_names[order]
    if max_images is not None and len(image_names) > max_images:
        step = max(1, len(image_names) // max_images)
        freq = freq[::step]; image_names = image_names[::step]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.3 * len(cluster_ids) + 2),
                                       max(4, 0.18 * len(image_names) + 1)))
    im = ax.imshow(freq, aspect="auto", cmap="magma", vmin=0,
                   vmax=min(1.0, max(0.1, freq.max())))
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([str(c) for c in cluster_ids], fontsize=8)
    ax.set_yticks(range(len(image_names)))
    ax.set_yticklabels(image_names, fontsize=6)
    ax.set_xlabel("cluster"); ax.set_ylabel("source_image")
    ax.set_title("Per-image cluster frequencies")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="fraction")
    return ax


def plot_image_level_umap(
    freq: np.ndarray, image_names: np.ndarray, *,
    seed: int = 0, ax=None, label_every: int = 5,
):
    """One point per image; coordinates from UMAP of frequency vectors."""
    import matplotlib.pyplot as plt
    try:
        import umap as _umap
        n_neighbors = max(2, min(15, len(freq) - 1))
        embed = _umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.3,
            metric="euclidean", random_state=seed, n_jobs=1,
        ).fit_transform(freq)
        used = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        embed = PCA(n_components=2, random_state=seed).fit_transform(freq)
        used = "PCA"
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(embed[:, 0], embed[:, 1], s=40, alpha=0.7)
    for i, name in enumerate(image_names):
        if i % label_every == 0:
            ax.annotate(name, (embed[i, 0], embed[i, 1]), fontsize=6, alpha=0.7)
    ax.set_xlabel(f"{used}-1"); ax.set_ylabel(f"{used}-2")
    ax.set_title(f"Image-level {used} of cluster-frequency vectors  "
                 f"(N={len(image_names)})")
    return ax


def plot_intra_image_entropy(rows, ax=None, threshold: float = 0.5):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sizes = np.array([r["size"]         for r in rows])
    ents  = np.array([r["norm_entropy"] for r in rows])
    cs    = [r["cluster"]               for r in rows]
    ax.scatter(sizes, ents, s=40, alpha=0.7)
    for s, e, c in zip(sizes, ents, cs):
        ax.annotate(str(c), (s, e), fontsize=7, alpha=0.7)
    ax.axhline(threshold, color="red", ls="--",
               label=f"image-concentrated  (norm entropy < {threshold})")
    ax.set_xscale("log")
    ax.set_xlabel("cluster size"); ax.set_ylabel("normalised entropy")
    ax.set_title("Per-cluster source-image diversity")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Group-level visualisations
# ---------------------------------------------------------------------------

def plot_2d_groups(
    Z2: np.ndarray,
    source_images: np.ndarray,
    group_map: dict[str, str],
    *,
    ax=None,
    title: str = "UMAP-2D coloured by group",
    point_size: float = 1.5,
    alpha: float = 0.15,
    rng_seed: int = 0,
    max_points_per_group: int | None = 20000,
    facet: bool = False,
    facet_ncols: int = 3,
):
    """Scatter of patches in UMAP space coloured by biological group.

    Notes
    -----
    With ~200k patches a single overlaid scatter is unreadable: the last
    group drawn covers everything beneath it, marker alpha saturates, and
    the largest group (Psilocin) hides every smaller one. This function
    therefore (a) shuffles plotting order so no group sits on top, (b)
    optionally subsamples each group to ``max_points_per_group`` for
    plotting only (the underlying data is unchanged), and (c) supports
    a faceted mode that gives each group its own panel against an
    all-other-patches grey backdrop — by far the most informative view
    of overlapping group distributions.

    Parameters
    ----------
    point_size, alpha
        Tuned for ~200k-point UMAPs. Smaller / more transparent than the
        old defaults (3.0 / 0.6).
    max_points_per_group
        If set, randomly subsample each group to this many points for
        the scatter only. ``None`` disables subsampling.
    facet
        If ``True``, return a ``(fig, axes)`` tuple with one subplot per
        group; ignores ``ax``.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(rng_seed)

    src = np.asarray([str(s) for s in source_images])
    group_arr = np.asarray([group_map.get(s) for s in src], dtype=object)
    groups = sorted({g for g in group_arr.tolist() if g is not None})

    # ----- a categorical colour palette that scales beyond Set1's 9 hues ----
    palette: list = []
    for cmap_name, n_take in (("tab10", 10), ("tab20", 20), ("Set3", 12)):
        cmap = plt.get_cmap(cmap_name)
        for i in range(n_take):
            palette.append(cmap(i))
    while len(palette) < len(groups):
        palette.extend(palette)
    colour = {g: palette[i] for i, g in enumerate(groups)}

    def _subsample(mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if max_points_per_group is not None and idx.size > max_points_per_group:
            idx = rng.choice(idx, size=max_points_per_group, replace=False)
        return idx

    # ---------------- faceted view (one panel per group) -------------------
    if facet:
        n = len(groups)
        ncols = min(facet_ncols, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.0 * ncols, 3.5 * nrows),
            sharex=True, sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()
        # global axis limits so all facets share the same frame
        xlim = (Z2[:, 0].min(), Z2[:, 0].max())
        ylim = (Z2[:, 1].min(), Z2[:, 1].max())
        # background = every patch in light grey
        bg_idx = rng.choice(
            np.arange(len(Z2)),
            size=min(len(Z2), 60000),
            replace=False,
        )
        for ax_i, g in zip(axes, groups):
            ax_i.scatter(
                Z2[bg_idx, 0], Z2[bg_idx, 1],
                s=point_size * 0.6, color="lightgray", alpha=0.10,
                rasterized=True, linewidths=0,
            )
            idx = _subsample(group_arr == g)
            ax_i.scatter(
                Z2[idx, 0], Z2[idx, 1],
                s=point_size, color=colour[g], alpha=min(0.6, alpha * 3),
                rasterized=True, linewidths=0,
            )
            ax_i.set_title(f"{g}\nN={int((group_arr == g).sum())}", fontsize=9)
            ax_i.set_xlim(xlim); ax_i.set_ylim(ylim)
            ax_i.set_xticks([]); ax_i.set_yticks([])
        for ax_i in axes[len(groups):]:
            ax_i.set_visible(False)
        fig.suptitle(title, y=1.0)
        fig.tight_layout()
        return fig, axes

    # ---------------- single overlaid panel (improved defaults) ------------
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    # 1) unmapped patches first (so they sit at the bottom)
    unmapped = group_arr == None  # noqa: E711
    if unmapped.any():
        idx = _subsample(unmapped)
        ax.scatter(
            Z2[idx, 0], Z2[idx, 1],
            s=point_size * 0.6, color="lightgray", alpha=0.10,
            rasterized=True, linewidths=0, label="unmapped",
        )

    # 2) shuffle group draw order to avoid systematic Z-stacking
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)

    # 3) per-group scatter — collect handles in stable (alphabetical) order
    handles: dict = {}
    for g in shuffled_groups:
        idx = _subsample(group_arr == g)
        ax.scatter(
            Z2[idx, 0], Z2[idx, 1],
            s=point_size, color=colour[g], alpha=alpha,
            rasterized=True, linewidths=0,
        )
        # marker for legend with full opacity
        handles[g] = plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=colour[g], markeredgecolor="none",
            markersize=6, label=g,
        )

    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(
        handles=[handles[g] for g in groups],
        loc="best", fontsize=8, framealpha=0.85,
    )
    return ax


def plot_group_cluster_heatmap(
    group_counts: np.ndarray,
    group_names: Sequence[str],
    cluster_ids: np.ndarray,
    *,
    ax=None,
    normalise: bool = True,
):
    """Heatmap of group × cluster frequencies (rows normalised by default)."""
    import matplotlib.pyplot as plt

    data = group_counts.astype(np.float64)
    if normalise:
        row_sum = data.sum(axis=1, keepdims=True).clip(min=1)
        data = data / row_sum
    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(5, 0.4 * len(cluster_ids) + 2),
                     max(2, 0.5 * len(group_names) + 1)),
        )
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([str(c) for c in cluster_ids], fontsize=8)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names, fontsize=10)
    ax.set_xlabel("cluster"); ax.set_ylabel("group")
    ax.set_title("Group × cluster frequencies")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label="fraction" if normalise else "count")
    return ax


def plot_group_frequency_boxplots(
    freq: np.ndarray,
    image_names: np.ndarray,
    group_map: dict[str, str],
    cluster_ids: np.ndarray,
    *,
    max_clusters: int = 20,
    figsize: tuple[float, float] | None = None,
    q_values: np.ndarray | None = None,
):
    """Per-cluster boxplot of image frequencies, split by group.

    Shows which clusters drive group differences. When ``q_values`` is
    given (one corrected p per cluster, e.g. from
    :func:`per_cluster_kruskal_wallis`), each cluster is annotated with
    ``*``/``**``/``***`` for q < 0.05/0.01/0.001 and ``n.s.`` otherwise.
    """
    import matplotlib.pyplot as plt

    groups = sorted(set(group_map.values()))
    k = min(len(cluster_ids), max_clusters)
    cids = cluster_ids[:k]

    if figsize is None:
        figsize = (max(8, 1.2 * k), 4)
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.8 / len(groups)
    cmap = plt.get_cmap("Set1")

    for gi, g in enumerate(groups):
        img_mask = np.array(
            [group_map.get(str(img)) == g for img in image_names]
        )
        if not img_mask.any():
            continue
        positions = np.arange(k) + (gi - len(groups) / 2 + 0.5) * width
        data = [freq[img_mask, ci] for ci in range(k)]
        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.85,
            patch_artist=True, showfliers=False, medianprops={"color": "black"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(cmap(gi % 9))
            patch.set_alpha(0.6)
        ax.plot([], [], color=cmap(gi % 9), label=g, linewidth=6, alpha=0.6)

    ax.set_xticks(range(k))
    ax.set_xticklabels([str(c) for c in cids], fontsize=8)
    ax.set_xlabel("cluster")
    ax.set_ylabel("per-image frequency")
    ax.set_title("Cluster frequency by group")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if q_values is not None:
        q_arr = np.asarray(q_values, dtype=np.float64)
        # headroom for stars
        ymax_per_k = freq[:, :k].max(axis=0) if freq.shape[1] >= k else \
            np.full(k, np.nan)
        ylim_top = float(np.nanmax(ymax_per_k)) if np.isfinite(
            np.nanmax(ymax_per_k)) else 1.0
        ax.set_ylim(top=ylim_top * 1.15)
        for ki in range(k):
            if ki >= len(q_arr) or not np.isfinite(q_arr[ki]):
                continue
            q = q_arr[ki]
            if q < 0.001:
                ann, weight = "***", "bold"
            elif q < 0.01:
                ann, weight = "**", "bold"
            elif q < 0.05:
                ann, weight = "*", "bold"
            else:
                ann, weight = "n.s.", "normal"
            ax.text(ki, ymax_per_k[ki] * 1.05, ann,
                    ha="center", va="bottom",
                    fontsize=9, fontweight=weight)

    fig.tight_layout()
    return fig
