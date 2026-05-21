"""Embedding extraction: run a frozen encoder over a patch dataset.

Returns globally-pooled features plus per-patch metadata, optionally
caching to disk so the GPU pass runs only once per checkpoint.
"""
from __future__ import annotations

import numpy as np


def extract_patch_embeddings(
    encoder,
    dataset,
    *,
    device,
    batch_size: int = 64,
    num_workers: int = 0,
    cache_path=None,
    force_recompute: bool = False,
):
    """Run frozen encoder over dataset; return globally-pooled features.

    Returns (Z, filenames, source_images, image_indices). Caches to
    ``cache_path`` if set.
    """
    import torch
    from pathlib import Path

    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not force_recompute:
        d = np.load(cache_path, allow_pickle=True).item()
        return (d["Z"].astype(np.float32),
                np.asarray(d["filenames"]),
                np.asarray(d["source_images"]),
                np.asarray(d["image_indices"], dtype=np.int64))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(str(device) != "cpu"),
    )
    feats = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True).contiguous()
            z = encoder(x)[-1].mean(dim=(-2, -1))
            feats.append(z.float().cpu().numpy())
    Z = np.concatenate(feats, axis=0).astype(np.float32)

    records = getattr(dataset, "records", None)
    if records is None:
        # TransformedSubset(subset=Subset(base=PatchDataset)) is the common
        # wrapper chain; resolve it.
        sub = getattr(dataset, "subset", None)
        if sub is None and hasattr(dataset, "indices"):
            sub = dataset
        if sub is not None:
            # sub is the full PatchDataset (no Subset wrapper)
            sub_records = getattr(sub, "records", None)
            if sub_records is not None and not hasattr(sub, "indices"):
                records = sub_records
            else:
                # sub is a Subset: resolve base dataset + indices
                base = getattr(sub, "dataset", None)
                indices = getattr(sub, "indices", None)
                base_records = getattr(base, "records", None)
                if base_records is not None and indices is not None:
                    records = [base_records[int(i)] for i in indices]
    if records is None:
        filenames = np.array([f"row_{i:06d}.npy" for i in range(Z.shape[0])])
        source_images = np.array(["UNKNOWN"] * Z.shape[0])
        image_indices = np.zeros(Z.shape[0], dtype=np.int64)
    else:
        # Tolerate heterogeneous index.csv schemas: older zip-tiler outputs
        # only write ``source_npy`` / ``source_path`` and omit ``source_image``
        # and ``image_index``. Fall back to whatever per-source identifier
        # is available, and derive a stable per-image int when missing.
        def _resolve_source_image(r):
            val = (r.get("source_image")
                   or r.get("source_npy")
                   or r.get("source_path"))
            if not val:
                return "UNKNOWN"
            s = str(val).replace("\\", "/")
            return s.rsplit("/", 1)[-1] or s

        filenames     = np.array([r["filename"] for r in records])
        source_images = np.array([_resolve_source_image(r) for r in records])

        parsed_indices: list[int] = []
        needs_derive = False
        for r in records:
            raw = str(r.get("image_index", "")).strip()
            if not raw:
                needs_derive = True
                break
            try:
                parsed_indices.append(int(raw))
            except ValueError:
                needs_derive = True
                break
        if not needs_derive and len(parsed_indices) == len(records):
            image_indices = np.array(parsed_indices, dtype=np.int64)
        else:
            seen: dict[str, int] = {}
            indices_list: list[int] = []
            for s in source_images.tolist():
                if s not in seen:
                    seen[s] = len(seen)
                indices_list.append(seen[s])
            image_indices = np.array(indices_list, dtype=np.int64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path,
                {"Z": Z, "filenames": filenames,
                 "source_images": source_images,
                 "image_indices": image_indices},
                allow_pickle=True)
    return Z, filenames, source_images, image_indices
