# `imaging/` — microscopy browsing and figure helpers

Utilities for inspecting raw microscopy files and rendering thesis
figures. Not part of the installable package (`pyproject.toml` packages
`src/synaptic_ssl/` only); run directly with `python -m imaging.X` or
`python imaging/X.py` from the repo root.

## Files

| File                    | Role |
|:------------------------|:-----|
| `__init__.py`           | Marks the directory as a package so `from imaging import …` works. |
| `volume.py`             | `load_full_volume` (AICSImage → `(C, Z, Y, X)`) and z-axis interpolation. |
| `browser.py`            | `VSIBrowser`: thumbnail preview + napari 3D view of `.vsi` files. |
| `view3d.py`             | Quick napari 3D viewer for a single volume. |
| `preview_ets_stack.py`  | CLI: before/after preprocessing PNG of an `.ets` z-stack. |
| `preview_patch_grid.py` | CLI: reassembled image with the 128×128 patch grid overlaid. |
| `draw_sim_vicreg_architecture.py` | CLI: render the SimMIM + VICReg + Fourier pretraining schematic as SVG (thesis figure). |

The two CLI scripts (`preview_*.py`) inject `src/` into `sys.path`, so
they work without `pip install -e .`. The library modules (`volume`,
`browser`, `view3d`) need `napari` + `aicsimageio`: install via
`pip install -e ".[imaging]"`.
