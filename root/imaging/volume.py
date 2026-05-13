"""Load volumes and interpolate z-axis for confocal microscopy data."""

from __future__ import annotations

import gc

import numpy as np
from aicsimageio.aics_image import AICSImage
from numpy.typing import NDArray
from scipy.ndimage import zoom


def load_full_volume(file_path: str) -> NDArray:
    """Load a 3D volume via AICSImage. Return ``(C, Z, Y, X)``."""
    img = AICSImage(file_path)

    print(f"Physical pixel sizes: {img.physical_pixel_sizes}")
    print(f"Dimensions: {img.dims}")
    print(f"Shape: {img.shape}")

    data = img.get_image_data("CZYX", S=0, T=0)
    print(f"Loaded shape: {data.shape} (C, Z, Y, X)")

    if data.shape[1] < 10:
        print(f"WARNING: Only {data.shape[1]} z-slices detected")
        print("This may be a preview rather than full resolution data")

    return data


def interpolate_z_axis(
    data: NDArray,
    z_pixel_size: float = 0.15,
    xy_pixel_size: float = 0.10685428060522417,
) -> NDArray:
    """Z-interpolate to isotropic voxels via scipy ``zoom`` (linear, in-RAM)."""
    interpolation_factor = z_pixel_size / xy_pixel_size

    print(f"Z-interpolation factor: {interpolation_factor:.4f}")
    print(f"Input shape: {data.shape}")

    zoom_factors = (1.0, interpolation_factor, 1.0, 1.0)
    result = zoom(data, zoom_factors, order=1, prefilter=False)

    print(f"Output shape: {result.shape}")
    return result


def interpolate_z_memory_efficient(
    data: NDArray,
    xy_pixel_size: float = 0.10685428060522417,
    z_pixel_size: float = 0.15,
    chunk_size: int = 50,
) -> NDArray:
    """Z-interpolate to isotropic voxels, processing Y in chunks.

    Spill to memmap above 1 GB output to stay within RAM.
    """
    c, z, y, x = data.shape

    interpolation_factor = z_pixel_size / xy_pixel_size
    new_z = int(np.round((z - 1) * interpolation_factor + 1))

    print(f"Original shape: {data.shape}")
    print(f"Target z-slices: {new_z} (factor: {interpolation_factor:.4f})")
    print(f"Memory-efficient processing with chunk size: {chunk_size}")

    input_size_mb = data.nbytes / (1024 ** 2)
    output_size_mb = c * new_z * y * x * data.itemsize / (1024 ** 2)
    print(f"Input size: {input_size_mb:.1f} MB, Output size: {output_size_mb:.1f} MB")

    if output_size_mb > 1000:
        print("Using memory mapping for large output array")
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        new_data = np.memmap(
            temp_file.name, dtype=data.dtype, mode="w+", shape=(c, new_z, y, x)
        )
    else:
        new_data = np.empty((c, new_z, y, x), dtype=data.dtype)

    for channel_idx in range(c):
        print(f"Processing channel {channel_idx + 1}/{c}...")

        for y_start in range(0, y, chunk_size):
            y_end = min(y_start + chunk_size, y)
            chunk = data[channel_idx, :, y_start:y_end, :]
            interpolated_chunk = zoom(
                chunk, (interpolation_factor, 1.0, 1.0), order=1, prefilter=False
            )
            new_data[channel_idx, :, y_start:y_end, :] = interpolated_chunk

            del chunk, interpolated_chunk
            gc.collect()

            if (y_start // chunk_size) % 10 == 0:
                progress = (y_start / y) * 100
                print(f"  Progress: {progress:.1f}%")

    actual_z_spacing = (z - 1) * z_pixel_size / (new_z - 1)
    print(f"Final z-spacing: {actual_z_spacing:.6f} (target: {xy_pixel_size:.6f})")
    print(f"Pixel aspect ratio: {actual_z_spacing / xy_pixel_size:.4f}")

    return new_data


def plot_napari(data: NDArray) -> None:
    """Open napari with attenuated-MIP rendering on ``data``."""
    import napari

    viewer = napari.Viewer()
    print(f"Adding image to napari with shape: {data.shape}")
    viewer.add_image(data, name="volume", rendering="attenuated_mip", colormap="turbo")
    print("Launching napari viewer...")
    napari.run()
