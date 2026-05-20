"""Load 3D microscopy volumes and z-interpolate to isotropic voxels."""

from __future__ import annotations

import gc
import logging

import numpy as np
from aicsimageio.aics_image import AICSImage
from numpy.typing import NDArray
from scipy.ndimage import zoom


logger = logging.getLogger(__name__)


def load_full_volume(file_path: str) -> NDArray:
    """Load the highest-resolution 3D volume. Return ``(C, Z, Y, X)``."""
    img = AICSImage(file_path)

    logger.info("Physical pixel sizes: %s", img.physical_pixel_sizes)
    logger.info("Dimensions: %s", img.dims)
    logger.info("Shape: %s", img.shape)

    data = img.get_image_data("CZYX", S=0, T=0)
    logger.info("Loaded shape: %s (C, Z, Y, X)", data.shape)

    if data.shape[1] < 10:
        logger.warning(
            "Only %d z-slices detected; may be a preview rather than full "
            "resolution data",
            data.shape[1],
        )

    return data


def interpolate_z_axis(data: NDArray,
                       z_pixel_size = 0.15, # microns
                       xy_pixel_size = 0.10685428060522417) -> NDArray:
    """Z-interpolate to isotropic voxels via scipy ``zoom`` (linear, in-RAM)."""
    interpolation_factor = z_pixel_size / xy_pixel_size

    logger.info("Z-interpolation factor: %.4f", interpolation_factor)
    logger.info("Input shape: %s", data.shape)

    # Zoom Z only (axis 1 in CZYX); X, Y unchanged.
    zoom_factors = (1.0, interpolation_factor, 1.0, 1.0)
    result = zoom(data, zoom_factors, order=1, prefilter=False)

    logger.info("Output shape: %s", result.shape)
    return result

def interpolate_z_memory_efficient(data: NDArray, xy_pixel_size: float = 0.10685428060522417,
                                   z_pixel_size: float = 0.15, chunk_size: int = 50) -> NDArray:
    """Z-interpolate to isotropic voxels, processing Y in chunks.

    Spill to memmap above 1 GB output.
    """
    c, z, y, x = data.shape

    interpolation_factor = z_pixel_size / xy_pixel_size
    new_z = int(np.round((z - 1) * interpolation_factor + 1))

    logger.info("Original shape: %s", data.shape)
    logger.info(
        "Target z-slices: %d (factor: %.4f)", new_z, interpolation_factor,
    )
    logger.info("Memory-efficient processing with chunk size: %d", chunk_size)

    input_size_mb = data.nbytes / (1024 ** 2)
    output_size_mb = c * new_z * y * x * data.itemsize / (1024 ** 2)
    logger.info(
        "Input size: %.1f MB, Output size: %.1f MB",
        input_size_mb, output_size_mb,
    )

    zoom_factors = (1.0, interpolation_factor, 1.0, 1.0)  # (C, Z, Y, X)

    if output_size_mb > 1000:
        logger.info("Using memory mapping for large output array")
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        new_data = np.memmap(temp_file.name, dtype=data.dtype, mode='w+', shape=(c, new_z, y, x))
    else:
        new_data = np.empty((c, new_z, y, x), dtype=data.dtype)

    for channel_idx in range(c):
        logger.info("Processing channel %d/%d...", channel_idx + 1, c)

        for y_start in range(0, y, chunk_size):
            y_end = min(y_start + chunk_size, y)
            chunk = data[channel_idx, :, y_start:y_end, :]
            # Interpolate along Z only (axis=0 inside the per-channel slice).
            interpolated_chunk = zoom(chunk, (interpolation_factor, 1.0, 1.0), order=1, prefilter=False)
            new_data[channel_idx, :, y_start:y_end, :] = interpolated_chunk

            del chunk, interpolated_chunk
            gc.collect()

            if (y_start // chunk_size) % 10 == 0:
                progress = (y_start / y) * 100
                logger.info("  Progress: %.1f%%", progress)

    actual_z_spacing = (z - 1) * z_pixel_size / (new_z - 1)
    logger.info(
        "Final z-spacing: %.6f (target: %.6f)", actual_z_spacing, xy_pixel_size,
    )
    logger.info(
        "Pixel aspect ratio: %.4f", actual_z_spacing / xy_pixel_size,
    )

    return new_data

