import os
import gc

import numpy as np
from scipy.ndimage import zoom
from aicsimageio.aics_image import AICSImage

import napari

from numpy.typing import NDArray

def load(file_path: str) -> NDArray:
    img = AICSImage(file_path)
    data = img.get_image_data("CZYX", S=0, T=0)
    return data

def interpolate_z_memory_efficient(data: NDArray, xy_pixel_size: float = 0.10685428060522417, z_pixel_size: float = 0.15, chunk_size: int = 50) -> NDArray:
    """
    Memory-efficient z-axis interpolation to create exactly square pixels.
    Uses linear interpolation and processes data in chunks to minimize RAM usage.
    
    Parameters:
    - data: Input data with shape (C, Z, Y, X)
    - xy_pixel_size: Size of xy pixels in physical units (e.g., micrometers)
    - z_pixel_size: Size of z pixels in physical units (e.g., micrometers)
    - chunk_size: Number of Y slices to process at once (reduce for lower memory usage)
    
    Returns:
    - Interpolated data where z-axis spacing matches xy pixel size for square pixels
    """
    c, z, y, x = data.shape
    
    # Calculate the exact interpolation factor to make pixels square
    interpolation_factor = z_pixel_size / xy_pixel_size
    new_z = int(np.round((z - 1) * interpolation_factor + 1))
    
    print(f"Original shape: {data.shape}")
    print(f"Target z-slices: {new_z} (factor: {interpolation_factor:.4f})")
    print(f"Memory-efficient processing with chunk size: {chunk_size}")
    
    # Calculate zoom factors for each axis (only z-axis changes)
    zoom_factors = (1.0, interpolation_factor, 1.0, 1.0)  # (C, Z, Y, X)
    
    # Estimate memory usage
    input_size_mb = data.nbytes / (1024**2)
    output_size_mb = c * new_z * y * x * data.itemsize / (1024**2)
    print(f"Input size: {input_size_mb:.1f} MB, Output size: {output_size_mb:.1f} MB")
    
    # Create output array with memory mapping if large
    if output_size_mb > 1000:  # > 1GB
        print("Using memory mapping for large output array")
        # Create a temporary file for memory mapping
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        new_data = np.memmap(temp_file.name, dtype=data.dtype, mode='w+', shape=(c, new_z, y, x))
    else:
        new_data = np.empty((c, new_z, y, x), dtype=data.dtype)
    
    # Process each channel separately to save memory
    for channel_idx in range(c):
        print(f"Processing channel {channel_idx + 1}/{c}...")
        
        # Process in chunks along Y axis to minimize memory usage
        for y_start in range(0, y, chunk_size):
            y_end = min(y_start + chunk_size, y)
            
            # Extract chunk for this channel
            chunk = data[channel_idx, :, y_start:y_end, :]
            
            # Use scipy's zoom for fast linear interpolation
            # Only interpolate along z-axis (axis=0)
            interpolated_chunk = zoom(chunk, (interpolation_factor, 1.0, 1.0), order=1, prefilter=False)
            
            # Store result
            new_data[channel_idx, :, y_start:y_end, :] = interpolated_chunk
            
            # Force garbage collection to free memory
            del chunk, interpolated_chunk
            gc.collect()
            
            if (y_start // chunk_size) % 10 == 0:
                progress = (y_start / y) * 100
                print(f"  Progress: {progress:.1f}%")
    
    # Verify the new pixel spacing
    actual_z_spacing = (z - 1) * z_pixel_size / (new_z - 1)
    print(f"Final z-spacing: {actual_z_spacing:.6f} (target: {xy_pixel_size:.6f})")
    print(f"Pixel aspect ratio: {actual_z_spacing / xy_pixel_size:.4f}")
    
    return new_data

def interpolate_z_fast(data: NDArray, xy_pixel_size: float = 0.10685428060522417, z_pixel_size: float = 0.15) -> NDArray:
    """
    Fastest interpolation using scipy.ndimage.zoom - processes entire array at once.
    Use this if you have enough RAM.
    """
    interpolation_factor = z_pixel_size / xy_pixel_size
    
    print(f"Fast interpolation: {data.shape} -> factor {interpolation_factor:.4f}")
    
    # Zoom only along z-axis (axis=1 in CZYX format)
    zoom_factors = (1.0, interpolation_factor, 1.0, 1.0)
    
    # Use linear interpolation (order=1) for speed, no prefiltering for memory efficiency
    result = zoom(data, zoom_factors, order=1, prefilter=False)
    
    print(f"Result shape: {result.shape}")
    return result

def plot_napari(data: NDArray):
    viewer = napari.Viewer()
    print(f"Adding image to napari with shape: {data.shape}")
    viewer.add_image(data, name='volume', rendering='attenuated_mip', colormap='turbo')
    print("Launching napari viewer...")
    napari.run()

if __name__ == "__main__":

    folder = '/run/media/anokhver/Data/Veronika/ctu/Microscopy/20251030/'

    # Find all .vsi files
    vsi_files = [f for f in os.listdir(folder) if f.endswith('.vsi')]
    print(f"Found {len(vsi_files)} VSI files:")
    print(vsi_files)

    # Load the first one
    file_path = os.path.join(folder, vsi_files[0])
    data = load(file_path)

    print(f"Loading file: {file_path}")
    data = load(file_path)
    print(f"Original data shape: {data.shape}")
    data = interpolate_z_fast(data)
    print(f"Interpolated data shape: {data.shape}")
    napari.run()
    plot_napari(data)