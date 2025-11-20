import os
from pathlib import Path
from typing import List, Optional
import gc

import numpy as np
from scipy.ndimage import zoom
from aicsimageio.aics_image import AICSImage
import matplotlib.pyplot as plt
import napari
from numpy.typing import NDArray

import matplotlib
matplotlib.use('Qt5Agg')

class VSIBrowser:
    def __init__(self):
        self.folder_path: Optional[Path] = None
        self.vsi_files: List[Path] = []
        self.current_page = 0
        self.items_per_page = 4
        self.xy_pixel_size = 0.10685428060522417
        self.z_pixel_size = 0.15

    def set_folder(self, folder_path: str) -> bool:
        """Set and validate folder path, load VSI file list."""
        path = Path(folder_path).expanduser()

        if not path.exists():
            print(f"Error: Folder '{folder_path}' does not exist")
            return False

        if not path.is_dir():
            print(f"Error: '{folder_path}' is not a directory")
            return False

        self.folder_path = path
        self.vsi_files = sorted(path.glob("*.vsi"))
        self.current_page = 0

        if not self.vsi_files:
            print(f"No VSI files found in {folder_path}")
            return False

        return True

    def load_thumbnail(self, file_path: Path, target_size: int = 256) -> NDArray:
        """Load VSI as regular image - fast preview without microscopy reader."""
        try:
            from PIL import Image

            # Open VSI directly as image
            img = Image.open(file_path)

            # Convert to grayscale numpy array
            data = np.array(img.convert('L'))

            # Downsample if needed
            if max(data.shape) > target_size:
                scale = target_size / max(data.shape)
                new_size = (int(data.shape[1] * scale), int(data.shape[0] * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                data = np.array(img.convert('L'))

            return data
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return np.zeros((target_size, target_size))

    def show_page(self):
        """Display current page of 4 thumbnails."""
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.vsi_files))

        page_files = self.vsi_files[start_idx:end_idx]

        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes = axes.flatten()

        print(f"\nPage {self.current_page + 1}/{self.total_pages()}")
        print(f"Showing files {start_idx}-{end_idx - 1} of {len(self.vsi_files)}")

        for i, (ax, file_path) in enumerate(zip(axes, page_files)):
            global_idx = start_idx + i
            thumbnail = self.load_thumbnail(file_path)

            ax.imshow(thumbnail, cmap='gray')
            ax.set_title(f"[{global_idx}] {file_path.name}", fontsize=10)
            ax.axis('off')

        # Hide unused subplots
        for i in range(len(page_files), 4):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def total_pages(self) -> int:
        """Calculate total number of pages."""
        return (len(self.vsi_files) + self.items_per_page - 1) // self.items_per_page

    def next_page(self) -> bool:
        """Go to next page if available."""
        if self.current_page < self.total_pages() - 1:
            self.current_page += 1
            return True
        print("Already on last page")
        return False

    def prev_page(self) -> bool:
        """Go to previous page if available."""
        if self.current_page > 0:
            self.current_page -= 1
            return True
        print("Already on first page")
        return False

    def load_full_volume(self, file_path: Path) -> NDArray:
        """Load full 3D volume (from viewer3D)."""
        img = AICSImage(file_path)
        data = img.get_image_data("CZYX", S=0, T=0)
        return data

    def interpolate_z_fast(self, data: NDArray) -> NDArray:
        """From viewer3D - make z-spacing match xy pixel size."""
        interpolation_factor = self.z_pixel_size / self.xy_pixel_size

        print(f"Fast interpolation: {data.shape} -> factor {interpolation_factor:.4f}")

        zoom_factors = (1.0, interpolation_factor, 1.0, 1.0)
        result = zoom(data, zoom_factors, order=1, prefilter=False)

        print(f"Result shape: {result.shape}")
        return result

    def plot_napari(self, data: NDArray, title: str):
        """From viewer3D - exact napari visualization."""
        viewer = napari.Viewer()
        print(f"Adding image to napari with shape: {data.shape}")
        viewer.add_image(data, name='volume', rendering='attenuated_mip', colormap='turbo')
        viewer.title = title
        print("Launching napari viewer...")
        napari.run()

    def visualize(self, idx: int):
        """Load and visualize a VSI file by index."""
        if idx < 0 or idx >= len(self.vsi_files):
            print(f"Invalid index. Choose 0-{len(self.vsi_files) - 1}")
            return

        file_path = self.vsi_files[idx]
        print(f"\nLoading: {file_path.name}")

        # Load full volume
        data = self.load_full_volume(file_path)
        print(f"Original data shape: {data.shape}")

        # Interpolate z-axis
        data = self.interpolate_z_fast(data)
        print(f"Interpolated data shape: {data.shape}")

        # Visualize with napari
        self.plot_napari(data, f"VSI Browser - {file_path.name}")


def main():
    browser = VSIBrowser()

    print("=" * 60)
    print("VSI File Browser")
    print("=" * 60)

    while True:
        # Get folder path
        folder_input = input("\nEnter folder path (or 'q' to quit): ").strip()

        if folder_input.lower() == 'q':
            print("Goodbye!")
            break

        if not browser.set_folder(folder_input):
            continue

        print(f"Found {len(browser.vsi_files)} VSI files")

        # Browse with thumbnails
        while True:
            plt.close('all')  # Close previous matplotlib windows
            browser.show_page()

            print("\nCommands: [index] to visualize | 'n' next | 'p' previous | 'b' new folder | 'q' quit")
            choice = input("Your choice: ").strip()

            if choice.lower() == 'q':
                plt.close('all')
                print("Goodbye!")
                return

            if choice.lower() == 'b':
                plt.close('all')
                break

            if choice.lower() == 'n':
                if browser.next_page():
                    continue

            if choice.lower() == 'p':
                if browser.prev_page():
                    continue

            # Try to parse as index
            try:
                idx = int(choice)
                plt.close('all')
                browser.visualize(idx)
            except ValueError:
                print("Invalid input. Use number, 'n', 'p', 'b', or 'q'")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()