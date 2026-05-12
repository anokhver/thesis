from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import napari
import numpy as np
from numpy.typing import NDArray


def configure_matplotlib_backend(backend: str = "Qt5Agg") -> None:
    """Set matplotlib backend (call before any plotting)."""
    import matplotlib
    matplotlib.use(backend)


class VSIBrowser:
    """Browse VSI files with thumbnail preview and napari 3D view."""

    # Physical dimensions from microscopy setup

    def __init__(self, items_per_page: int = 4):
        self.folder_path: Optional[Path] = None
        self.vsi_files: List[Path] = []
        self.current_page = 0
        self.items_per_page = items_per_page

    def set_folder(self, folder_path: str) -> bool:
        """Set and validate folder path, load VSI file list."""
        path = Path(folder_path).expanduser().resolve()

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

        print(f"Found {len(self.vsi_files)} VSI files")
        return True

    def load_thumbnail(self, file_path: Path, target_size: int = 256) -> NDArray:
        """Load embedded VSI preview via PIL for fast thumbnailing."""
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                # Convert to grayscale
                gray_img = img.convert('L')

                # Downsample if needed to maintain responsiveness
                if max(gray_img.size) > target_size:
                    scale = target_size / max(gray_img.size)
                    new_size = (int(gray_img.width * scale), int(gray_img.height * scale))
                    gray_img = gray_img.resize(new_size, Image.Resampling.LANCZOS)

                return np.array(gray_img)

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return np.zeros((target_size, target_size), dtype=np.uint8)

    def show_page(self) -> None:
        """Display current page in a 2x2 grid of thumbnails."""
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.vsi_files))
        page_files = self.vsi_files[start_idx:end_idx]

        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes = axes.flatten()

        print(f"\nPage {self.current_page + 1}/{self.total_pages()}")
        print(f"Showing files {start_idx} to {end_idx - 1} of {len(self.vsi_files)}")

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
        if not self.vsi_files:
            return 0
        return (len(self.vsi_files) + self.items_per_page - 1) // self.items_per_page

    def next_page(self) -> bool:
        """Advance to next page if available."""
        if self.current_page < self.total_pages() - 1:
            self.current_page += 1
            return True
        print("Already on last page")
        return False

    def prev_page(self) -> bool:
        """Go back to previous page if available."""
        if self.current_page > 0:
            self.current_page -= 1
            return True
        print("Already on first page")
        return False

    def visualize_napari(self, data: NDArray, title: str) -> None:
        """Launch napari viewer with 3D volume rendering."""
        viewer = napari.Viewer()
        viewer.add_image(
            data,
            name='volume',
            rendering='attenuated_mip',  # Maximum intensity projection with attenuation
            colormap='turbo'
        )
        viewer.title = title
        print("Launching napari viewer...")
        napari.run()

    def visualize(self, idx: int) -> None:
        """Load and visualize a VSI file by index in napari."""
        from imaging.volume import load_full_volume, interpolate_z_axis

        if not self._validate_index(idx):
            return

        file_path = self.vsi_files[idx]
        print(f"\nLoading: {file_path.name}")

        try:
            # Load and process volume
            data = load_full_volume(str(file_path))
            data = interpolate_z_axis(data)
            data = data[0:1]

            # Launch visualization
            self.visualize_napari(data, f"VSI Browser - {file_path.name}")

        except Exception as e:
            print(f"Error visualizing file: {e}")
            import traceback
            traceback.print_exc()

    def _validate_index(self, idx: int) -> bool:
        """Check if index is valid for current file list."""
        if idx < 0 or idx >= len(self.vsi_files):
            print(f"Invalid index. Choose 0 to {len(self.vsi_files) - 1}")
            return False
        return True


class BrowserInterface:
    """Run the command-line interface for ``VSIBrowser``."""

    def __init__(self, browser: VSIBrowser):
        self.browser = browser

    def run(self) -> None:
        """Run the interaction loop."""
        self._print_header()

        while True:
            if not self._folder_selection_loop():
                break

            if not self._browsing_loop():
                break

        print("Goodbye!")

    def _print_header(self) -> None:
        print("=" * 60)
        print("VSI File Browser")
        print("=" * 60)

    def _folder_selection_loop(self) -> bool:
        """Handle folder selection. Returns False to exit."""
        while True:
            # folder_input = input("\nEnter folder path (or 'q' to quit): ").strip()
            folder_input = '/run/media/anokhver/Data/Veronika/ctu/Microscopy/Microscopy/20251030/'

            if folder_input.lower() == 'q':
                return False

            if self.browser.set_folder(folder_input):
                return True

    def _browsing_loop(self):
        """Handle file browsing and visualization. Returns False to exit."""
        while True:
            plt.close('all')
            self.browser.show_page()

            self._print_commands()
            choice = input("Your choice: ").strip().lower()

            action = self._parse_command(choice)
            if action == 'quit':
                plt.close('all')
                return False
            elif action == 'back':
                plt.close('all')
                return True
            elif action == 'continue':
                continue

    def _print_commands(self) -> None:
        print("\nCommands:")
        print("  [index] - visualize file")
        print("  'n' - next page")
        print("  'p' - previous page")
        print("  'b' - change folder")
        print("  'q' - quit")

    def _parse_command(self, choice: str) -> str:
        """Parse user command and execute. Returns 'quit', 'back', or 'continue'."""
        if choice == 'q':
            return 'quit'
        elif choice == 'b':
            return 'back'
        elif choice == 'n':
            self.browser.next_page()
            return 'continue'
        elif choice == 'p':
            self.browser.prev_page()
            return 'continue'
        else:
            # Try to parse as index
            try:
                idx = int(choice)
                plt.close('all')
                self.browser.visualize(idx)
            except ValueError:
                print("Invalid input. Enter a number or command.")
            except Exception as e:
                print(f"Error: {e}")
            return 'continue'


def main():
    browser = VSIBrowser()
    interface = BrowserInterface(browser)
    interface.run()


if __name__ == "__main__":
    main()