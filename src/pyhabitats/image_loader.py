"""Loads medical images from disk.

This module provides ImageLoader class, which is responsible for loading image
files from disk.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk


class ImageLoader:
    """Class that loads medical images from disk.

    It can load files supported by SimpleITK (e.g. .nii, .nrrd, .jpg). Handles
    both single image files and directories containing multiple images. In the
    case of multiple images, loader stacks them into single numpy array.
    """

    def load_image(self, image_path: str) -> np.ndarray:
        """Loads image from file or directory.

        Args:
            filepath: Path to the image file or directory of images.

        Returns:
            A numpy array representing the loaded image(s). Values inside the array
            represent features used for clustering.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")

        if path.is_file():
            return self._read_file(str(path))

        if path.is_dir():
            files = sorted([f for f in path.iterdir() if f.is_file()])

            if not files:
                raise FileNotFoundError(f"No files found in directory: {image_path}")

            arrays = [self._read_file(str(file)) for file in files]

            return np.stack(arrays, axis=-1)

    def _read_file(self, filepath: str) -> np.ndarray:
        image = sitk.ReadImage(filepath)
        array = sitk.GetArrayFromImage(image)
        return array
