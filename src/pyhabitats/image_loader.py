"""Loads medical images from disk.

This module provides ImageLoader class, which is responsible for loading image
files from disk.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk


class ImageLoader:
    """Medical images loader.

    Loads files supported by SimpleITK (e.g. .nii, .nrrd, .jpg). Handles both
    single image files and directories containing multiple images. In the
    case of multiple images, loader stacks them into single numpy array.
    """

    def load_image(self, image_path: str) -> np.ndarray:
        """Loads image from a file or directory.

        Args:
            image_path (str): Path to the image file or a directory of images.

        Returns:
            np.ndarray: A numpy array representing the loaded image(s). Values
                inside the array represent features used in clusterization.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")

        if path.is_file():
            return self._read_file(path)

        if path.is_dir():
            return self._read_directory(path)

    def _read_file(self, filepath: Path) -> np.ndarray:
        try:
            image = sitk.ReadImage(str(filepath))
            array = sitk.GetArrayFromImage(image)
            return array
        except RuntimeError as e:
            raise ValueError(f"Failed to read image file: {filepath}. Error: {e}")

    def _read_directory(self, directory: Path) -> np.ndarray:
        files = sorted([f for f in directory.iterdir() if f.is_file()])

        if not files:
            raise FileNotFoundError(f"No files found in directory: {directory}")

        arrays = [self._read_file(str(file)) for file in files]

        if len(arrays) == 1:
            # Add a message that only one file was found and is read as single image
            return arrays[0]

        try:
            return np.stack(arrays, axis=-1)
        except ValueError:
            raise ValueError(
                f"All input images must have the same shape. Found shapes: {', '.join(str(arr.shape) for arr in arrays)}."
            )
