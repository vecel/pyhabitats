"""Loads medical images from disk.

This module provides DataLoader class, which is responsible for loading image
files from disk.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk


class DataLoader:
    """Class that loads medical images from disk.

    It can load files supported by SimpleITK (e.g. .nii, .nrrd, .jpg). Handles
    both single image files and directories containing multiple images. In the
    case of multiple images, loader stacks them into single numpy array.
    """

    def load(self, filepath: str) -> np.ndarray:
        """Loads image from file or directory.

        Args:
            filepath: Path to the image file or directory of images.

        Returns:
            A numpy array representing the loaded image(s).
        """
        # TODO: Add docs that array elements are features the algorithm is run on
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if path.is_file():
            return self._read_file(str(path))

        if path.is_dir():
            files = sorted([f for f in path.iterdir() if f.is_file()])

            if not files:
                raise FileNotFoundError(f"No files found in directory: {filepath}")

            arrays = [self._read_file(str(file)) for file in files]

            if arrays[0].ndim == 2:
                return np.stack(arrays, axis=-1)
            else:
                return np.concatenate(arrays, axis=-1)

    def _read_file(self, filepath: str) -> np.ndarray:
        image = sitk.ReadImage(filepath)
        array = sitk.GetArrayFromImage(image)
        return array
