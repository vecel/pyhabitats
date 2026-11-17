"""Extracts habitats from images.

This module provides the HabitatsExtractor class, which is responsible
for identifying and extracting habitat regions using input image and mask
data.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from .cluster_algorithm import ClusterAlgorithm
from .image_loader import ImageLoader


class HabitatsExtractor:
    """Extractor for identifying distinct habitats from image.

    This class allows to specify number of habitats and clusterization algorithm
    used during extraction. Supports predefined algorithms and custom ones compatible
    with `ClusterAlgorithm` type.
    """

    def __init__(self) -> None:
        self._supported_algorithms = ["kmeans", "agglomerative"]

    def execute(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        habitats_number: int,
        algorithm: str | ClusterAlgorithm,
    ) -> np.ndarray:
        """Identifies habitats in image.

        Applies a specified clustering algorithm to the region of the input image
        defined by the provided mask. It partitions the masked pixels into the
        requested number of classes and generates a mask for each resulting cluster.

        Args:
            image (np.ndarray): A numpy array representing the image.
            mask (np.ndarray): A numpy array representing the mask. Must be
                binary.
            habitats_number (int): Number of habitats to extract. If set to 1, provided
                mask will be returned as output.
            algorithm (str | ClusterAlgorithm): Clusterization algorithm used for
                extraction.

        Returns:
            np.ndarray: A numpy array with extracted habitats. Each habitat is identified
                by positive, integer value 1, 2, 3 and so on. Background is identifed by value 0.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image argument must be a numpy array, got {type(image)}")

        if not isinstance(mask, np.ndarray):
            raise TypeError(f"mask argument must be a numpy array, got {type(mask)}")

        if not isinstance(habitats_number, int) or isinstance(habitats_number, bool):
            raise TypeError(
                f"habitats_number must be an integer, got {type(habitats_number)}"
            )

        if not isinstance(algorithm, (str, ClusterAlgorithm)):
            raise TypeError(
                f"algorithm must be a string from list: {self._supported_algorithms} or an instance of ClusterAlgorithm, got {type(algorithm)}"
            )

        self._check_image_and_mask_compatibility(image, mask)
        alg = self._get_algorithm(algorithm, n_clusters=habitats_number)

        features_number = image.shape[-1] if image.ndim > mask.ndim else 1
        alg.fit(image[mask == 1].reshape(-1, features_number))

        habitats = np.zeros_like(mask, dtype=int)

        if hasattr(alg, "labels_"):
            habitats[mask == 1] = alg.labels_ + 1
        elif hasattr(alg, "labels"):
            habitats[mask == 1] = alg.labels + 1
        else:
            raise AttributeError(
                "Did not found neither .labels nor .labels_ attribute for algorithm"
            )

        return habitats

    def _check_image_and_mask_compatibility(
        self, image: np.ndarray, mask: np.ndarray
    ) -> None:
        n = mask.ndim
        if not (image.ndim == mask.ndim or image.ndim == mask.ndim + 1):
            raise ValueError(
                f"image must have either same dimensions as mask or one more, got image.ndim={image.ndim}, mask.ndim={mask.ndim}"
            )

        if image.shape[:n] != mask.shape:
            raise ValueError(
                f"image leading dimensions have to match mask dimensions, got mask.shape={mask.shape}, image.shape={image.shape}"
            )

    def _get_algorithm(
        self, algorithm: str | ClusterAlgorithm, **kwargs: dict[str, int]
    ) -> ClusterAlgorithm:
        """Returns the clustering algorithm based on the provided string.

        Args:
            algorithm: A string representing the desired clustering algorithm.

        Returns:
            An instance of the specified clustering algorithm.
        """
        if isinstance(algorithm, ClusterAlgorithm):
            return algorithm

        if algorithm == "kmeans":
            return KMeans(n_clusters=kwargs.get("n_clusters", 2))
        elif algorithm == "agglomerative":
            return AgglomerativeClustering(n_clusters=kwargs.get("n_clusters", 2))
        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Supported algorithms are: {self._supported_algorithms}"
            )
