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
        self._loader = ImageLoader()

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
        image_input = None
        mask_input = None
        if not isinstance(image_input, (str, np.ndarray)):
            raise TypeError(
                f"image_input must be a string representing a file path or a numpy array, got {type(image_input)}"
            )

        if not isinstance(mask_input, (str, np.ndarray)):
            raise TypeError(
                f"mask_input must be a string representing a file path or a numpy array, got {type(mask_input)}"
            )

        if not isinstance(habitats_number, int) or isinstance(habitats_number, bool):
            raise TypeError(
                f"habitats_number must be an integer, got {type(habitats_number)}"
            )

        if not isinstance(algorithm, (str, ClusterAlgorithm)):
            raise TypeError(
                f"algorithm must be a string from {self._supported_algorithms} or an instance of ClusterAlgorithm, got {type(algorithm)}"
            )

        image = self._get_image_from_input(image_input)
        mask = self._get_image_from_input(mask_input)
        alg = self._get_algorithm(algorithm, n_clusters=habitats_number)

        features_number = image.shape[-1] if image.ndim > 3 else 1
        alg.fit(image[mask == 1].reshape(-1, features_number))

        habitats = []

        labels = np.full_like(mask, -1, dtype=int)
        labels[mask == 1] = alg.labels_

        for i in range(habitats_number):
            habitat_mask = (labels == i) * 1
            habitat_mask = habitat_mask.astype(np.uint8)
            habitats.append(habitat_mask)

        return habitats

    def _get_image_from_input(self, input: str | np.ndarray) -> np.ndarray:
        if isinstance(input, str):
            return self._loader.load_image(input)
        return input

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
