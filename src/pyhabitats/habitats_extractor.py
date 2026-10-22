"""Extracts habitats from images.

This module provides the HabitatsExtractor class, which is responsible
for identifying and extracting habitat regions using input image and mask
data.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from .clustering_algorithm import ClusteringAlgorithm
from .image_loader import ImageLoader


# TODO: Adjust clusterization methods for 3d and 4d data.
class HabitatsExtractor:
    """Class that finds habitats in medical image."""

    def __init__(self) -> None:
        self._supported_algorithms = ["kmeans", "agglomerative"]
        self._loader = ImageLoader()

    def execute(
        self,
        image_input: np.ndarray,
        mask_input: np.ndarray,
        habitats_number: int,
        algorithm: str | ClusteringAlgorithm,
    ) -> np.ndarray:
        """Extracts habitats from image and mask.

        Applies a specified clustering algorithm to the region of the input image
        defined by the provided mask. It partitions the masked pixels into the
        requested number of classes and generates a separate binary mask for each
        resulting cluster.

        Args:
            image_input: Path to the image or a numpy array representing the image.
            mask_input: Path to the mask or a numpy array representing the mask.
            habitats_number: Number of habitats to extract.
            algorithm: Clusterization algorithm used for extraction.
            binarization_method: Method used for binarizing the mask.

        Returns:
            A list of numpy arrays, each representing a binary mask for one habitat.

            If habitats_number is 1, the method returns a list containing mask given
            as input.
        """
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

        if not isinstance(algorithm, (str, ClusteringAlgorithm)):
            raise TypeError(
                f"algorithm must be a string from {self._supported_algorithms} or an instance of ClusteringAlgorithm, got {type(algorithm)}"
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
        self, algorithm: str | ClusteringAlgorithm, **kwargs: dict[str, int]
    ) -> ClusteringAlgorithm:
        """Returns the clustering algorithm based on the provided string.

        Args:
            algorithm: A string representing the desired clustering algorithm.

        Returns:
            An instance of the specified clustering algorithm.
        """
        if isinstance(algorithm, ClusteringAlgorithm):
            return algorithm

        if algorithm == "kmeans":
            return KMeans(n_clusters=kwargs.get("n_clusters", 2))
        elif algorithm == "agglomerative":
            return AgglomerativeClustering(n_clusters=kwargs.get("n_clusters", 2))
        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Supported algorithms are: {self._supported_algorithms}"
            )
