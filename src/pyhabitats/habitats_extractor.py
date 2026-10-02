"""Extracts habitats from images.

This module provides the HabitatsExtractor class, which is responsible
for identifying and extracting habitat regions using input image and mask
data.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from .data_loader import DataLoader
from .types import ClusterAlgorithm


class HabitatsExtractor:
    """Class that finds habitats in medical image."""

    def __init__(self) -> None:
        self._supported_algorithms = ["kmeans", "agglomerative"]

    def execute(
        self,
        image_input: str | np.ndarray,
        mask_input: str | np.ndarray,
        habitats_number: int,
        algorithm: str | ClusterAlgorithm,
    ) -> list[np.ndarray]:
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

        Returns:
            A list of numpy arrays, each representing a binary mask for one habitat.

            If habitats_number is 1, the method returns a list containing mask given
            as input.
        """
        if isinstance(image_input, str) and isinstance(mask_input, str):
            loader = DataLoader()
            image = loader.load(image_input)
            mask = loader.load(mask_input)

        if isinstance(image_input, np.ndarray) and isinstance(mask_input, np.ndarray):
            image = image_input
            mask = mask_input

        if isinstance(algorithm, str):
            alg = self._get_algorithm(algorithm, n_clusters=habitats_number)
        elif isinstance(algorithm, ClusterAlgorithm):
            alg = algorithm

        alg.fit(image[mask > 128])

        habitats = []

        labels = np.full(image.shape[:2], -1, dtype=np.int32)
        labels[mask > 128] = alg.labels_

        for i in range(habitats_number):
            habitat_mask = (labels == i) * 255
            habitat_mask = habitat_mask.astype(np.uint8)
            habitats.append(habitat_mask)

        return habitats

    def _get_algorithm(
        self, algorithm: str, **kwargs: dict[str, int]
    ) -> ClusterAlgorithm:
        """Returns the clustering algorithm based on the provided string.

        Args:
            algorithm: A string representing the desired clustering algorithm.

        Returns:
            An instance of the specified clustering algorithm.
        """
        if algorithm == "kmeans":
            return KMeans(n_clusters=kwargs.get("n_clusters", 2))
        elif algorithm == "agglomerative":
            return AgglomerativeClustering(n_clusters=kwargs.get("n_clusters", 2))
        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Supported algorithms are: {self._supported_algorithms}"
            )
