"""Defines type definitions for clustering algorithms.

This module provides the necessary typing interfaces to support both standard
scikit-learn clustering algorithms and custom implementations within the
habitat analysis pipeline.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.base import ClusterMixin


@runtime_checkable
class CustomClusterAlgorithm(Protocol):
    """Protocol for custom clustering algorithms.

    Defines the required interface for custom clustering algorithms,
    ensuring they provide a fitting mechanism and expose the resulting cluster labels
    in a manner compatible with standard implementations.

    Attributes:
        labels (np.ndarray | None): An array of cluster labels assigned to the input
            data after the model is fitted. Typically remains `None` until `fit` is
            called.
    """

    labels: np.ndarray | None = None

    def fit(self, X: Any, **kwargs: Any) -> Any:
        """Fits the clustering algorithm to the provided training data.

        Args:
            X (Any): The training input samples (e.g., an array of features).
            **kwargs (Any): Additional keyword arguments for the fitting process.

        Returns:
            Any: The fitted instance of the algorithm (typically `self`).
        """
        ...


ClusterAlgorithm = ClusterMixin | CustomClusterAlgorithm
