import pytest
from typing import Any
import numpy as np

class OneHabitatClusterAlgorithm:
    """ClusterAlgorithm that identifies only one habitat with label 1."""
    labels = None

    def fit(self, X: Any, **kwargs: Any) -> Any:
        self.labels = np.ones(len(X))

@pytest.fixture
def one_habitat_clusterer():
    return OneHabitatClusterAlgorithm()