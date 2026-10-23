"""Runs habitat analysis pipeline.

This module provides the HabitatAnalysisPipeline class, which runs radiomic features extraction
with respect to habitat regions.
"""

import numpy as np
import pandas as pd

from .cluster_algorithm import ClusterAlgorithm


class HabitatAnalysisPipeline:
    """Pipeline for executing habitat analysis from loading raw images to features
    extraction.

    Simplifies habitat analysis pipeline by wrapping ImageLoader, HabitatsExtractor and
    FeaturesExtractor into one object with simple, one-method API. Pipeline execution is
    equivalent to carrying out the subsequent steps of the analysis:
    1. Loading raw image and mask from disk.
    2. Applying clusterization algorithm to determine habitats.
    3. Extracting radiomic features for each habitat.

    Attributes:
        habitats (np.ndarray | None): Habitats mask determined during pipeline execution.
            Set to `None` before pipeline execution.

    Args:
        params (dict | None, optional): Configuration parameters for the analysis
            pipeline. Specifies filters applied to image and radiomic features to
            extract. Configuration is compatible with pyradiomics configuration
            (see: https://pyradiomics.readthedocs.io/en/latest/customization.html).
            Defaults to `None`.
    """

    habitats: np.ndarray | None

    def __init__(self, params: dict | None = None) -> None:
        self.params = params
        self.habitats = None

    def execute(
        self,
        image_input: str,
        mask_input: str,
        habitats_number: int,
        algorithm: str | ClusterAlgorithm,
    ) -> pd.DataFrame:
        """Executes the habitat analysis pipeline on the provided image and mask.

        Args:
            image_input (str): Path to the file or directory containing input image
                (images).
            mask_input (str): Path to the input mask file. Mask must be binary and
                contain only zero and one values for, respectively, backgroud and
                region of interest.
            habitats_number (int): Number of habitats to identify. Must be positive.
                If set to one the computed habitat will be equal to mask input.
            algorithm (str | ClusterAlgorithm): The clustering algorithm to apply.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the habitat analysis.
                Each row corresponds to one habitat identified by value in column
                `habitat_id`. Extracted radiomic features are stored in the remaining
                columns.
        """
