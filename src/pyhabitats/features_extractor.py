"""Extracts radiomic features with respect to habitats.

This module provides the FeaturesExtractor class, which is responsible
for extracting radiomic features for provided image and habitats mask.
"""

import numpy as np
import pandas as pd


class FeaturesExtractor:
    """Radiomic features extractor.

    Utilizes a pyradiomics extractor to compute features for
    different habitats defined by the input mask, aggregating
    the results into a single dataset.

    Args:
        params (dict | None, optional): Configuration parameters for the analysis
            pipeline. Specifies filters applied to image and features to extract.
            Configuration is compatible with pyradiomics configuration
            (see: https://pyradiomics.readthedocs.io/en/latest/customization.html).
            Defaults to `None`.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params

    def execute(self, image_input: np.ndarray, mask_input: np.ndarray) -> pd.DataFrame:
        """Executes the feature extraction process on the given image and mask.

        Initializes the pyradiomics extractor, runs the extraction
        process for each distinct habitat defined in the mask, and concatenates
        the computed features into a single DataFrame.

        Args:
            image_input (np.ndarray): An input image array.
            mask_input (np.ndarray): An input mask array defining the habitat regions.
                Must contain integer values where 0 represents background and positive
                values represent habitats.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the habitat analysis.
                Each row corresponds to one habitat identified by value in column
                `habitat_id`. Extracted radiomic features are stored in the remaining
                columns.
        """
        # create pyradiomics extractor
        # run it for every habitat
        # concat to data frame
