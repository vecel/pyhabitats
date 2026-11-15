"""Extracts radiomic features with respect to habitats.

This module provides the FeaturesExtractor class, which is responsible
for extracting radiomic features for provided image and habitats mask.
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

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
        self.params = params or {}
        self._extractor = featureextractor.RadiomicsFeatureExtractor(**self.params)

    def execute(self, image: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
        """Executes the feature extraction process on the given image and mask.

        Initializes the pyradiomics extractor, runs the extraction
        process for each distinct habitat defined in the mask, and concatenates
        the computed features into a single DataFrame.

        Args:
            image (np.ndarray): An input image array.
            mask (np.ndarray): An input mask array defining the habitat regions.
                Must contain integer values where 0 represents background and positive
                values represent habitats.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the habitat analysis.
                Each row corresponds to one habitat identified by value in column
                `habitat_id`. Extracted radiomic features are stored in the remaining
                columns.
        """

        sitk_image = sitk.GetImageFromArray(image)
        
        habitat_ids = np.unique(mask)
        habitat_ids = habitat_ids[habitat_ids > 0] 
        
        results = []
        for h_id in habitat_ids:
            habitat_features = self._extract_single_habitat(sitk_image, mask, h_id)
            results.append(habitat_features)
            
        df = pd.DataFrame(results)
        
        if not df.empty and "habitat_id" in df.columns:
            cols = ["habitat_id"] + [c for c in df.columns if c != "habitat_id"]
            df = df[cols]
            
        return df

    def _extract_single_habitat(self, sitk_image: sitk.Image, mask: np.ndarray, habitat_id: int) -> dict:
        """Extracts features for a single habitat identified by habitat_id."""
        
        h_mask_np = (mask == habitat_id).astype(np.uint8)
        sitk_mask = sitk.GetImageFromArray(h_mask_np)
        sitk_mask.CopyInformation(sitk_image)
        
        try:
            features = self._extractor.execute(sitk_image, sitk_mask)
            
            clean_features = {
                k: v for k, v in features.items() if not k.startswith("diagnostics_")
            }
            clean_features["habitat_id"] = habitat_id
            return clean_features
            
        except Exception:
            return {"habitat_id": habitat_id}