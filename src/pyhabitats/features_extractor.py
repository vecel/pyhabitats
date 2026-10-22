import numpy as np
import pandas as pd


class FeaturesExtractor:
    def __init__(self, params: dict | None = None) -> None:
        self.params = params

    def execute(self, image_input: np.ndarray, mask_input: np.ndarray) -> pd.DataFrame:
        # create pyradiomics extractor
        # run it for every habitat
        # concat to data frame
        pass
