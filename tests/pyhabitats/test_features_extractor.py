import numpy as np
import pandas as pd
import pytest

from pyhabitats.features_extractor import FeaturesExtractor


@pytest.fixture
def extractor() -> FeaturesExtractor:
    return FeaturesExtractor()


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.random.rand(10, 10)


def test_init_passes_params_to_pyradiomics():
    custom_params = {
        "minimumROISize": 50
    }
    
    ext = FeaturesExtractor(params=custom_params)
    assert ext._extractor.settings["minimumROISize"] == 50


def test_execute_extracts_features_and_cleans_diagnostics(extractor: FeaturesExtractor, sample_image: np.ndarray):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:4, 0:4] = 1
    mask[5:9, 5:9] = 2
    
    df = extractor.execute(sample_image, mask)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.columns[0] == "habitat_id"
    assert "diagnostics_Versions_PyRadiomics" not in df.columns
    assert "original_firstorder_Mean" in df.columns
    assert df.loc[0, "habitat_id"] == 1
    assert df.loc[1, "habitat_id"] == 2


def test_execute_handles_pyradiomics_exceptions_gracefully(extractor: FeaturesExtractor, sample_image: np.ndarray):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:1, 0:1] = 1  
    mask[5:9, 5:9] = 2  
    
    df = extractor.execute(sample_image, mask)
    
    assert len(df) == 2
    
    failed_habitat = df[df["habitat_id"] == 1].iloc[0]
    assert pd.isna(failed_habitat.get("original_firstorder_Mean"))
    
    success_habitat = df[df["habitat_id"] == 2].iloc[0]
    assert not pd.isna(success_habitat.get("original_firstorder_Mean"))


def test_execute_returns_empty_dataframe_for_empty_mask(extractor: FeaturesExtractor, sample_image: np.ndarray):
    empty_mask = np.zeros((10, 10), dtype=np.uint8)
    
    df = extractor.execute(sample_image, empty_mask)
    
    assert isinstance(df, pd.DataFrame)
    assert df.empty