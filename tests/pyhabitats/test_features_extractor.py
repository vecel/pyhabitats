import numpy as np
import pandas as pd
import pytest

from pyhabitats.features_extractor import FeaturesExtractor


@pytest.fixture
def extractor() -> FeaturesExtractor:
    return FeaturesExtractor()


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.ones((10, 10, 10), dtype=np.float32)

@pytest.fixture
def sample_mask() -> np.ndarray:
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:5, 2:5, 2:5] = 1
    mask[6:9, 6:9, 6:9] = 2
    return mask

def test_execute_extracts_features(extractor: FeaturesExtractor, sample_image: np.ndarray):
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[0:4, 0:4, 0:4] = 1
    mask[5:9, 5:9, 5:9] = 2
    
    df = extractor.execute(sample_image, mask)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.columns[0] == "habitat_id"
    assert len(df.columns) > 1
    assert df.loc[0, "habitat_id"] == 1
    assert df.loc[1, "habitat_id"] == 2

def test_execute_handles_extraction_exceptions(extractor: FeaturesExtractor, sample_image: np.ndarray):
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[0:1, 0:1, 0:1] = 1 
    mask[5:9, 5:9, 5:9] = 2 
    
    df = extractor.execute(sample_image, mask)
    
    assert len(df) == 2
    
    feature_cols = [col for col in df.columns if col != "habitat_id"]
    assert len(feature_cols) > 0, "No features were extracted for the successful habitat"
    
    sample_feature = feature_cols[0]
    
    failed_habitat = df[df["habitat_id"] == 1].iloc[0]
    assert pd.isna(failed_habitat.get(sample_feature))
    
    success_habitat = df[df["habitat_id"] == 2].iloc[0]
    assert not pd.isna(success_habitat.get(sample_feature))


def test_execute_raises_error_for_empty_mask(extractor: FeaturesExtractor, sample_image: np.ndarray):
    empty_mask = np.zeros_like(sample_image)
    with pytest.raises(ValueError, match="Mask does not contain ROI"):
        extractor.execute(sample_image, empty_mask)