import pytest
import numpy as np

from pyhabitats.habitats_extractor import HabitatsExtractor

@pytest.fixture
def extractor() -> HabitatsExtractor:
    return HabitatsExtractor()

@pytest.fixture
def image_fixture() -> np.ndarray:
    return np.array([[1, 2], [3, 4]])

@pytest.fixture
def mask_fixture() -> np.ndarray:
    return np.array([[1, 0], [0, 1]])


@pytest.mark.parametrize("invalid_image", [123, None, [], {}, set()])
def test_raises_type_error_when_image_input_has_invalid_type(extractor, invalid_image, mask_fixture):
    with pytest.raises(TypeError, match="image_input"):
        extractor.execute(
            image_input=invalid_image,
            mask_input=mask_fixture,
            habitats_number=3,
            algorithm="kmeans"
        )

@pytest.mark.parametrize("invalid_mask", [123, None, [], {}, set()])
def test_raises_type_error_when_mask_input_has_invalid_type(extractor, image_fixture, invalid_mask):
    with pytest.raises(TypeError, match="mask_input"):
        extractor.execute(
            image_input=image_fixture,
            mask_input=invalid_mask,
            habitats_number=3,
            algorithm="kmeans"
        )

@pytest.mark.parametrize("invalid_habitats", [3.14, "3", None, [], True])
def test_raises_type_error_when_habitats_number_has_invalid_type(extractor, image_fixture, mask_fixture, invalid_habitats):
    with pytest.raises(TypeError, match="habitats_number"):
        extractor.execute(
            image_input=image_fixture,
            mask_input=mask_fixture,
            habitats_number=invalid_habitats,
            algorithm="kmeans"
        )

@pytest.mark.parametrize("invalid_algorithm", [123, None, [], {}])
def test_raises_type_error_when_algorithm_has_invalid_type(extractor, image_fixture, mask_fixture, invalid_algorithm):
    with pytest.raises(TypeError, match="algorithm"):
        extractor.execute(
            image_input=image_fixture,
            mask_input=mask_fixture,
            habitats_number=3,
            algorithm=invalid_algorithm
        )

def test_provides_supported_algorithms_list_when_algorithm_is_not_supported(extractor, image_fixture, mask_fixture):
     with pytest.raises(ValueError, match=str(extractor._supported_algorithms)):
        extractor.execute(
            image_input=image_fixture,
            mask_input=mask_fixture,
            habitats_number=3,
            algorithm="unsupported_algorithm"
        )