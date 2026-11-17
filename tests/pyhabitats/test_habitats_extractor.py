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

@pytest.fixture
def shaped_image(request):
    shape = request.param
    return np.zeros(shape)

@pytest.fixture
def shaped_mask(request):
    shape = request.param
    return np.ones(shape)
    

@pytest.mark.parametrize("invalid_image", [123, None, [], {}, set()])
def test_raises_type_error_when_image_input_has_invalid_type(extractor, invalid_image, mask_fixture):
    with pytest.raises(TypeError, match="image argument must be a numpy array"):
        extractor.execute(
            image=invalid_image,
            mask=mask_fixture,
            habitats_number=3,
            algorithm="kmeans"
        )

@pytest.mark.parametrize("invalid_mask", [123, None, [], {}, set()])
def test_raises_type_error_when_mask_input_has_invalid_type(extractor, image_fixture, invalid_mask):
    with pytest.raises(TypeError, match="mask argument must be a numpy array"):
        extractor.execute(
            image=image_fixture,
            mask=invalid_mask,
            habitats_number=3,
            algorithm="kmeans"
        )

@pytest.mark.parametrize("invalid_habitats", [3.14, "3", None, [], True])
def test_raises_type_error_when_habitats_number_has_invalid_type(extractor, image_fixture, mask_fixture, invalid_habitats):
    with pytest.raises(TypeError, match="habitats_number must be an integer"):
        extractor.execute(
            image=image_fixture,
            mask=mask_fixture,
            habitats_number=invalid_habitats,
            algorithm="kmeans"
        )

@pytest.mark.parametrize("invalid_algorithm", [123, None, [], {}])
def test_raises_type_error_when_algorithm_has_invalid_type(extractor, image_fixture, mask_fixture, invalid_algorithm):
    with pytest.raises(TypeError, match="algorithm must be"):
        extractor.execute(
            image=image_fixture,
            mask=mask_fixture,
            habitats_number=3,
            algorithm=invalid_algorithm
        )

def test_provides_supported_algorithms_list_when_algorithm_is_not_supported(extractor, image_fixture, mask_fixture):
     with pytest.raises(ValueError, match=str(extractor._supported_algorithms)):
        extractor.execute(
            image=image_fixture,
            mask=mask_fixture,
            habitats_number=3,
            algorithm="unsupported_algorithm"
        )

@pytest.mark.parametrize("shaped_image,shaped_mask", [((4, 4, 2, 3), (4, 4)), ((2, 6), (4, 4, 4))], indirect=True)
def test_raises_value_error_when_image_and_mask_have_incompatible_dimensions(extractor, shaped_image, shaped_mask):
    with pytest.raises(ValueError, match="image must have either same dimensions as mask or one more"):
        extractor.execute(
            shaped_image,
            shaped_mask,
            2,
            "kmeans"
        )

@pytest.mark.parametrize("shaped_image,shaped_mask", [((4, 2), (4, 4))], indirect=True)
def test_raises_value_error_when_image_shape_is_incompatible_with_mask_shape(extractor, shaped_image, shaped_mask):
    with pytest.raises(ValueError, match="image leading dimensions have to match mask dimensions"):
        extractor.execute(
            shaped_image,
            shaped_mask,
            2,
            "kmeans"
        )

def test_fits_algorithm(extractor, image_fixture, mask_fixture):
    habitats = extractor.execute(image_fixture, mask_fixture, habitats_number=2, algorithm="kmeans")
    assert habitats.shape == (2, 2)
    assert len(np.unique(habitats)) == 3

@pytest.mark.parametrize("shaped_image,shaped_mask", [((4, 6, 2), (4, 6))], indirect=True)
def test_habitats_mask_is_coherent_with_input_mask(extractor, shaped_image, shaped_mask):
    habitats = extractor.execute(shaped_image, shaped_mask, habitats_number=2, algorithm="kmeans")
    assert habitats.shape == shaped_mask.shape

def test_fits_custom_clusterization_algorithm(extractor, image_fixture, mask_fixture, one_habitat_clusterer):
    habitats = extractor.execute(image_fixture, mask_fixture, habitats_number=2, algorithm=one_habitat_clusterer)
    # No asserts, we just want this test to pass without raising any exception

def test_returns_mask_when_habitats_number_is_one(extractor, image_fixture, mask_fixture):
    habitats = extractor.execute(image_fixture, mask_fixture, habitats_number=1, algorithm="kmeans")
    assert np.array_equal(habitats, mask_fixture)

@pytest.mark.parametrize("shaped_image,shaped_mask", [((4, 6, 10), (4, 6))], indirect=True)
def test_fits_algorithm_for_high_dimensional_data(extractor, shaped_image, shaped_mask):
    habitats = extractor.execute(shaped_image, shaped_mask, habitats_number=3, algorithm="kmeans")
    assert habitats.shape == (4, 6)
