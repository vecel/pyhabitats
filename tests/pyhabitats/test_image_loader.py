import numpy as np
import pytest
import SimpleITK as sitk

from pyhabitats.image_loader import ImageLoader


@pytest.fixture
def loader() -> ImageLoader:
    return ImageLoader()


@pytest.fixture
def supported_file(tmp_path: pytest.TempPathFactory) -> str:
    file_path = tmp_path / "supported_image.nii"
    
    image = sitk.Image(10, 10, sitk.sitkUInt8)
    sitk.WriteImage(image, str(file_path))
    
    return str(file_path)

@pytest.fixture
def unsupported_file(tmp_path: pytest.TempPathFactory) -> str:
    file_path = tmp_path / "unsupported_image.txt"
    file_path.write_text("This is not a supported image format.")

    return str(file_path)

@pytest.fixture
def image(request, tmp_path: pytest.TempPathFactory) -> str:
    file_path = tmp_path / "image.nii"
    
    shape = request.param 
    image = sitk.Image(shape, sitk.sitkUInt8)
    sitk.WriteImage(image, str(file_path))
    
    return str(file_path)

@pytest.fixture
def directory(request, tmp_path: pytest.TempPathFactory) -> str:
    directory = tmp_path / "scans"
    directory.mkdir()

    shapes = request.param
    for i, shape in enumerate(shapes):
        image = sitk.Image(shape[::-1], sitk.sitkUInt8)
        sitk.WriteImage(image, str(directory / f"image_{i}.nii"))

    return str(directory)


def test_raises_error_when_file_does_not_exist(loader: ImageLoader, tmp_path: pytest.TempPathFactory) -> None:
    missing_path = tmp_path / "does_not_exist.jpg"
    
    with pytest.raises(FileNotFoundError, match=f"File not found: {missing_path}"):
        loader.load_image(missing_path)

def test_raises_error_when_directory_is_empty(loader: ImageLoader, tmp_path: pytest.TempPathFactory) -> None:
    empty_directory = tmp_path / "empty_dir"
    empty_directory.mkdir()

    with pytest.raises(FileNotFoundError, match=f"No files found in directory: {empty_directory}"):
        loader.load_image(empty_directory)

def test_loads_image_from_file(loader: ImageLoader, supported_file: str) -> None:
    result = loader.load_image(supported_file)
    assert isinstance(result, np.ndarray)

def test_raises_error_for_unsupported_file_format(loader: ImageLoader, unsupported_file: str) -> None:
    with pytest.raises(ValueError, match=f"Failed to read image file: {unsupported_file}"):
        loader.load_image(unsupported_file)

@pytest.mark.parametrize("directory", [[(3, 4)]], indirect=True)
def test_loads_single_image_from_directory(loader: ImageLoader, directory: str) -> None:
    result = loader.load_image(directory)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (3, 4)

@pytest.mark.parametrize("image", [[128, 128]], indirect=True)
def test_loads_2d_data(loader, image):
    result = loader.load_image(image)
    assert result.ndim == 2
    assert result.shape == (128, 128)

@pytest.mark.parametrize("image", [[32, 32, 32]], indirect=True)
def test_loads_3d_data(loader, image):
    result = loader.load_image(image)
    assert result.ndim == 3
    assert result.shape == (32, 32, 32)

@pytest.mark.parametrize("directory", [2 * [(5, 8)]], indirect=True)
def test_stacks_2d_images_into_3d_when_loading_from_directory(loader, directory):
    result = loader.load_image(directory)
    assert result.ndim == 3
    assert result.shape == (5, 8, 2)

@pytest.mark.parametrize("directory", [3 * [(4, 5, 6)]], indirect=True)
def test_stacks_3d_images_into_4d_when_loading_from_directory(loader, directory):
    result = loader.load_image(directory)
    assert result.ndim == 4
    assert result.shape == (4, 5, 6, 3)

@pytest.mark.parametrize("directory", [[(10, 20), (5, 5)]], indirect=True)
def test_raises_error_when_images_in_directory_differ_in_shape(loader, directory):
    with pytest.raises(ValueError, match="All input images must have the same shape. Found shapes: \\(10, 20\\), \\(5, 5\\)."):
        loader.load_image(directory)