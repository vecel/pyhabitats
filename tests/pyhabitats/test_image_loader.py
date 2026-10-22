import numpy as np
import pytest

from pyhabitats.image_loader import ImageLoader


@pytest.fixture
def loader() -> ImageLoader:
    return ImageLoader()


@pytest.fixture
def directory(tmp_path: pytest.TempPathFactory) -> str:
    directory = tmp_path / "test_data"
    file1 = directory / "file1.nii"
    file2 = directory / "file2.nii"

    directory.mkdir()
    file1.touch()
    file2.touch()

    return str(directory)

@pytest.fixture
def volume_image_fixture() -> np.ndarray:
    return np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


def test_raises_error_when_file_does_not_exist(loader: ImageLoader, tmp_path: pytest.TempPathFactory) -> None:
    missing_path = tmp_path / "does_not_exist.jpg"
    
    with pytest.raises(FileNotFoundError, match="File not found"):
        loader.load_image(missing_path)

def test_raises_error_when_directory_is_empty(loader: ImageLoader, tmp_path: pytest.TempPathFactory) -> None:
    empty_directory = tmp_path / "empty_dir"
    empty_directory.mkdir()

    with pytest.raises(FileNotFoundError, match="No files found in directory"):
        loader.load_image(empty_directory)

def test_stacks_2d_arrays(
    loader: ImageLoader, mocker: pytest.Mock, directory: str
) -> None:
    mocker.patch.object(
        loader,
        "_read_file",
        side_effect=[np.array([[1, 0], [0, 1]]), np.array([[1, 1], [0, 0]])],
    )

    result = loader.load_image(directory)

    assert result.shape == (2, 2, 2)
    assert np.array_equal(result[:, :, 0], np.array([[1, 0], [0, 1]]))
    assert np.array_equal(result[:, :, 1], np.array([[1, 1], [0, 0]]))


def test_stacks_3d_arrays(loader: ImageLoader, mocker: pytest.Mock, directory: str, volume_image_fixture) -> None:
    mocker.patch.object(
        loader,
        "_read_file",
        side_effect=[volume_image_fixture, volume_image_fixture],
    )

    result = loader.load_image(directory)

    assert result.shape == (2, 2, 3, 2)