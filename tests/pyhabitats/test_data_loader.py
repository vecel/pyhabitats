import numpy as np
import pytest

from pyhabitats.data_loader import DataLoader


@pytest.fixture
def loader() -> DataLoader:
    return DataLoader()


@pytest.fixture
def directory(tmp_path: pytest.TempPathFactory) -> str:
    directory = tmp_path / "test_data"
    file1 = directory / "file1.nii"
    file2 = directory / "file2.nii"

    directory.mkdir()
    file1.touch()
    file2.touch()

    return str(directory)


def test_stacks_2d_arrays(
    loader: DataLoader, mocker: pytest.Mock, directory: str
) -> None:
    mocker.patch.object(
        loader,
        "_read_file",
        side_effect=[np.array([[1, 0], [0, 1]]), np.array([[1, 1], [0, 0]])],
    )

    result = loader.load(directory)

    assert result.shape == (2, 2, 2)
    assert np.array_equal(result[:, :, 0], np.array([[1, 0], [0, 1]]))
    assert np.array_equal(result[:, :, 1], np.array([[1, 1], [0, 0]]))


def test_concatenates_3d_arrays(
    loader: DataLoader, mocker: pytest.Mock, directory: str
) -> None:
    mocker.patch.object(
        loader,
        "_read_file",
        side_effect=[np.array([[[1, 0], [0, 1]]]), np.array([[[1, 1], [0, 0]]])],
    )

    result = loader.load(directory)

    assert result.shape == (1, 2, 4)
    assert np.array_equal(result[0, 0, :], np.array([1, 0, 1, 1]))
    assert np.array_equal(result[0, 1, :], np.array([0, 1, 0, 0]))
