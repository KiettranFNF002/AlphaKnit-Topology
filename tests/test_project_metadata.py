import tomllib
from packaging.version import Version


def test_pyproject_version_is_pep440():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    Version(data["project"]["version"])


def test_pytest_collection_is_scoped_to_tests_dir():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
