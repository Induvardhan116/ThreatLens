from pathlib import Path

import pytest

from backend.config import PROJECT_ROOT, resolve_data_dir
from backend.app import DATA_DIR as API_DATA_DIR
from backend.app import DEFAULT_CORS_ORIGINS, configured_cors_origins
from evidence.evidence_engine import DATA_DIR as EVIDENCE_DATA_DIR


def test_data_directory_defaults_to_project_data():
    assert resolve_data_dir("") == PROJECT_ROOT / "data"


def test_api_and_evidence_engine_use_the_same_data_directory():
    assert API_DATA_DIR == EVIDENCE_DATA_DIR == resolve_data_dir()


def test_relative_data_directory_is_resolved_from_project_root():
    assert resolve_data_dir("runtime-data") == (PROJECT_ROOT / "runtime-data").resolve()


def test_absolute_data_directory_is_preserved():
    assert resolve_data_dir(str(Path("C:/threatlens-data"))) == Path(
        "C:/threatlens-data"
    ).resolve()


def test_cors_origins_default_to_local_development_origins():
    assert configured_cors_origins("") == list(DEFAULT_CORS_ORIGINS)


def test_cors_origins_parse_and_normalize_urls():
    assert configured_cors_origins(
        " https://threatlens.example, http://localhost:5173/ "
    ) == [
        "https://threatlens.example",
        "http://localhost:5173",
    ]


@pytest.mark.parametrize(
    "value",
    [
        "https://threatlens.example/path",
        "https://threatlens.example?preview=1",
        "ftp://threatlens.example",
        "https://",
        "https://threatlens.example:invalid",
    ],
)
def test_cors_origins_reject_non_origins(value):
    with pytest.raises(ValueError, match="THREATLENS_CORS_ORIGINS"):
        configured_cors_origins(value)
