import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_dir(configured_dir: Optional[str] = None) -> Path:
    value = configured_dir
    if value is None:
        value = os.getenv("THREATLENS_DATA_DIR")

    path = (
        Path(value).expanduser()
        if value and value.strip()
        else PROJECT_ROOT / "data"
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()
