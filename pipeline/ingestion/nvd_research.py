from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    HTTPError,
    RequestException,
    SSLError,
    Timeout,
)


NVD_API_URL = (
    "https://services.nvd.nist.gov/rest/json/cves/2.0"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nvd"
    / "research"
)

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    RAW_DATA_DIR
    / "nvd_2024_2026.json"
)

WINDOW_DIR = (
    RAW_DATA_DIR
    / "windows"
)

WINDOW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


START_DATE = datetime(
    2024,
    1,
    1,
    0,
    0,
    0,
    tzinfo=timezone.utc,
)

END_DATE = datetime(
    2026,
    9,
    9,
    23,
    59,
    59,
    999000,
    tzinfo=timezone.utc,
)

WINDOW_DAYS = 120

RESULTS_PER_PAGE = 2000

REQUEST_DELAY_SECONDS = 6

TIMEOUT_SECONDS = 180

MAX_RETRIES = 8


def format_nvd_datetime(
    value: datetime,
) -> str:

    value = value.astimezone(
        timezone.utc
    )

    return value.strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )


def build_date_windows() -> list[
    tuple[datetime, datetime]
]:

    windows: list[
        tuple[datetime, datetime]
    ] = []

    current_start = START_DATE

    while current_start <= END_DATE:

        candidate_end = (
            current_start
            + timedelta(days=WINDOW_DAYS)
            - timedelta(milliseconds=1)
        )

        current_end = min(
            candidate_end,
            END_DATE,
        )

        windows.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = (
            current_end
            + timedelta(milliseconds=1)
        )

    return windows


def window_file(
    window_number: int,
) -> Path:

    return (
        WINDOW_DIR
        / f"window_{window_number:02d}.json"
    )


def save_window(
    window_number: int,
    start_date: datetime,
    end_date: datetime,
    records: list[dict[str, Any]],
) -> None:

    output = {
        "source": "NVD",
        "window_number": window_number,
        "start": format_nvd_datetime(
            start_date
        ),
        "end": format_nvd_datetime(
            end_date
        ),
        "records": records,
    }

    path = window_file(
        window_number
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_completed_window(
    window_number: int,
) -> list[dict[str, Any]] | None:

    path = window_file(
        window_number
    )

    if not path.exists():
        return None

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        records = data.get(
            "records"
        )

        if not isinstance(
            records,
            list,
        ):
            return None

        return records

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return None


def request_page(
    start_date: datetime,
    end_date: datetime,
    start_index: int,
) -> dict[str, Any]:

    params = {
        "pubStartDate": format_nvd_datetime(
            start_date
        ),
        "pubEndDate": format_nvd_datetime(
            end_date
        ),
        "startIndex": start_index,
        "resultsPerPage": RESULTS_PER_PAGE,
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = requests.get(
                NVD_API_URL,
                params=params,
                timeout=TIMEOUT_SECONDS,
                headers={
                    "User-Agent": (
                        "ThreatLens/1.0 "
                        "(cybersecurity research)"
                    ),
                    "Accept": (
                        "application/json"
                    ),
                    "Connection": "close",
                },
            )

            if response.status_code == 200:

                return response.json()

            if response.status_code == 429:

                wait_time = min(
                    60 * attempt,
                    300,
                )

                print(
                    f"Rate limited "
                    f"(attempt "
                    f"{attempt}/{MAX_RETRIES})."
                )

                print(
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

                continue

            if response.status_code in {
                500,
                502,
                503,
                504,
            }:

                wait_time = min(
                    20 * attempt,
                    180,
                )

                print(
                    f"Server error "
                    f"{response.status_code} "
                    f"(attempt "
                    f"{attempt}/{MAX_RETRIES})."
                )

                print(
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

                continue

            response.raise_for_status()

        except (
            ChunkedEncodingError,
            ConnectionError,
            Timeout,
            SSLError,
        ) as exc:

            if attempt == MAX_RETRIES:
                raise

            wait_time = min(
                15 * attempt,
                180,
            )

            print()
            print(
                f"Network/stream error "
                f"(attempt "
                f"{attempt}/{MAX_RETRIES}):"
            )

            print(
                f"  {type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"Retrying in "
                f"{wait_time} seconds..."
            )

            time.sleep(
                wait_time
            )

        except HTTPError:

            raise

        except RequestException as exc:

            if attempt == MAX_RETRIES:
                raise

            wait_time = min(
                15 * attempt,
                180,
            )

            print(
                f"Request error "
                f"(attempt "
                f"{attempt}/{MAX_RETRIES}):"
            )

            print(exc)

            time.sleep(
                wait_time
            )

    raise RuntimeError(
        "NVD request failed after "
        "maximum retries."
    )


def download_window(
    window_number: int,
    total_windows: int,
    start_date: datetime,
    end_date: datetime,
) -> list[dict[str, Any]]:

    existing = load_completed_window(
        window_number
    )

    if existing is not None:

        print()
        print(
            "=" * 65
        )

        print(
            f"WINDOW "
            f"{window_number}/{total_windows} "
            f"ALREADY COMPLETE"
        )

        print(
            "Records:",
            len(existing),
        )

        print(
            "=" * 65
        )

        return existing

    print()
    print(
        "=" * 65
    )

    print(
        f"WINDOW "
        f"{window_number}/{total_windows}"
    )

    print(
        "Start:",
        format_nvd_datetime(
            start_date
        ),
    )

    print(
        "End:  ",
        format_nvd_datetime(
            end_date
        ),
    )

    print(
        "=" * 65
    )

    records: list[
        dict[str, Any]
    ] = []

    start_index = 0

    total_results: int | None = None

    page_number = 0

    while True:

        page_number += 1

        print()
        print(
            f"Requesting page "
            f"{page_number} "
            f"(startIndex="
            f"{start_index})..."
        )

        data = request_page(
            start_date=start_date,
            end_date=end_date,
            start_index=start_index,
        )

        vulnerabilities = data.get(
            "vulnerabilities",
            [],
        )

        if not isinstance(
            vulnerabilities,
            list,
        ):

            raise ValueError(
                "Invalid NVD "
                "'vulnerabilities' field."
            )

        if total_results is None:

            total_results = data.get(
                "totalResults"
            )

            print(
                "Window total results:",
                total_results,
            )

        records.extend(
            vulnerabilities
        )

        print(
            "Window records collected:",
            len(records),
        )

        if not vulnerabilities:
            break

        if (
            total_results is not None
            and len(records)
            >= total_results
        ):
            break

        start_index += len(
            vulnerabilities
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    save_window(
        window_number=window_number,
        start_date=start_date,
        end_date=end_date,
        records=records,
    )

    print()
    print(
        f"Window {window_number} "
        f"saved successfully."
    )

    return records


def deduplicate_records(
    vulnerabilities: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:

    unique: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in vulnerabilities:

        cve = (
            item
            .get("cve", {})
            .get("id")
        )

        if not cve:
            continue

        unique[cve] = item

    return list(
        unique.values()
    )


def main() -> None:

    print(
        "ThreatLens - NVD research "
        "dataset ingestion"
    )

    print(
        "=" * 65
    )

    print()
    print(
        "Research period:"
    )

    print(
        "  Start:",
        format_nvd_datetime(
            START_DATE
        ),
    )

    print(
        "  End:  ",
        format_nvd_datetime(
            END_DATE
        ),
    )

    windows = build_date_windows()

    print()
    print(
        "NVD-compatible windows:",
        len(windows),
    )

    print(
        "Maximum window size:",
        WINDOW_DAYS,
        "days",
    )

    all_vulnerabilities: list[
        dict[str, Any]
    ] = []

    for index, (
        window_start,
        window_end,
    ) in enumerate(
        windows,
        start=1,
    ):

        window_records = (
            download_window(
                window_number=index,
                total_windows=len(
                    windows
                ),
                start_date=window_start,
                end_date=window_end,
            )
        )

        all_vulnerabilities.extend(
            window_records
        )

        print()
        print(
            f"Total collected so far:",
            len(
                all_vulnerabilities
            ),
        )

    print()
    print(
        "=" * 65
    )

    print(
        "DEDUPLICATION"
    )

    print(
        "=" * 65
    )

    records_before = len(
        all_vulnerabilities
    )

    unique_vulnerabilities = (
        deduplicate_records(
            all_vulnerabilities
        )
    )

    records_after = len(
        unique_vulnerabilities
    )

    print(
        "Records before deduplication:",
        records_before,
    )

    print(
        "Unique CVEs:",
        records_after,
    )

    print(
        "Duplicates removed:",
        records_before
        - records_after,
    )

    output = {
        "source": "NVD",
        "api": NVD_API_URL,
        "date_range": {
            "start": format_nvd_datetime(
                START_DATE
            ),
            "end": format_nvd_datetime(
                END_DATE
            ),
        },
        "window_days": WINDOW_DAYS,
        "downloaded_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "records_collected": (
            records_after
        ),
        "vulnerabilities": (
            unique_vulnerabilities
        ),
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=" * 65
    )

    print(
        "NVD INGESTION COMPLETE"
    )

    print(
        "=" * 65
    )

    print(
        "Unique CVEs:",
        records_after,
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
