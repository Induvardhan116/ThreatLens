from datetime import date, datetime

import pytest
from pydantic import ValidationError

from pipeline.schema import VulnerabilityRecord


def test_valid_vulnerability_record():
    record = VulnerabilityRecord(
        cve_id="CVE-2024-12345",
        description="Example vulnerability.",
        published_date=datetime(2024, 1, 1),
        cvss_v4_score=8.8,
        cvss_v4_vector="CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H",
        severity="HIGH",
        cwe_ids=["CWE-79"],
        affected_cpes=[
            "cpe:2.3:a:example:product:1.0:*:*:*:*:*:*:*"
        ],
        references=["https://example.com/advisory"],
        known_exploited=True,
        kev_date_added=date(2024, 1, 10),
        attack_technique_ids=["T1190"],
        reference_count=1,
        affected_cpe_count=1,
        cwe_count=1,
        attack_technique_count=1,
        description_length=25,
        days_since_publication=10,
        dataset_split="train",
        exploitation_label=1,
    )

    assert record.cve_id == "CVE-2024-12345"
    assert record.cvss_v4_score == 8.8
    assert record.known_exploited is True


def test_invalid_cve_id():
    with pytest.raises(ValidationError):
        VulnerabilityRecord(
            cve_id="NOT-A-CVE",
        )


def test_cvss_score_must_be_between_zero_and_ten():
    with pytest.raises(ValidationError):
        VulnerabilityRecord(
            cve_id="CVE-2024-12345",
            cvss_v4_score=11.0,
        )


def test_negative_derived_count_is_rejected():
    with pytest.raises(ValidationError):
        VulnerabilityRecord(
            cve_id="CVE-2024-12345",
            reference_count=-1,
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        VulnerabilityRecord(
            cve_id="CVE-2024-12345",
            made_up_field="should fail",
        )
