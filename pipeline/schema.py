from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ======================================================================
# THREATLENS RISK CATEGORY CONTRACT
# ======================================================================

def risk_category(score: float) -> str:
    """
    Convert a ThreatLens score into the authoritative risk category.

    This is the single canonical category contract used by the
    prediction and API layers.

    Contract:
        CRITICAL >= 0.50
        HIGH     >= 0.20
        MEDIUM   >= 0.05
        LOW      >= 0.01
        MINIMAL  <  0.01
    """

    score = float(score)

    if score >= 0.50:
        return "CRITICAL"

    if score >= 0.20:
        return "HIGH"

    if score >= 0.05:
        return "MEDIUM"

    if score >= 0.01:
        return "LOW"

    return "MINIMAL"


# ----------------------------------------------------------------------
# Canonical category ordering
# ----------------------------------------------------------------------

RISK_CATEGORIES = [
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
    "MINIMAL",
]


class VulnerabilityRecord(BaseModel):
    """
    Canonical ThreatLens vulnerability representation.

    The model intentionally separates source data, derived context,
    and evaluation-related information so that feature engineering
    can avoid accidental target leakage.
    """

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    cve_id: str = Field(
        ...,
        pattern=r"^CVE-\d{4}-\d{4,}$",
        description="Canonical CVE identifier.",
    )

    # ------------------------------------------------------------------
    # NVD / CVE source information
    # ------------------------------------------------------------------

    description: Optional[str] = None

    published_date: Optional[datetime] = None
    last_modified_date: Optional[datetime] = None

    cvss_v4_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
    )

    cvss_v4_vector: Optional[str] = None

    cvss_v3_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
    )

    cvss_v3_vector: Optional[str] = None

    severity: Optional[str] = None

    cwe_ids: list[str] = Field(
        default_factory=list
    )

    affected_cpes: list[str] = Field(
        default_factory=list
    )

    references: list[str] = Field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # CISA KEV source information
    # ------------------------------------------------------------------

    known_exploited: bool = False

    kev_date_added: Optional[date] = None
    kev_due_date: Optional[date] = None

    kev_vendor_project: Optional[str] = None
    kev_product: Optional[str] = None
    kev_vulnerability_name: Optional[str] = None

    kev_required_action: Optional[str] = None

    ransomware_use: Optional[bool] = None

    # ------------------------------------------------------------------
    # MITRE ATT&CK relationships
    # ------------------------------------------------------------------

    attack_technique_ids: list[str] = Field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Derived contextual features
    # ------------------------------------------------------------------

    reference_count: int = Field(
        default=0,
        ge=0,
    )

    affected_cpe_count: int = Field(
        default=0,
        ge=0,
    )

    cwe_count: int = Field(
        default=0,
        ge=0,
    )

    attack_technique_count: int = Field(
        default=0,
        ge=0,
    )

    description_length: int = Field(
        default=0,
        ge=0,
    )

    days_since_publication: Optional[int] = Field(
        default=None,
        ge=0,
    )

    # ------------------------------------------------------------------
    # Evaluation metadata
    # ------------------------------------------------------------------

    dataset_split: Optional[str] = None

    # This field is deliberately NOT used as a model feature.
    # It exists only for evaluation and ground-truth analysis.
    exploitation_label: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
    )
