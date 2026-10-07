import csv
import json
import re
from pathlib import Path

from fastapi import UploadFile

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS = BASE_DIR / "data" / "processed" / "inference" / "threatlens_predictions.csv"
ENRICHED = BASE_DIR / "data" / "processed" / "enriched" / "threatlens_dataset.json"

CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)


def extract_cves(text: str):
    found = {
        value.upper()
        for value in CVE_PATTERN.findall(text or "")
    }
    return sorted(found)


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".csv", ".json", ".md", ".log"}:
        return content.decode("utf-8", errors="replace")

    if suffix == ".pdf":
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))

        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    raise ValueError(
        "Unsupported file type. Use PDF, TXT, CSV, JSON, MD or LOG."
    )


def load_prediction_index():
    if not PREDICTIONS.exists():
        return {}

    index = {}

    with PREDICTIONS.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            cve = str(row.get("cve_id", "")).upper().strip()

            if cve:
                index[cve] = row

    return index


def load_enriched_index():
    if not ENRICHED.exists():
        return {}

    with ENRICHED.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    index = {}

    for record in records:
        cve = str(record.get("cve_id", "")).upper().strip()

        if cve:
            index[cve] = record

    return index


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def verify_cve(cve_id, predictions, enriched):
    prediction = predictions.get(cve_id)
    record = enriched.get(cve_id)

    if not prediction and not record:
        return {
            "cve_id": cve_id,
            "status": "UNVERIFIED",
            "reason": "CVE was not found in the available ThreatLens evidence dataset.",
            "evidence": [],
        }

    evidence = []
    conflicts = []

    if record:
        evidence.append({
            "type": "NVD_RECORD",
            "status": "VERIFIED",
            "detail": "Vulnerability record exists in the ThreatLens normalized dataset.",
        })

        if record.get("cwe_ids"):
            evidence.append({
                "type": "CWE",
                "status": "VERIFIED",
                "detail": ", ".join(record.get("cwe_ids", [])),
            })

        if record.get("affected_cpes"):
            evidence.append({
                "type": "AFFECTED_PRODUCTS",
                "status": "VERIFIED",
                "detail": f"{len(record.get('affected_cpes', []))} affected CPE entries.",
            })

        if record.get("references"):
            evidence.append({
                "type": "REFERENCES",
                "status": "VERIFIED",
                "detail": f"{len(record.get('references', []))} references available.",
            })

        known_exploited = record.get("known_exploited")

        if known_exploited is True:
            evidence.append({
                "type": "CISA_KEV",
                "status": "VERIFIED",
                "detail": "CVE is present in the ThreatLens CISA KEV enrichment.",
            })
        elif known_exploited is False:
            evidence.append({
                "type": "CISA_KEV",
                "status": "VERIFIED",
                "detail": "CVE is not marked as known exploited in the current KEV enrichment.",
            })

    result = {
        "cve_id": cve_id,
        "status": "VERIFIED",
        "evidence": evidence,
        "conflicts": conflicts,
    }

    if prediction:
        result["threatlens_score"] = safe_float(
            prediction.get("threatlens_score")
        )
        result["risk_category"] = prediction.get("risk_category")

        evidence.append({
            "type": "THREATLENS_MODEL",
            "status": "VERIFIED",
            "detail": (
                f"Production ThreatLens classification: "
                f"{prediction.get('risk_category')} "
                f"with score {prediction.get('threatlens_score')}."
            ),
        })

    return result


async def verify_upload(upload: UploadFile):
    content = await upload.read()

    if len(content) > 10 * 1024 * 1024:
        raise ValueError("File exceeds the 10 MB upload limit.")

    text = extract_text(upload.filename or "upload.txt", content)

    cves = extract_cves(text)

    predictions = load_prediction_index()
    enriched = load_enriched_index()

    results = [
        verify_cve(cve, predictions, enriched)
        for cve in cves
    ]

    verified = sum(
        1 for item in results
        if item["status"] == "VERIFIED"
    )

    return {
        "filename": upload.filename,
        "file_size": len(content),
        "cves_found": len(cves),
        "verified": verified,
        "unverified": len(results) - verified,
        "results": results,
        "verification_mode": "ThreatLens evidence verification",
        "ai_status": "Evidence verification complete; AI explanation can be requested from verified evidence.",
    }
