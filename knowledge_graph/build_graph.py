from pathlib import Path
import json
import csv
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ENRICHED_PATH = (
    ROOT / "data" / "processed" / "enriched" /
    "threatlens_dataset.json"
)

PREDICTIONS_PATH = (
    ROOT / "data" / "processed" / "inference" /
    "threatlens_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "knowledge_graph"
)

GRAPH_PATH = OUTPUT_DIR / "threatlens_graph.graphml"
NODES_PATH = OUTPUT_DIR / "threatlens_graph_nodes.csv"
EDGES_PATH = OUTPUT_DIR / "threatlens_graph_edges.csv"
SUMMARY_PATH = OUTPUT_DIR / "threatlens_graph_summary.json"

# Number of CVEs per risk category.
PER_CATEGORY = 100


def load_json_records():
    with ENRICHED_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "records",
            "data",
            "vulnerabilities",
            "items",
        ):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Could not find vulnerability records in enriched dataset."
    )


def normalize_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    return [str(value).strip()]


def risk_category(score):
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


def safe_bool(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def add_node(graph, node_id, node_type, label, **attrs):
    clean_attrs = {
        key: "" if value is None else str(value)
        for key, value in attrs.items()
    }

    graph.add_node(
        node_id,
        node_type=node_type,
        label=str(label),
        **clean_attrs,
    )


def main():

    print("=" * 72)
    print("THREATLENS - STEP 28.1")
    print("CORRECTED THREAT KNOWLEDGE GRAPH BUILDER")
    print("=" * 72)

    if not ENRICHED_PATH.exists():
        raise FileNotFoundError(
            f"Enriched dataset not found: {ENRICHED_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction dataset not found: {PREDICTIONS_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("-" * 72)
    print("LOADING DATA")
    print("-" * 72)

    records = load_json_records()
    predictions = pd.read_csv(PREDICTIONS_PATH)

    print(f"Enriched records:       {len(records):,}")
    print(f"Prediction records:     {len(predictions):,}")

    required_prediction_columns = {
        "record_id",
        "threatlens_score",
        "risk_category",
    }

    missing = required_prediction_columns - set(
        predictions.columns
    )

    if missing:
        raise ValueError(
            f"Prediction file missing columns: {sorted(missing)}"
        )

    if len(predictions) != len(records):
        raise ValueError(
            "Prediction and enriched dataset row counts differ."
        )

    # Attach production predictions to the enriched records.
    prediction_map = {}

    for _, row in predictions.iterrows():

        record_id = int(row["record_id"])

        prediction_map[record_id] = {
            "score": float(row["threatlens_score"]),
            "risk_category": str(row["risk_category"]),
        }

    print("Prediction alignment: PASS")

    # Build combined records.
    combined = []

    for index, record in enumerate(records):

        record_id = index + 1

        prediction = prediction_map.get(record_id)

        if prediction is None:
            continue

        combined.append(
            {
                "record_id": record_id,
                "record": record,
                "score": prediction["score"],
                "risk_category": prediction[
                    "risk_category"
                ],
            }
        )

    print(f"Aligned records:       {len(combined):,}")

    # Select records by authoritative production risk category.
    selected = []

    categories = [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
        "MINIMAL",
    ]

    print()
    print("-" * 72)
    print("CATEGORY SELECTION")
    print("-" * 72)

    for category in categories:

        candidates = [
            item
            for item in combined
            if item["risk_category"] == category
        ]

        candidates.sort(
            key=lambda item: (
                item["score"],
                safe_bool(
                    item["record"].get(
                        "known_exploited"
                    )
                ),
            ),
            reverse=True,
        )

        chosen = candidates[:PER_CATEGORY]

        selected.extend(chosen)

        print(
            f"{category:<10} "
            f"available={len(candidates):>7,} "
            f"selected={len(chosen):>5,}"
        )

    print()
    print(f"Total CVEs selected:   {len(selected):,}")

    graph = nx.MultiDiGraph()

    counters = {
        "CVE": 0,
        "CWE": 0,
        "CPE": 0,
        "Reference": 0,
        "RiskCategory": 0,
        "KEV": 0,
    }

    # Track unique nodes to avoid incorrect counts.
    seen_types = {
        "CVE": set(),
        "CWE": set(),
        "CPE": set(),
        "Reference": set(),
        "RiskCategory": set(),
        "KEV": set(),
    }

    for item in selected:

        record = item["record"]

        cve_id = str(
            record.get("cve_id", "")
        ).strip()

        if not cve_id:
            continue

        score = item["score"]
        category = item["risk_category"]

        cve_node = f"CVE:{cve_id}"

        add_node(
            graph,
            cve_node,
            "CVE",
            cve_id,
            record_id=item["record_id"],
            threatlens_score=f"{score:.6f}",
            risk_category=category,
            severity=record.get("severity"),
            known_exploited=record.get(
                "known_exploited",
                False,
            ),
            kev_date_added=record.get(
                "kev_date_added"
            ),
            published_date=record.get(
                "published_date"
            ),
        )

        if cve_node not in seen_types["CVE"]:
            seen_types["CVE"].add(cve_node)

        # Risk category.
        risk_node = f"RISK:{category}"

        if risk_node not in seen_types["RiskCategory"]:

            add_node(
                graph,
                risk_node,
                "RiskCategory",
                category,
            )

            seen_types["RiskCategory"].add(
                risk_node
            )

        graph.add_edge(
            cve_node,
            risk_node,
            relation="HAS_RISK",
        )

        # Known exploited status.
        if safe_bool(
            record.get("known_exploited")
        ):

            kev_node = "SIGNAL:CISA_KEV"

            if kev_node not in seen_types["KEV"]:

                add_node(
                    graph,
                    kev_node,
                    "ThreatSignal",
                    "CISA Known Exploited Vulnerabilities",
                )

                seen_types["KEV"].add(kev_node)

            graph.add_edge(
                cve_node,
                kev_node,
                relation="KNOWN_EXPLOITED",
            )

        # CWE.
        for cwe in normalize_list(
            record.get("cwe_ids")
        ):

            cwe_id = cwe

            if not cwe_id.startswith("CWE-"):
                cwe_id = f"CWE-{cwe_id}"

            node = f"CWE:{cwe_id}"

            if node not in seen_types["CWE"]:

                add_node(
                    graph,
                    node,
                    "CWE",
                    cwe_id,
                )

                seen_types["CWE"].add(node)

            graph.add_edge(
                cve_node,
                node,
                relation="HAS_CWE",
            )

        # Affected CPE.
        for cpe in normalize_list(
            record.get("affected_cpes")
        ):

            node = f"CPE:{cpe}"

            if node not in seen_types["CPE"]:

                add_node(
                    graph,
                    node,
                    "CPE",
                    cpe,
                )

                seen_types["CPE"].add(node)

            graph.add_edge(
                cve_node,
                node,
                relation="AFFECTS",
            )

        # References.
        references = normalize_list(
            record.get("references")
        )

        for index, reference in enumerate(
            references
        ):

            reference_id = (
                f"{cve_id}:{index}"
            )

            node = f"REF:{reference_id}"

            if node not in seen_types["Reference"]:

                add_node(
                    graph,
                    node,
                    "Reference",
                    reference,
                    url=reference,
                )

                seen_types["Reference"].add(
                    node
                )

            graph.add_edge(
                cve_node,
                node,
                relation="REFERENCES",
            )

    counters["CVE"] = len(seen_types["CVE"])
    counters["CWE"] = len(seen_types["CWE"])
    counters["CPE"] = len(seen_types["CPE"])
    counters["Reference"] = len(
        seen_types["Reference"]
    )
    counters["RiskCategory"] = len(
        seen_types["RiskCategory"]
    )
    counters["KEV"] = len(seen_types["KEV"])

    # Save GraphML.
    nx.write_graphml(
        graph,
        GRAPH_PATH,
    )

    # Save nodes.
    with NODES_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "node_id",
                "node_type",
                "label",
                "record_id",
                "threatlens_score",
                "risk_category",
                "severity",
                "known_exploited",
                "published_date",
                "kev_date_added",
                "url",
            ]
        )

        for node_id, attrs in graph.nodes(
            data=True
        ):

            writer.writerow(
                [
                    node_id,
                    attrs.get(
                        "node_type",
                        "",
                    ),
                    attrs.get(
                        "label",
                        "",
                    ),
                    attrs.get(
                        "record_id",
                        "",
                    ),
                    attrs.get(
                        "threatlens_score",
                        "",
                    ),
                    attrs.get(
                        "risk_category",
                        "",
                    ),
                    attrs.get(
                        "severity",
                        "",
                    ),
                    attrs.get(
                        "known_exploited",
                        "",
                    ),
                    attrs.get(
                        "published_date",
                        "",
                    ),
                    attrs.get(
                        "kev_date_added",
                        "",
                    ),
                    attrs.get(
                        "url",
                        "",
                    ),
                ]
            )

    # Save edges.
    with EDGES_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "source",
                "target",
                "relation",
            ]
        )

        for source, target, attrs in graph.edges(
            data=True
        ):

            writer.writerow(
                [
                    source,
                    target,
                    attrs.get(
                        "relation",
                        "",
                    ),
                ]
            )

    risk_distribution = {}

    for _, attrs in graph.nodes(
        data=True
    ):

        if attrs.get("node_type") == "CVE":

            category = attrs.get(
                "risk_category",
                "UNKNOWN",
            )

            risk_distribution[category] = (
                risk_distribution.get(
                    category,
                    0,
                )
                + 1
            )

    relationship_distribution = {}

    for _, _, attrs in graph.edges(
        data=True
    ):

        relation = attrs.get(
            "relation",
            "UNKNOWN",
        )

        relationship_distribution[
            relation
        ] = (
            relationship_distribution.get(
                relation,
                0,
            )
            + 1
        )

    summary = {
        "project": "ThreatLens",
        "graph_version": "1.1",
        "selection_method": (
            "Production ThreatLens risk category "
            "with score ordering"
        ),
        "source_records": len(records),
        "prediction_records": len(predictions),
        "selected_cves": len(selected),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": counters,
        "risk_distribution": risk_distribution,
        "relationship_distribution": (
            relationship_distribution
        ),
        "attack_technique_nodes": 0,
        "attack_technique_note": (
            "No ATT&CK technique IDs are populated "
            "in the current enriched dataset."
        ),
        "outputs": {
            "graphml": str(GRAPH_PATH),
            "nodes": str(NODES_PATH),
            "edges": str(EDGES_PATH),
            "summary": str(SUMMARY_PATH),
        },
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("GRAPH BUILD COMPLETE")
    print("=" * 72)

    print(
        f"Nodes:                  "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Edges:                  "
        f"{graph.number_of_edges():,}"
    )

    print()
    print("Node types:")

    for node_type, count in counters.items():

        print(
            f"  {node_type:<18} "
            f"{count:,}"
        )

    print()
    print("Risk distribution:")

    for category in categories:

        print(
            f"  {category:<10} "
            f"{risk_distribution.get(category, 0):,}"
        )

    print()
    print("Relationships:")

    for relation, count in sorted(
        relationship_distribution.items()
    ):

        print(
            f"  {relation:<18} "
            f"{count:,}"
        )

    print()
    print("Saved:")

    print(
        f"  {GRAPH_PATH}"
    )

    print(
        f"  {NODES_PATH}"
    )

    print(
        f"  {EDGES_PATH}"
    )

    print(
        f"  {SUMMARY_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
