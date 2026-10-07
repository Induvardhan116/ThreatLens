from pathlib import Path
import json
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = (
    ROOT / "data" / "processed" / "knowledge_graph" /
    "threatlens_graph.graphml"
)

NODES_PATH = (
    ROOT / "data" / "processed" / "knowledge_graph" /
    "threatlens_graph_nodes.csv"
)

EDGES_PATH = (
    ROOT / "data" / "processed" / "knowledge_graph" /
    "threatlens_graph_edges.csv"
)

SUMMARY_PATH = (
    ROOT / "data" / "processed" / "knowledge_graph" /
    "threatlens_graph_summary.json"
)


def main():

    print("=" * 72)
    print("THREATLENS - STEP 28.2")
    print("THREAT KNOWLEDGE GRAPH VALIDATION")
    print("=" * 72)

    checks = []

    def check(name, passed):
        checks.append((name, bool(passed)))
        print(
            f"{name:<45} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()
    print("-" * 72)
    print("1. FILE AVAILABILITY")
    print("-" * 72)

    check("GraphML file", GRAPH_PATH.exists())
    check("Nodes CSV", NODES_PATH.exists())
    check("Edges CSV", EDGES_PATH.exists())
    check("Summary JSON", SUMMARY_PATH.exists())

    if not all(path.exists() for path in (
        GRAPH_PATH,
        NODES_PATH,
        EDGES_PATH,
        SUMMARY_PATH,
    )):
        print()
        print("Graph validation cannot continue.")
        raise SystemExit(1)

    print()
    print("-" * 72)
    print("2. GRAPH LOAD")
    print("-" * 72)

    graph = nx.read_graphml(GRAPH_PATH)

    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        summary = json.load(f)

    check(
        "GraphML loads successfully",
        graph.number_of_nodes() > 0,
    )

    check(
        "Graph node count matches CSV",
        graph.number_of_nodes() == len(nodes),
    )

    check(
        "Graph edge count matches CSV",
        graph.number_of_edges() == len(edges),
    )

    print()
    print("-" * 72)
    print("3. NODE INTEGRITY")
    print("-" * 72)

    required_node_columns = {
        "node_id",
        "node_type",
        "label",
    }

    check(
        "Required node columns",
        required_node_columns.issubset(
            nodes.columns
        ),
    )

    check(
        "Node IDs unique",
        nodes["node_id"].is_unique,
    )

    check(
        "Node IDs non-empty",
        nodes["node_id"].notna().all()
        and
        (nodes["node_id"].astype(str).str.len() > 0).all(),
    )

    valid_node_types = {
        "CVE",
        "CWE",
        "CPE",
        "Reference",
        "RiskCategory",
        "ThreatSignal",
    }

    actual_node_types = set(
        nodes["node_type"].dropna().astype(str)
    )

    check(
        "Node types valid",
        actual_node_types.issubset(
            valid_node_types
        ),
    )

    print()
    print("-" * 72)
    print("4. CVE INTEGRITY")
    print("-" * 72)

    cves = nodes[
        nodes["node_type"] == "CVE"
    ].copy()

    valid_categories = {
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
        "MINIMAL",
    }

    check(
        "CVE count is 500",
        len(cves) == 500,
    )

    check(
        "Every CVE has risk category",
        cves["risk_category"].notna().all()
        and
        cves["risk_category"].isin(
            valid_categories
        ).all(),
    )

    scores = pd.to_numeric(
        cves["threatlens_score"],
        errors="coerce",
    )

    check(
        "CVE scores numeric",
        scores.notna().all(),
    )

    check(
        "CVE scores within 0..1",
        ((scores >= 0) & (scores <= 1)).all(),
    )

    print()
    print("-" * 72)
    print("5. RISK CATEGORY CONTRACT")
    print("-" * 72)

    def expected_category(score):

        if score >= 0.50:
            return "CRITICAL"

        if score >= 0.20:
            return "HIGH"

        if score >= 0.05:
            return "MEDIUM"

        if score >= 0.01:
            return "LOW"

        return "MINIMAL"

    expected = scores.apply(
        expected_category
    )

    actual = cves[
        "risk_category"
    ].astype(str)

    matches = int(
        (expected.values == actual.values).sum()
    )

    check(
        "Risk categories match production contract",
        matches == len(cves),
    )

    print(
        f"Matches: {matches}/{len(cves)}"
    )

    print()
    print("-" * 72)
    print("6. EDGE INTEGRITY")
    print("-" * 72)

    required_edge_columns = {
        "source",
        "target",
        "relation",
    }

    check(
        "Required edge columns",
        required_edge_columns.issubset(
            edges.columns
        ),
    )

    check(
        "No empty edge sources",
        edges["source"].notna().all(),
    )

    check(
        "No empty edge targets",
        edges["target"].notna().all(),
    )

    graph_node_ids = set(
        nodes["node_id"].astype(str)
    )

    edge_nodes = set(
        edges["source"].astype(str)
    ) | set(
        edges["target"].astype(str)
    )

    check(
        "All edge nodes exist",
        edge_nodes.issubset(
            graph_node_ids
        ),
    )

    valid_relations = {
        "AFFECTS",
        "HAS_CWE",
        "HAS_RISK",
        "KNOWN_EXPLOITED",
        "REFERENCES",
        "MAPPED_TO",
    }

    actual_relations = set(
        edges["relation"].dropna().astype(str)
    )

    check(
        "Edge relations valid",
        actual_relations.issubset(
            valid_relations
        ),
    )

    print()
    print("-" * 72)
    print("7. CVE CONNECTIVITY")
    print("-" * 72)

    cve_ids = set(
        cves["node_id"].astype(str)
    )

    connected_cves = set()

    for source, target in zip(
        edges["source"].astype(str),
        edges["target"].astype(str),
    ):
        if source in cve_ids:
            connected_cves.add(source)

        if target in cve_ids:
            connected_cves.add(target)

    check(
        "No orphan CVE nodes",
        connected_cves == cve_ids,
    )

    print(
        f"Connected CVEs: "
        f"{len(connected_cves)}/{len(cve_ids)}"
    )

    print()
    print("-" * 72)
    print("8. KEV CONSISTENCY")
    print("-" * 72)

    kev_edges = edges[
        edges["relation"] == "KNOWN_EXPLOITED"
    ]

    kev_cves = set(
        kev_edges["source"].astype(str)
    )

    cve_kev_values = cves[
        cves["known_exploited"].astype(str).str.lower()
        == "true"
    ]

    check(
        "KEV edge count matches KEV CVEs",
        len(kev_edges) == len(cve_kev_values),
    )

    check(
        "KEV edges originate from CVEs",
        kev_cves.issubset(cve_ids),
    )

    print(
        f"KEV CVEs: {len(cve_kev_values)}"
    )

    print(
        f"KEV edges: {len(kev_edges)}"
    )

    print()
    print("-" * 72)
    print("9. SUMMARY CONSISTENCY")
    print("-" * 72)

    check(
        "Summary node count",
        summary.get("nodes")
        == graph.number_of_nodes(),
    )

    check(
        "Summary edge count",
        summary.get("edges")
        == graph.number_of_edges(),
    )

    check(
        "Summary CVE count",
        summary.get("selected_cves")
        == len(cves),
    )

    print()
    print("=" * 72)

    passed = sum(
        1 for _, result in checks
        if result
    )

    failed = len(checks) - passed

    print(
        f"TOTAL CHECKS: {len(checks)}"
    )

    print(
        f"PASSED:       {passed}"
    )

    print(
        f"FAILED:       {failed}"
    )

    print(
        f"PASS RATE:    "
        f"{(passed / len(checks) * 100):.2f}%"
    )

    if failed == 0:
        print()
        print("GRAPH STATUS: READY")
    else:
        print()
        print("GRAPH STATUS: REVIEW")

    print("=" * 72)


if __name__ == "__main__":
    main()
