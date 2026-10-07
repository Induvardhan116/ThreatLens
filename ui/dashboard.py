from pathlib import Path
import json
import math
import sys

import pandas as pd
import streamlit as st


# ============================================================
# THREATLENS — PROFESSIONAL SECURITY INTELLIGENCE UI
# ============================================================

st.set_page_config(
    page_title="ThreatLens | Security Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "inference"
    / "threatlens_predictions.csv"
)

PREDICTION_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "inference"
    / "threatlens_prediction_summary.csv"
)

MODEL_REPORT = (
    ROOT
    / "data"
    / "processed"
    / "inference"
    / "threatlens_inference_report.json"
)

FINAL_AUDIT = (
    ROOT
    / "data"
    / "processed"
    / "inference"
    / "threatlens_final_audit.json"
)

GRAPH_NODES = (
    ROOT
    / "data"
    / "processed"
    / "knowledge_graph"
    / "threatlens_graph_nodes.csv"
)

GRAPH_EDGES = (
    ROOT
    / "data"
    / "processed"
    / "knowledge_graph"
    / "threatlens_graph_edges.csv"
)

GRAPH_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "knowledge_graph"
    / "threatlens_graph_summary.json"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

/* Main background */
.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(30, 90, 150, 0.14),
            transparent 30%
        ),
        radial-gradient(
            circle at 10% 20%,
            rgba(25, 80, 130, 0.08),
            transparent 25%
        ),
        #070b10;
    color: #e8eef7;
}

/* Hide Streamlit chrome */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Main content */
.block-container {
    max-width: 1480px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0b1017 0%,
            #080c12 100%
        );
    border-right: 1px solid #1c2733;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}

/* Sidebar text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #93a4b8 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #263443;
    background: #101721;
    color: #c8d5e4;
    min-height: 42px;
    font-weight: 600;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    border-color: #2f8cff;
    background: #132234;
    color: #ffffff;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background:
        linear-gradient(
            135deg,
            #1769d5,
            #2688ff
        );
    border-color: #2b8dff;
    color: white;
}

/* Inputs */
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #0d141d !important;
    border: 1px solid #253342 !important;
    color: #e7eef7 !important;
    border-radius: 10px !important;
}

/* Cards */
.tl-card {
    background:
        linear-gradient(
            145deg,
            rgba(18, 27, 37, 0.96),
            rgba(11, 17, 24, 0.96)
        );
    border: 1px solid #202d3b;
    border-radius: 16px;
    padding: 22px;
    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.20);
}

.tl-card:hover {
    border-color: #2d4154;
}

.tl-label {
    color: #71859b;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.tl-value {
    margin-top: 7px;
    font-size: 30px;
    line-height: 1;
    font-weight: 750;
    color: #f4f8fc;
}

.tl-sub {
    margin-top: 9px;
    color: #71859b;
    font-size: 12px;
}

/* Page heading */
.tl-kicker {
    color: #43a5ff;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.tl-title {
    font-size: 38px;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.035em;
    color: #f4f8fc;
}

.tl-description {
    color: #8293a6;
    max-width: 850px;
    margin-top: 10px;
    font-size: 14px;
    line-height: 1.6;
}

/* Brand */
.tl-brand {
    padding: 4px 4px 22px 4px;
    border-bottom: 1px solid #1b2631;
    margin-bottom: 22px;
}

.tl-brand-row {
    display: flex;
    align-items: center;
    gap: 11px;
}

.tl-brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        linear-gradient(
            135deg,
            #1769d5,
            #36a2ff
        );
    box-shadow:
        0 0 25px rgba(36, 137, 255, 0.20);
    color: white;
    font-size: 19px;
}

.tl-brand-name {
    color: #f3f7fb;
    font-size: 19px;
    font-weight: 800;
}

.tl-brand-caption {
    color: #607286;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* Status pill */
.tl-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.20);
    color: #65dc91;
    font-size: 11px;
    font-weight: 700;
}

.tl-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #39d779;
    box-shadow: 0 0 9px rgba(57, 215, 121, 0.7);
}

/* Hero */
.tl-hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(
            circle at 90% 10%,
            rgba(42, 143, 255, 0.14),
            transparent 32%
        ),
        linear-gradient(
            145deg,
            #101a25,
            #0c121a
        );
    border: 1px solid #263747;
    border-radius: 20px;
    padding: 30px;
}

.tl-hero h2 {
    margin: 0;
    color: #f4f8fc;
    font-size: 25px;
}

.tl-hero p {
    color: #8497ab;
    max-width: 760px;
    line-height: 1.65;
}

/* Risk badges */
.risk-critical {
    color: #ff7373;
    background: rgba(255, 77, 77, 0.10);
    border: 1px solid rgba(255, 77, 77, 0.22);
}

.risk-high {
    color: #ffb454;
    background: rgba(255, 180, 84, 0.10);
    border: 1px solid rgba(255, 180, 84, 0.22);
}

.risk-medium {
    color: #ffd45c;
    background: rgba(255, 212, 92, 0.10);
    border: 1px solid rgba(255, 212, 92, 0.22);
}

.risk-low {
    color: #54c7ff;
    background: rgba(84, 199, 255, 0.10);
    border: 1px solid rgba(84, 199, 255, 0.22);
}

.risk-minimal {
    color: #6fdf9a;
    background: rgba(111, 223, 154, 0.08);
    border: 1px solid rgba(111, 223, 154, 0.18);
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #202d3b;
    border-radius: 12px;
    overflow: hidden;
}

/* Dividers */
hr {
    border-color: #1d2935 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #0e151e !important;
    border: 1px solid #202d3b !important;
    border-radius: 10px !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: transparent;
}

[data-testid="stMetricLabel"] {
    color: #71859b !important;
}

[data-testid="stMetricValue"] {
    color: #f3f7fb !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #080c11;
}

::-webkit-scrollbar-thumb {
    background: #243241;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #34506b;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def load_predictions():
    if not PREDICTIONS.exists():
        return pd.DataFrame()

    df = pd.read_csv(PREDICTIONS)

    if "record_id" not in df.columns:
        df.insert(
            0,
            "record_id",
            range(1, len(df) + 1)
        )

    if "threatlens_score" in df.columns:
        df["threatlens_score"] = pd.to_numeric(
            df["threatlens_score"],
            errors="coerce"
        ).fillna(0)

    return df


@st.cache_data(ttl=30, show_spinner=False)
def load_graph():
    nodes = (
        pd.read_csv(GRAPH_NODES)
        if GRAPH_NODES.exists()
        else pd.DataFrame()
    )

    edges = (
        pd.read_csv(GRAPH_EDGES)
        if GRAPH_EDGES.exists()
        else pd.DataFrame()
    )

    return nodes, edges


@st.cache_data(ttl=30, show_spinner=False)
def load_json(path):
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


predictions = load_predictions()
graph_nodes, graph_edges = load_graph()
model_report = load_json(MODEL_REPORT)
final_audit = load_json(FINAL_AUDIT)
graph_summary = load_json(GRAPH_SUMMARY)


# ============================================================
# HELPERS
# ============================================================

def number(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def score(value):
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "0.000"


def risk_class(value):
    value = str(value).upper()

    return {
        "CRITICAL": "risk-critical",
        "HIGH": "risk-high",
        "MEDIUM": "risk-medium",
        "LOW": "risk-low",
        "MINIMAL": "risk-minimal",
    }.get(value, "risk-minimal")


def risk_badge(value):
    cls = risk_class(value)

    return f"""
    <span style="
        display:inline-flex;
        align-items:center;
        padding:5px 10px;
        border-radius:999px;
        font-size:11px;
        font-weight:800;
        letter-spacing:.06em;
        {(
            "color:#ff7373;background:rgba(255,77,77,.10);"
            "border:1px solid rgba(255,77,77,.22);"
            if cls == "risk-critical"
            else
            "color:#ffb454;background:rgba(255,180,84,.10);"
            "border:1px solid rgba(255,180,84,.22);"
            if cls == "risk-high"
            else
            "color:#ffd45c;background:rgba(255,212,92,.10);"
            "border:1px solid rgba(255,212,92,.22);"
            if cls == "risk-medium"
            else
            "color:#54c7ff;background:rgba(84,199,255,.10);"
            "border:1px solid rgba(84,199,255,.22);"
            if cls == "risk-low"
            else
            "color:#6fdf9a;background:rgba(111,223,154,.08);"
            "border:1px solid rgba(111,223,154,.18);"
        )}
    ">
        {str(value).upper()}
    </span>
    """


def card(label, value, subtitle=""):
    st.markdown(
        f"""
        <div class="tl-card">
            <div class="tl-label">{label}</div>
            <div class="tl-value">{value}</div>
            <div class="tl-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(kicker, title, description):
    st.markdown(
        f"""
        <div class="tl-kicker">{kicker}</div>
        <div class="tl-title">{title}</div>
        <div class="tl-description">{description}</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="tl-brand">
            <div class="tl-brand-row">
                <div class="tl-brand-icon">🛡</div>
                <div>
                    <div class="tl-brand-name">ThreatLens</div>
                    <div class="tl-brand-caption">
                        Security Intelligence
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            color:#53677b;
            font-size:10px;
            font-weight:800;
            letter-spacing:.14em;
            margin:0 0 8px 4px;
        ">
            WORKSPACE
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Command Center",
            "Vulnerability Intel",
            "Attack Graph",
            "Evidence",
            "Investigation",
            "System",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            color:#53677b;
            font-size:10px;
            font-weight:800;
            letter-spacing:.14em;
            margin:0 0 8px 4px;
        ">
            QUICK ANALYZE
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_cve = st.text_input(
        "CVE",
        value="CVE-2025-9242",
        label_visibility="collapsed",
        placeholder="Enter CVE ID",
    ).strip().upper()

    if st.button(
        "Analyze vulnerability",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["selected_cve"] = quick_cve
        st.session_state["page"] = "Investigation"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tl-status">
            <span class="tl-dot"></span>
            Intelligence engine online
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SELECTED CVE
# ============================================================

selected_cve = st.session_state.get(
    "selected_cve",
    quick_cve
)

if selected_cve:
    selected_cve = selected_cve.upper().strip()


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    page_header(
        "COMMAND CENTER",
        "Security intelligence, at a glance.",
        "ThreatLens combines vulnerability severity, exploitation intelligence "
        "and contextual machine learning into one analyst workspace.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tl-hero">
            <h2>Know what deserves attention first.</h2>
            <p>
                A production vulnerability corpus with exploitation-aware
                prioritization and deterministic evidence workflows.
                Built for analysts who need signal before noise.
            </p>
            <div class="tl-status">
                <span class="tl-dot"></span>
                MODEL READY · ThreatLens Conservative v1.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    total = len(predictions)

    critical = int(
        (predictions["risk_category"] == "CRITICAL").sum()
    ) if not predictions.empty else 0

    high = int(
        (predictions["risk_category"] == "HIGH").sum()
    ) if not predictions.empty else 0

    kev = int(
        predictions["exploitation_label"].sum()
    ) if (
        not predictions.empty
        and "exploitation_label" in predictions.columns
    ) else 0

    mean_score = (
        predictions["threatlens_score"].mean()
        if not predictions.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card(
            "Vulnerabilities",
            number(total),
            "Production prediction corpus",
        )

    with c2:
        card(
            "Critical",
            number(critical),
            "Highest ThreatLens category",
        )

    with c3:
        card(
            "Known Exploited",
            number(kev),
            "CISA KEV-linked records",
        )

    with c4:
        card(
            "Mean Model Score",
            score(mean_score),
            "Across production predictions",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])

    with left:

        st.markdown(
            '<div class="tl-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Risk distribution"
        )

        if not predictions.empty:

            distribution = (
                predictions["risk_category"]
                .value_counts()
                .reindex(
                    [
                        "CRITICAL",
                        "HIGH",
                        "MEDIUM",
                        "LOW",
                        "MINIMAL",
                    ],
                    fill_value=0,
                )
            )

            chart = pd.DataFrame(
                {
                    "Risk": distribution.index,
                    "Vulnerabilities": distribution.values,
                }
            ).set_index("Risk")

            st.bar_chart(
                chart,
                height=310,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            '<div class="tl-card">',
            unsafe_allow_html=True,
        )

        st.markdown("### Highest-priority vulnerabilities")

        if not predictions.empty:

            top = (
                predictions
                .sort_values(
                    "threatlens_score",
                    ascending=False,
                )
                .head(8)
                [
                    [
                        "cve_id",
                        "threatlens_score",
                        "risk_category",
                    ]
                ]
                .copy()
            )

            top["threatlens_score"] = top[
                "threatlens_score"
            ].round(3)

            top.columns = [
                "CVE",
                "Score",
                "Risk",
            ]

            st.dataframe(
                top,
                hide_index=True,
                use_container_width=True,
                height=310,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tl-card">
            <div class="tl-label">Research Position</div>
            <div style="
                font-size:20px;
                font-weight:750;
                margin-top:8px;
                color:#f1f6fb;
            ">
                Context-aware prioritization beyond severity-only ranking.
            </div>
            <div style="
                color:#8194a8;
                line-height:1.6;
                margin-top:9px;
                max-width:900px;
            ">
                ThreatLens evaluates vulnerability context using a
                chronological ML evaluation framework and compares
                prioritization against CVSS-only and contextual-rule baselines.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# VULNERABILITY INTELLIGENCE
# ============================================================

elif page == "Vulnerability Intel":

    page_header(
        "VULNERABILITY INTELLIGENCE",
        "Find the vulnerabilities that matter.",
        "Search the production ThreatLens ranking and inspect exploitation-aware risk.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    q = st.text_input(
        "Search CVE",
        placeholder="CVE-2025-9242",
    ).strip().upper()

    f1, f2 = st.columns(2)

    with f1:
        category = st.selectbox(
            "Risk category",
            [
                "ALL",
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
                "MINIMAL",
            ],
        )

    with f2:
        kev_filter = st.selectbox(
            "Exploitation",
            [
                "ALL",
                "KNOWN EXPLOITED",
            ],
        )

    df = predictions.copy()

    if q:
        df = df[
            df["cve_id"]
            .astype(str)
            .str.contains(
                q,
                case=False,
                na=False,
            )
        ]

    if category != "ALL":
        df = df[
            df["risk_category"] == category
        ]

    if (
        kev_filter == "KNOWN EXPLOITED"
        and "exploitation_label" in df.columns
    ):
        df = df[
            df["exploitation_label"] == 1
        ]

    df = (
        df.sort_values(
            "threatlens_score",
            ascending=False,
        )
        .head(100)
        .copy()
    )

    st.markdown("<br>", unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        card(
            "Results",
            number(len(df)),
            "Matching vulnerabilities",
        )

    with b:
        card(
            "Highest Score",
            score(
                df["threatlens_score"].max()
                if not df.empty
                else 0
            ),
            "Top result",
        )

    with c:
        card(
            "Critical Results",
            number(
                (
                    df["risk_category"] == "CRITICAL"
                ).sum()
                if not df.empty
                else 0
            ),
            "Require immediate review",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if not df.empty:

        display = df[
            [
                "cve_id",
                "threatlens_score",
                "risk_category",
            ]
            + (
                ["exploitation_label"]
                if "exploitation_label" in df.columns
                else []
            )
        ].copy()

        display["threatlens_score"] = display[
            "threatlens_score"
        ].round(3)

        display.columns = [
            "CVE",
            "ThreatLens Score",
            "Risk",
        ] + (
            ["Known Exploited"]
            if "exploitation_label" in df.columns
            else []
        )

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            height=520,
        )

        st.info(
            "Select a CVE from Quick Analyze in the sidebar "
            "to open its investigation workspace."
        )

    else:
        st.warning("No vulnerabilities matched your filters.")


# ============================================================
# ATTACK GRAPH
# ============================================================

elif page == "Attack Graph":

    page_header(
        "THREAT KNOWLEDGE GRAPH",
        "See how vulnerability intelligence connects.",
        "Explore CVE, CWE, CPE, reference, exploitation and risk relationships.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    n_nodes = len(graph_nodes)
    n_edges = len(graph_edges)

    g1, g2, g3 = st.columns(3)

    with g1:
        card(
            "Graph Nodes",
            number(n_nodes),
            "Knowledge entities",
        )

    with g2:
        card(
            "Graph Edges",
            number(n_edges),
            "Relationships",
        )

    with g3:
        card(
            "CVE Entities",
            number(
                (
                    graph_nodes["node_type"] == "CVE"
                ).sum()
                if (
                    not graph_nodes.empty
                    and "node_type" in graph_nodes.columns
                )
                else 0
            ),
            "Vulnerability nodes",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if not graph_nodes.empty:

        st.markdown(
            """
            <div class="tl-card">
                <div class="tl-label">Graph Overview</div>
                <div style="
                    font-size:21px;
                    font-weight:750;
                    margin-top:8px;
                ">
                    Knowledge graph topology
                </div>
                <div style="
                    color:#8194a8;
                    margin-top:7px;
                ">
                    The production graph contains the entities and
                    relationships used by the ThreatLens intelligence layer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        node_counts = (
            graph_nodes["node_type"]
            .value_counts()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            node_counts,
            height=360,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander(
            "Inspect graph entities"
        ):

            st.dataframe(
                graph_nodes.head(250),
                hide_index=True,
                use_container_width=True,
                height=420,
            )

        with st.expander(
            "Inspect graph relationships"
        ):

            st.dataframe(
                graph_edges.head(250),
                hide_index=True,
                use_container_width=True,
                height=420,
            )

    else:
        st.warning(
            "Knowledge graph artifacts were not found."
        )


# ============================================================
# EVIDENCE
# ============================================================

elif page == "Evidence":

    page_header(
        "EVIDENCE ENGINE",
        "Trace every conclusion to evidence.",
        "ThreatLens separates known facts, inferred context and unknown information.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    cve = st.text_input(
        "CVE identifier",
        value=selected_cve,
    ).strip().upper()

    if st.button(
        "Load evidence",
        type="primary",
    ):

        try:

            sys.path.insert(
                0,
                str(ROOT),
            )

            from evidence.evidence_engine import EvidenceEngine

            engine = EvidenceEngine()

            result = engine.search(cve)

            if result.get("status") == "UNKNOWN":

                st.warning(
                    "No evidence was found for this CVE."
                )

            else:

                summary = result.get(
                    "summary",
                    {},
                )

                a, b, c = st.columns(3)

                with a:
                    card(
                        "Known",
                        number(
                            summary.get(
                                "known",
                                0,
                            )
                        ),
                        "Verified evidence",
                    )

                with b:
                    card(
                        "Inferred",
                        number(
                            summary.get(
                                "inferred",
                                0,
                            )
                        ),
                        "Derived context",
                    )

                with c:
                    card(
                        "Unknown",
                        number(
                            summary.get(
                                "unknown",
                                0,
                            )
                        ),
                        "Unavailable evidence",
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.json(
                    result,
                    expanded=False,
                )

        except Exception as exc:

            st.error(
                f"Evidence Engine error: {exc}"
            )


# ============================================================
# INVESTIGATION
# ============================================================

elif page == "Investigation":

    page_header(
        "AI INVESTIGATOR",
        "Investigate a vulnerability.",
        "Deterministic evidence drives the investigation; the local AI layer explains the evidence without inventing facts.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    cve = st.text_input(
        "CVE identifier",
        value=selected_cve,
    ).strip().upper()

    if st.button(
        "Investigate CVE",
        type="primary",
    ):

        try:

            sys.path.insert(
                0,
                str(ROOT),
            )

            from ai.investigator import AIInvestigator

            investigator = AIInvestigator()

            result = investigator.investigate(cve)

            if result.get("status") == "UNKNOWN":

                st.warning(
                    "The investigator does not have sufficient evidence for this CVE."
                )

            else:

                finding = result.get(
                    "finding",
                    "No finding available.",
                )

                confidence = result.get(
                    "confidence",
                    "UNKNOWN",
                )

                risk_score = result.get(
                    "risk_score",
                    0,
                )

                risk = result.get(
                    "risk_category",
                    "UNKNOWN",
                )

                a, b, c = st.columns(3)

                with a:
                    card(
                        "Status",
                        result.get(
                            "status",
                            "UNKNOWN",
                        ),
                        "Investigation state",
                    )

                with b:
                    card(
                        "Risk",
                        risk,
                        "ThreatLens category",
                    )

                with c:
                    card(
                        "Risk Score",
                        score(risk_score),
                        confidence,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div class="tl-hero">
                        <div class="tl-kicker">
                            INVESTIGATION FINDING
                        </div>
                        <h2>
                            {finding}
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                with st.expander(
                    "View deterministic investigation data"
                ):
                    st.json(
                        result,
                        expanded=False,
                    )

        except Exception as exc:

            st.error(
                f"Investigator error: {exc}"
            )


# ============================================================
# SYSTEM
# ============================================================

elif page == "System":

    page_header(
        "SYSTEM",
        "ThreatLens platform health.",
        "Production artifacts, model contract and knowledge graph status.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    model_name = (
        model_report
        .get(
            "model",
            {}
        )
        .get(
            "name",
            "ThreatLens Conservative",
        )
        if isinstance(model_report, dict)
        else "ThreatLens Conservative"
    )

    model_version = (
        model_report
        .get(
            "model",
            {}
        )
        .get(
            "version",
            "1.0",
        )
        if isinstance(model_report, dict)
        else "1.0"
    )

    audit_status = (
        "PASS"
        if final_audit
        else "AVAILABLE"
    )

    a, b, c, d = st.columns(4)

    with a:
        card(
            "Model",
            "READY",
            model_name,
        )

    with b:
        card(
            "Version",
            str(model_version),
            "Production model",
        )

    with c:
        card(
            "Predictions",
            number(len(predictions)),
            "Production records",
        )

    with d:
        card(
            "Audit",
            audit_status,
            "Consolidated validation",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class="tl-card">
                <div class="tl-label">
                    Production Artifacts
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        artifacts = {
            "Model artifact": ROOT
            / "data"
            / "processed"
            / "model"
            / "threatlens_conservative_model.joblib",

            "Prediction dataset": PREDICTIONS,

            "Inference report": MODEL_REPORT,

            "Final audit": FINAL_AUDIT,

            "Graph nodes": GRAPH_NODES,

            "Graph edges": GRAPH_EDGES,
        }

        rows = []

        for name, path in artifacts.items():

            rows.append(
                {
                    "Artifact": name,
                    "Status": (
                        "READY"
                        if path.exists()
                        else "MISSING"
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

    with right:

        st.markdown(
            """
            <div class="tl-card">
                <div class="tl-label">
                    Risk Contract
                </div>
                <div style="
                    font-size:18px;
                    font-weight:750;
                    margin-top:10px;
                ">
                    Authoritative ThreatLens thresholds
                </div>
                <div style="
                    color:#8799ac;
                    line-height:2;
                    margin-top:12px;
                ">
                    CRITICAL &nbsp; ≥ 0.50<br>
                    HIGH &nbsp;&nbsp;&nbsp;&nbsp; ≥ 0.20<br>
                    MEDIUM &nbsp;&nbsp; ≥ 0.05<br>
                    LOW &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ≥ 0.01<br>
                    MINIMAL &nbsp;&nbsp; &lt; 0.01
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="tl-card">
                <div class="tl-label">
                    Architecture
                </div>
                <div style="
                    color:#8799ac;
                    line-height:1.8;
                    margin-top:10px;
                ">
                    NVD → CISA KEV → Feature Engineering
                    → Conservative ML → Risk Ranking
                    → Knowledge Graph → Evidence Engine
                    → AI Investigator
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:45px;
        padding-top:18px;
        border-top:1px solid #18232e;
        display:flex;
        justify-content:space-between;
        color:#4f6275;
        font-size:11px;
    ">
        <span>THREATLENS · SECURITY INTELLIGENCE PLATFORM</span>
        <span>AI-POWERED · EVIDENCE-GROUNDED · ML RISK PRIORITIZATION</span>
    </div>
    """,
    unsafe_allow_html=True,
)
