import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import LogAnalysis from "./LogAnalysis.jsx";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  ChevronRight,
  Database,
  ExternalLink,
  FileSearch,
  Gauge,
  GitBranch,
  LayoutDashboard,
  Network,
  Search,
  Shield,
  ShieldAlert,
  Target,
  Terminal,
  Zap,
} from "lucide-react";
import "./App.css";
import "./Enterprise.css";

const RiskDistributionChart = lazy(() => import("./RiskDistributionChart.jsx"));

const API = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

const pagePaths = {
  dashboard: "/",
  vulnerabilities: "/vulnerabilities",
  graph: "/graph",
  research: "/research",
  investigator: "/investigator",
  evidence: "/evidence",
};

const pageTitles = {
  dashboard: "Threat Intelligence Overview",
  vulnerabilities: "Vulnerability Intelligence",
  graph: "Threat Knowledge Graph",
  research: "Research & Evaluation",
  investigator: "AI Investigator",
  evidence: "Evidence Engine",
};

const pathPages = Object.fromEntries(
  Object.entries(pagePaths).map(([page, path]) => [path, page])
);

const categoryMeta = {
  CRITICAL: { label: "Critical", className: "critical" },
  HIGH: { label: "High", className: "high" },
  MEDIUM: { label: "Medium", className: "medium" },
  LOW: { label: "Low", className: "low" },
  MINIMAL: { label: "Minimal", className: "minimal" },
};

function formatNumber(value) {
  if (value == null || value === "" || !Number.isFinite(Number(value))) {
    return "—";
  }
  return new Intl.NumberFormat("en-US").format(Number(value));
}

function formatScore(value) {
  if (value == null || value === "" || !Number.isFinite(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(3);
}

function apiErrorMessage(error, fallback) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (detail && typeof detail === "object") {
    return detail.error || JSON.stringify(detail);
  }
  return fallback;
}

function apiStatusMessage(system) {
  if (system?.status !== "ready") {
    return system?.status || "Check the API connection";
  }

  const unavailable = [
    !system.prediction_dataset && "prediction dataset",
    !system.knowledge_graph && "knowledge graph",
  ].filter(Boolean);

  return unavailable.length
    ? `Unavailable: ${unavailable.join(", ")}`
    : "Core services responding";
}

function vulnerabilityQuery(search, category, kevOnly) {
  const params = new URLSearchParams();

  if (search.trim()) params.set("search", search.trim());
  if (category !== "ALL") params.set("category", category);
  if (kevOnly) params.set("kev", "true");

  params.set("limit", "100");
  return params.toString();
}

function normalizeSummary(data) {
  const totalRecords =
    data.total_records ?? data.total_vulnerabilities ?? null;
  const knownExploited =
    data.known_exploited ?? data.kev_vulnerabilities ?? null;

  return {
    ...data,
    total_records: totalRecords,
    known_exploited: knownExploited,
    known_exploited_rate:
      data.known_exploited_rate ??
      (totalRecords != null && knownExploited != null && totalRecords
        ? Number(((knownExploited / totalRecords) * 100).toFixed(1))
        : null),
    model_features: data.model_features ?? data.feature_count ?? null,
  };
}

function normalizeVulnerability(item) {
  const publishedDate =
    item.published_date ??
    (item.publication_year
      ? `${item.publication_year}-${String(item.publication_month || 1).padStart(2, "0")}`
      : null);

  return {
    ...item,
    published_date: publishedDate,
    known_exploited:
      item.known_exploited ?? Number(item.exploitation_label) === 1,
  };
}

function CategoryBadge({ category }) {
  const meta = categoryMeta[category] || categoryMeta.MINIMAL;

  return (
    <span className={`category-badge ${meta.className}`}>
      {meta.label}
    </span>
  );
}

function StatCard({ icon: Icon, label, value, detail, accent }) {
  return (
    <div className={`stat-card ${accent || ""}`}>
      <div className="stat-icon">
        <Icon size={19} />
      </div>

      <div className="stat-content">
        <span>{label}</span>
        <strong>{value}</strong>
        {detail && <small>{detail}</small>}
      </div>
    </div>
  );
}

function App() {
  const location = useLocation();
  const routerNavigate = useNavigate();
  const activePage = pathPages[location.pathname] || "dashboard";
  const [summary, setSummary] = useState(null);
  const [vulnerabilities, setVulnerabilities] = useState([]);
  const [graphSummary, setGraphSummary] = useState(null);
  const [system, setSystem] = useState(null);

  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("ALL");
  const [kevOnly, setKevOnly] = useState(false);
  const [selected, setSelected] = useState(null);
  const [focusCve, setFocusCve] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingVulnerabilities, setLoadingVulnerabilities] = useState(false);
  const [error, setError] = useState("");
  const initialVulnerabilityLoad = useRef(true);

  const loadDashboard = useCallback(async (filters = {}) => {
    const filterSearch = filters.search || "";
    const filterCategory = filters.category || "ALL";
    const filterKevOnly = filters.kevOnly || false;

    try {
      setRefreshing(true);
      setError("");

      const results = await Promise.allSettled([
        axios.get(`${API}/dashboard/summary`),
        axios.get(
          `${API}/vulnerabilities?${vulnerabilityQuery(
            filterSearch,
            filterCategory,
            filterKevOnly
          )}`
        ),
        axios.get(`${API}/graph/summary`),
        axios.get(`${API}/system/status`),
      ]);

      const failures = [];
      const [summaryResult, vulnerabilitiesResult, graphResult, systemResult] =
        results;

      if (summaryResult.status === "fulfilled") {
        setSummary(normalizeSummary(summaryResult.value.data));
      } else {
        failures.push(
          apiErrorMessage(
            summaryResult.reason,
            "Dashboard summary is unavailable."
          )
        );
      }

      if (vulnerabilitiesResult.status === "fulfilled") {
        setVulnerabilities(
          (
            vulnerabilitiesResult.value.data.items ||
            vulnerabilitiesResult.value.data.results ||
            []
          ).map(normalizeVulnerability)
        );
      } else {
        failures.push(
          apiErrorMessage(
            vulnerabilitiesResult.reason,
            "Vulnerability data is unavailable."
          )
        );
      }

      if (graphResult.status === "fulfilled") {
        setGraphSummary({
          ...graphResult.value.data,
          cves:
            graphResult.value.data.cves ??
            graphResult.value.data.selected_cves,
          kev_edges:
            graphResult.value.data.kev_edges ??
            graphResult.value.data.relationship_distribution?.KNOWN_EXPLOITED,
        });
      } else {
        failures.push(
          apiErrorMessage(graphResult.reason, "Knowledge graph summary is unavailable.")
        );
      }

      if (systemResult.status === "fulfilled") {
        setSystem(systemResult.value.data);
      } else {
        setSystem(null);
        failures.push(
          apiErrorMessage(systemResult.reason, "System status is unavailable.")
        );
      }

      setError(failures.join(" "));
    } catch (err) {
      setError(apiErrorMessage(err, "Unable to connect to the ThreatLens Application API."));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  const loadVulnerabilities = useCallback(async (signal) => {
    try {
      setLoadingVulnerabilities(true);
      const response = await axios.get(
        `${API}/vulnerabilities?${vulnerabilityQuery(
          search,
          category,
          kevOnly
        )}`,
        { signal }
      );

      setVulnerabilities(
        (response.data.items || response.data.results || []).map(
          normalizeVulnerability
        )
      );
    } catch (err) {
      if (!axios.isCancel(err)) {
        setError(apiErrorMessage(err, "Failed to load vulnerability data."));
      }
    } finally {
      if (!signal?.aborted) setLoadingVulnerabilities(false);
    }
  }, [category, kevOnly, search]);

  const openVulnerability = useCallback(async (cveId) => {
    try {
      const response = await axios.get(
        `${API}/vulnerabilities/${encodeURIComponent(cveId)}`
      );

      setFocusCve(cveId);
      setSelected(normalizeVulnerability(response.data));
    } catch (err) {
      setError(apiErrorMessage(err, "Unable to load vulnerability details."));
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadDashboard();
    }, 0);

    return () => window.clearTimeout(timer);
  }, [loadDashboard]);

  useEffect(() => {
    if (loading) return undefined;
    if (initialVulnerabilityLoad.current) {
      initialVulnerabilityLoad.current = false;
      if (!search.trim() && category === "ALL" && !kevOnly) {
        return undefined;
      }
    }

    const controller = new AbortController();
    const timer = setTimeout(() => {
      void loadVulnerabilities(controller.signal);
    }, 250);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [search, category, kevOnly, loading, loadVulnerabilities]);

  const categoryData = useMemo(() => {
    if (!summary?.risk_distribution) return [];

    return Object.entries(summary.risk_distribution).map(([name, value]) => ({
      name,
      value,
    }));
  }, [summary]);

  const chartData = useMemo(
    () =>
      categoryData.map((item) => ({
        ...item,
        shortName:
          item.name === "CRITICAL"
            ? "Critical"
            : item.name === "MINIMAL"
              ? "Minimal"
              : item.name.charAt(0) + item.name.slice(1).toLowerCase(),
      })),
    [categoryData]
  );

  function navigate(page) {
    setSelected(null);
    routerNavigate(pagePaths[page] || pagePaths.dashboard);
  }

  useEffect(() => {
    if (!pathPages[location.pathname]) {
      routerNavigate(pagePaths.dashboard, { replace: true });
    }
  }, [location.pathname, routerNavigate]);

  useEffect(() => {
    document.title = `${pageTitles[activePage]} | ThreatLens`;
  }, [activePage]);

  const apiAvailable = system?.status === "ready";

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-logo">
          <Shield size={32} />
        </div>
        <h1>ThreatLens</h1>
        <p>Loading threat intelligence platform...</p>
        <div className="loading-bar">
          <span />
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button
          className="brand brand-home"
          onClick={() => navigate("dashboard")}
          aria-label="ThreatLens dashboard"
        >
          <div className="brand-mark">
            <Shield size={23} />
          </div>

          <div>
            <h1>ThreatLens</h1>
            <span>Threat Intelligence Platform</span>
          </div>
        </button>

        <nav aria-label="Primary navigation">
        <div className="sidebar-section">
          <span className="sidebar-label">PLATFORM</span>

          <button
            className={`nav-item ${
              activePage === "dashboard" ? "active" : ""
            }`}
            aria-current={activePage === "dashboard" ? "page" : undefined}
            title="Dashboard"
            onClick={() => navigate("dashboard")}
          >
            <LayoutDashboard size={18} />
            Dashboard
          </button>

          <button
            className={`nav-item ${
              activePage === "vulnerabilities" ? "active" : ""
            }`}
            aria-current={
              activePage === "vulnerabilities" ? "page" : undefined
            }
            title="Vulnerabilities"
            onClick={() => navigate("vulnerabilities")}
          >
            <ShieldAlert size={18} />
            Vulnerabilities
          </button>

          <button
            className={`nav-item ${activePage === "graph" ? "active" : ""}`}
            aria-current={activePage === "graph" ? "page" : undefined}
            title="Knowledge Graph"
            onClick={() => navigate("graph")}
          >
            <Network size={18} />
            Knowledge Graph
          </button>

          <button
            className={`nav-item ${activePage === "research" ? "active" : ""}`}
            aria-current={activePage === "research" ? "page" : undefined}
            title="Research"
            onClick={() => navigate("research")}
          >
            <BarChart3 size={18} />
            Research
          </button>
        </div>

        <div className="sidebar-section">
          <span className="sidebar-label">INTELLIGENCE</span>

          <button
            className={`nav-item ${
              activePage === "investigator" ? "active" : ""
            }`}
            aria-current={
              activePage === "investigator" ? "page" : undefined
            }
            title="AI Investigator"
            onClick={() => navigate("investigator")}
          >
            <BrainCircuit size={18} />
            AI Investigator
          </button>

          <button
            className={`nav-item ${
              activePage === "evidence" ? "active" : ""
            }`}
            aria-current={activePage === "evidence" ? "page" : undefined}
            title="Evidence Engine"
            onClick={() => navigate("evidence")}
          >
            <FileSearch size={18} />
            Evidence Engine
          </button>
        </div>
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span
              className={`status-dot ${apiAvailable ? "is-online" : "is-offline"}`}
            />
            <div>
              <strong>API {apiAvailable ? "Connected" : "Unavailable"}</strong>
              <small>
                {system
                  ? `${system.model || "ThreatLens"} v${system.model_version || "—"}`
                  : "Check the API connection"}
              </small>
            </div>
          </div>

          <div className="sidebar-footer">
            <span>AI + Cybersecurity</span>
            <span>2026</span>
          </div>
        </div>
      </aside>

      <main className="main-content" aria-busy={refreshing}>
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              Security Intelligence <ChevronRight size={13} />{" "}
              {pageTitles[activePage]}
            </div>

            <h2>{pageTitles[activePage]}</h2>
          </div>

          <div className="topbar-actions">
            <div
              className={`live-indicator ${apiAvailable ? "" : "disconnected"}`}
              role="status"
              aria-live="polite"
            >
              <span />
              {apiAvailable ? "Connected" : "Unavailable"}
            </div>

            <button
              className={`icon-button ${refreshing ? "is-refreshing" : ""}`}
              onClick={() => loadDashboard({ search, category, kevOnly })}
              disabled={refreshing}
              aria-label={refreshing ? "Refreshing dashboard" : "Refresh dashboard"}
              title={refreshing ? "Refreshing dashboard" : "Refresh dashboard"}
            >
              <Activity size={18} />
            </button>
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert" aria-live="assertive">
            <AlertTriangle size={17} />
            <span>{error}</span>
            <button onClick={() => setError("")}>Dismiss</button>
          </div>
        )}

        {activePage === "dashboard" && (
          <Dashboard
            summary={summary}
            graphSummary={graphSummary}
            system={system}
            categoryData={chartData}
            vulnerabilities={vulnerabilities}
            refreshing={refreshing}
            onOpen={openVulnerability}
            onNavigate={navigate}
          />
        )}

        {activePage === "vulnerabilities" && (
          <VulnerabilityPage
            vulnerabilities={vulnerabilities}
            search={search}
            setSearch={setSearch}
            category={category}
            setCategory={setCategory}
            kevOnly={kevOnly}
            setKevOnly={setKevOnly}
            loading={loadingVulnerabilities}
            onOpen={openVulnerability}
          />
        )}

        {activePage === "graph" && (
          <GraphPage
            graphSummary={graphSummary}
            vulnerabilities={vulnerabilities}
            initialCveId={focusCve}
          />
        )}

        {activePage === "research" && <ResearchPage summary={summary} />}

        {activePage === "investigator" && (
          <IntelligencePage
            title="AI Investigator"
            icon={BrainCircuit}
            endpoint="investigation"
            exampleCveId={focusCve || vulnerabilities[0]?.cve_id}
            description="Evidence-grounded investigation interface built on the ThreatLens intelligence pipeline."
            items={[
              "Deterministic evidence retrieval",
              "Structured vulnerability context",
              "Known / inferred / unknown separation",
              "LLM explanation with output validation",
              "No score override by the language model",
            ]}
          />
        )}

        {activePage === "evidence" && (
          <IntelligencePage
            title="Evidence Engine"
            icon={FileSearch}
            endpoint="evidence"
            exampleCveId={focusCve || vulnerabilities[0]?.cve_id}
            description="Structured evidence layer connecting vulnerability records, CISA KEV, CWE, references and graph relationships."
            items={[
              "CISA KEV evidence",
              "CWE evidence",
              "Affected product evidence",
              "Reference evidence",
              "Known-fact grounding for investigation",
            ]}
          />
        )}

        <SiteFooter />
      </main>

      {selected && (
        <DetailDrawer
          key={selected.cve_id}
          vulnerability={selected}
          onExplore={(page, cveId) => {
            setFocusCve(cveId);
            navigate(page);
          }}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

function SiteFooter() {
  const currentYear = new Date().getFullYear();

  const sections = [
    {
      title: "Explore",
      links: [
        { label: "Dashboard", to: pagePaths.dashboard },
        { label: "Vulnerabilities", to: pagePaths.vulnerabilities },
        { label: "Knowledge Graph", to: pagePaths.graph },
        { label: "Research", to: pagePaths.research },
      ],
    },
    {
      title: "Intelligence",
      links: [
        { label: "AI Investigator", to: pagePaths.investigator },
        { label: "Evidence Engine", to: pagePaths.evidence },
      ],
    },
    {
      title: "Data & Standards",
      links: [
        {
          label: "CISA KEV Catalog",
          href: "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
        },
        {
          label: "NVD CVE Search",
          href: "https://nvd.nist.gov/vuln/search",
        },
        {
          label: "CWE by MITRE",
          href: "https://cwe.mitre.org/",
        },
      ],
    },
    {
      title: "API Status",
      links: [
        { label: "System status", href: `${API}/system/status` },
        { label: "API health check", href: `${API}/health` },
      ],
    },
  ];

  return (
    <footer className="site-footer" aria-label="ThreatLens site footer">
      <div className="footer-main">
        <div className="footer-brand">
          <div className="footer-brand-lockup">
            <span className="footer-brand-mark">
              <Shield size={17} />
            </span>
            <strong>ThreatLens</strong>
          </div>
          <p>
            Vulnerability prioritization and evidence, built for clearer
            security decisions.
          </p>
        </div>

        {sections.map((section) => (
          <nav
            className="footer-column"
            aria-label={`${section.title} footer links`}
            key={section.title}
          >
            <h3>{section.title}</h3>
            {section.links.map((link) =>
              link.to ? (
                <Link to={link.to} key={link.label}>
                  {link.label}
                </Link>
              ) : (
                <a
                  href={link.href}
                  key={link.label}
                  target="_blank"
                  rel="noreferrer"
                >
                  {link.label}
                  <ExternalLink size={12} aria-hidden="true" />
                </a>
              )
            )}
          </nav>
        ))}
      </div>

      <div className="footer-bottom">
        <span>© {currentYear} ThreatLens</span>
        <span>Security intelligence for vulnerability prioritization</span>
      </div>
    </footer>
  );
}

function Dashboard({
  summary,
  graphSummary,
  system,
  categoryData,
  vulnerabilities,
  refreshing,
  onOpen,
  onNavigate,
}) {
  const topVulnerabilities = vulnerabilities.slice(0, 8);

  return (
    <div className="page">
      <section className="hero-panel">
        <div>
          <div className="eyebrow">
            <Zap size={14} />
            CONTEXT-AWARE VULNERABILITY PRIORITIZATION
          </div>

          <h3>
            Find the vulnerabilities
            <br />
            that demand attention.
          </h3>

          <p>
            ThreatLens combines CVSS, contextual security metadata,
            exploitation intelligence and machine learning to prioritize
            vulnerabilities using evidence beyond severity alone.
          </p>
          <div className="hero-actions">
            <button
              className="primary-button"
              onClick={() => onNavigate("vulnerabilities")}
            >
              Review vulnerabilities <ArrowUpRight size={16} />
            </button>
            <span className="hero-update">
              {refreshing
                ? "Updating intelligence…"
                : "Prioritization at a glance"}
            </span>
          </div>
        </div>

        <div className="hero-metric">
          <span>AVERAGE THREATLENS SCORE</span>
          <strong>
            {summary?.average_threatlens_score == null
              ? "—"
              : formatScore(summary.average_threatlens_score)}
          </strong>
          <small>{summary?.model || "ThreatLens Conservative"}</small>
        </div>
      </section>

      <section className="stats-grid">
        <StatCard
          icon={Database}
          label="Vulnerabilities"
          value={formatNumber(summary?.total_records)}
          detail="Production inference dataset"
        />

        <StatCard
          icon={Target}
          label="Known Exploited"
          value={formatNumber(summary?.known_exploited)}
          detail={
            summary?.known_exploited_rate == null
              ? "Rate unavailable"
              : `${summary.known_exploited_rate}% of dataset`
          }
          accent="danger"
        />

        <StatCard
          icon={Gauge}
          label="Critical"
          value={formatNumber(summary?.risk_distribution?.CRITICAL)}
          detail="Highest ThreatLens category"
          accent="critical-card"
        />

        <StatCard
          icon={BrainCircuit}
          label="Model"
          value={summary?.model?.replace("ThreatLens ", "") || "—"}
          detail={
            summary?.model_features
              ? `${summary.model_features} production features`
              : `Model version ${summary?.model_version || "unavailable"}`
          }
        />
      </section>

      <section className="dashboard-grid">
        <div className="panel chart-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">RISK DISTRIBUTION</span>
              <h3>Prioritized vulnerability landscape</h3>
            </div>

            <Shield size={19} />
          </div>

          <div className="chart-area">
            <Suspense
              fallback={
                <div className="chart-loading">Loading risk distribution…</div>
              }
            >
              <RiskDistributionChart data={categoryData} />
            </Suspense>
          </div>
        </div>

        <div className="panel model-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">PRODUCTION MODEL</span>
              <h3>ThreatLens intelligence</h3>
            </div>

            <BrainCircuit size={19} />
          </div>

          <div className="model-ring">
            <div className="ring-inner">
              <strong>{summary?.model_version || "—"}</strong>
              <span>model version</span>
            </div>
          </div>

          <div className="model-points">
            <div>
              <span className="point-icon">
                <Target size={14} />
              </span>
              <div>
                <strong>Context-aware ranking</strong>
                <small>Production risk categories</small>
              </div>
            </div>

            <div>
              <span className="point-icon">
                <GitBranch size={14} />
              </span>
              <div>
                <strong>Evidence-connected</strong>
                <small>CVE · CWE · CPE · Reference</small>
              </div>
            </div>

            <div>
              <span className="point-icon">
                <Shield size={14} />
              </span>
              <div>
                <strong>Scored production output</strong>
                <small>Risk categories and model scores</small>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <span className="panel-kicker">PRIORITY QUEUE</span>
            <h3>Highest-risk vulnerabilities</h3>
          </div>

          <button
            className="text-button"
            onClick={() => onNavigate("vulnerabilities")}
          >
            View all <ChevronRight size={15} />
          </button>
        </div>

        <VulnerabilityTable
          vulnerabilities={topVulnerabilities}
          onOpen={onOpen}
        />
      </section>

      <LogAnalysis />

      <section className="bottom-grid">
        <div className="panel mini-panel">
          <div className="mini-icon">
            <Network size={19} />
          </div>
          <div>
            <span>KNOWLEDGE GRAPH</span>
            <strong>{formatNumber(graphSummary?.nodes)} nodes</strong>
            <small>{formatNumber(graphSummary?.edges)} relationships</small>
          </div>
          <button onClick={() => onNavigate("graph")}>
            Explore <ChevronRight size={15} />
          </button>
        </div>

        <div className="panel mini-panel">
          <div className="mini-icon">
            <Terminal size={19} />
          </div>
          <div>
            <span>SYSTEM STATUS</span>
            <strong>{system?.status || "Unavailable"}</strong>
            <small>
              {apiStatusMessage(system)}
            </small>
          </div>
          <div
            className={`online-pill ${
              system?.status === "ready" ? "" : "disconnected"
            }`}
          >
            <span />
            {system?.status === "ready" ? "Connected" : "Unavailable"}
          </div>
        </div>
      </section>
    </div>
  );
}

function VulnerabilityPage({
  vulnerabilities,
  search,
  setSearch,
  category,
  setCategory,
  kevOnly,
  setKevOnly,
  loading,
  onOpen,
}) {
  return (
    <div className="page">
      <section className="page-intro">
        <div>
          <span className="panel-kicker">VULNERABILITY INTELLIGENCE</span>
          <h3>Investigate and prioritize CVEs</h3>
          <p>
            Search the production inference dataset and inspect the evidence
            behind each ThreatLens risk classification.
          </p>
        </div>
      </section>

      <div className="filter-bar">
        <div className="search-box">
          <Search size={17} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search CVE identifier..."
            aria-label="Search CVE identifier"
          />
        </div>

        <select
          value={category}
          onChange={(event) => setCategory(event.target.value)}
          aria-label="Filter by risk category"
        >
          <option value="ALL">All categories</option>
          <option value="CRITICAL">Critical</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
          <option value="MINIMAL">Minimal</option>
        </select>

        <button
          className={`filter-toggle ${kevOnly ? "selected" : ""}`}
          onClick={() => setKevOnly((value) => !value)}
          aria-pressed={kevOnly}
        >
          <AlertTriangle size={15} />
          CISA KEV only
        </button>
      </div>

      <div className="panel table-panel">
        <div className="table-toolbar">
          <span>
            {loading ? (
              <span role="status">Updating results…</span>
            ) : (
              <>
                Showing <strong>{vulnerabilities.length}</strong> returned
                results (API limit: 100)
              </>
            )}
          </span>
        </div>

        <VulnerabilityTable
          vulnerabilities={vulnerabilities}
          onOpen={onOpen}
          loading={loading}
          expanded
        />
      </div>
    </div>
  );
}

function VulnerabilityTable({ vulnerabilities, onOpen, expanded, loading }) {
  if (!vulnerabilities.length) {
    return (
      <div className="empty-state">
        {loading ? <Activity size={25} /> : <Search size={25} />}
        <strong>
          {loading ? "Loading vulnerabilities" : "No vulnerabilities found"}
        </strong>
        <span>
          {loading
            ? "The filtered results are being updated."
            : "Try changing the current filters."}
        </span>
      </div>
    );
  }

  return (
    <div className={`data-table ${expanded ? "expanded" : ""}`}>
      <div className="table-header">
        <span>CVE</span>
        <span>THREATLENS</span>
        <span>RISK</span>
        <span>CVSS</span>
        <span>KEV</span>
        <span />
      </div>

      {vulnerabilities.map((item) => (
        <button
          className="table-row"
          key={item.cve_id}
          onClick={() => onOpen(item.cve_id)}
        >
          <span className="cve-cell">
            <strong>{item.cve_id}</strong>
            <small>{item.published_date || "Publication unavailable"}</small>
          </span>

          <span className="score-cell">
            <strong>{formatScore(item.threatlens_score)}</strong>
            <div className="score-bar">
              <span
                style={{
                  width: `${Math.min(
                    Number(item.threatlens_score || 0) * 100,
                    100
                  )}%`,
                }}
              />
            </div>
          </span>

          <span>
            <CategoryBadge category={item.risk_category} />
          </span>

          <span className="cvss-cell">
            {item.cvss_score == null
              ? "—"
              : Number(item.cvss_score).toFixed(1)}
          </span>

          <span>
            {item.known_exploited ? (
              <span className="kev-badge">
                <span />
                KEV
              </span>
            ) : (
              <span className="muted">—</span>
            )}
          </span>

          <span className="row-arrow">
            <ChevronRight size={17} />
          </span>
        </button>
      ))}
    </div>
  );
}

function GraphPage({ graphSummary, vulnerabilities, initialCveId }) {
  const [cveId, setCveId] = useState(
    initialCveId || vulnerabilities[0]?.cve_id || ""
  );
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function exploreGraph(event) {
    event.preventDefault();
    const normalizedId = cveId.trim().toUpperCase();

    if (!normalizedId) {
      setError("Enter a CVE identifier to explore its graph.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      const response = await axios.get(
        `${API}/graph/${encodeURIComponent(normalizedId)}`
      );
      setGraph(response.data);
    } catch (err) {
      setGraph(null);
      setError(apiErrorMessage(err, "Unable to load graph relationships."));
    } finally {
      setLoading(false);
    }
  }

  const nodeLabels = new Map(
    (graph?.nodes || []).map((node) => [node.node_id, node.label || node.node_id])
  );

  return (
    <div className="page">
      <section className="page-intro">
        <div>
          <span className="panel-kicker">THREAT KNOWLEDGE GRAPH</span>
          <h3>Connect vulnerabilities to evidence</h3>
          <p>
            ThreatLens models CVEs and their contextual security relationships
            as a structured graph.
          </p>
        </div>
      </section>

      <div className="graph-stats">
        <div className="graph-stat">
          <Network size={19} />
          <span>Nodes</span>
          <strong>{formatNumber(graphSummary?.nodes)}</strong>
        </div>

        <div className="graph-stat">
          <GitBranch size={19} />
          <span>Edges</span>
          <strong>{formatNumber(graphSummary?.edges)}</strong>
        </div>

        <div className="graph-stat">
          <Target size={19} />
          <span>CVEs represented</span>
          <strong>{formatNumber(graphSummary?.cves)}</strong>
        </div>

        <div className="graph-stat">
          <AlertTriangle size={19} />
          <span>KEV links</span>
          <strong>{formatNumber(graphSummary?.kev_edges)}</strong>
        </div>
      </div>

      <section className="panel graph-explorer">
        <div className="panel-heading">
          <div>
            <span className="panel-kicker">LIVE GRAPH EXPLORER</span>
            <h3>Explore a vulnerability’s relationships</h3>
          </div>
          <Network size={19} />
        </div>

        <form className="graph-search-form" onSubmit={exploreGraph}>
          <label htmlFor="graph-cve">CVE identifier</label>
          <div className="graph-search-controls">
            <input
              id="graph-cve"
              value={cveId}
              onChange={(event) => setCveId(event.target.value)}
              placeholder="CVE-2025-9242"
              autoComplete="off"
            />
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Loading…" : "Explore graph"}
            </button>
          </div>
          <small>Results are loaded from the ThreatLens knowledge graph.</small>
        </form>

        {error && (
          <div className="inline-error" role="alert">
            <AlertTriangle size={15} />
            {error}
          </div>
        )}

        {graph && (
          <div className="graph-result">
            <div className="graph-result-header">
              <div>
                <span className="panel-kicker">RELATIONSHIPS FOR</span>
                <h3>{graph.cve_id}</h3>
              </div>
              <span className="graph-count">
                {graph.nodes.length} nodes · {graph.edges.length} edges
              </span>
            </div>

            <div className="graph-node-list">
              {graph.nodes.map((node) => (
                <div
                  className={`graph-entity ${
                    node.node_type === "CVE" ? "root-entity" : ""
                  }`}
                  key={node.node_id}
                >
                  <span>{node.node_type || "ENTITY"}</span>
                  <strong>{node.label || node.node_id}</strong>
                  {node.risk_category && (
                    <CategoryBadge category={node.risk_category} />
                  )}
                </div>
              ))}
            </div>

            {graph.edges.length ? (
              <div className="graph-edge-list">
                {graph.edges.map((edge, index) => (
                  <div
                    className="graph-edge"
                    key={`${edge.source}-${edge.relation}-${edge.target}-${index}`}
                  >
                    <strong>{nodeLabels.get(edge.source) || edge.source}</strong>
                    <span>{edge.relation || "RELATED_TO"}</span>
                    <strong>{nodeLabels.get(edge.target) || edge.target}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state compact-empty">
                <strong>No relationships returned</strong>
                <span>This CVE is present without connected graph records.</span>
              </div>
            )}
          </div>
        )}
      </section>

      <div className="panel table-panel">
        <div className="panel-heading">
          <div>
            <span className="panel-kicker">GRAPH SAMPLE</span>
            <h3>Choose a vulnerability to explore</h3>
          </div>
        </div>

        <VulnerabilityTable
          vulnerabilities={vulnerabilities.slice(0, 12)}
          onOpen={(selectedCveId) => {
            setCveId(selectedCveId);
            setGraph(null);
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        />
      </div>
    </div>
  );
}

function ResearchPage({ summary }) {
  const riskDistribution = Object.entries(summary?.risk_distribution || {}).map(
    ([name, value]) => ({
      name,
      value,
      shortName:
        name === "MINIMAL"
          ? "Minimal"
          : name.charAt(0) + name.slice(1).toLowerCase(),
    })
  );

  return (
    <div className="page">
      <section className="page-intro">
        <div>
          <span className="panel-kicker">RESEARCH & EVALUATION</span>
          <h3>Production model and evaluation data</h3>
          <p>
            Explore the model outputs currently served by the ThreatLens
            application API.
          </p>
        </div>
      </section>

      <div className="research-banner">
        <div className="research-icon">
          <AlertTriangle size={24} />
        </div>

        <div>
          <span>COMPARATIVE METRICS UNAVAILABLE</span>
          <strong>
            The application API does not currently provide CVSS-only or
            contextual-baseline evaluation metrics. Only live production
            summary values are shown below.
          </strong>
        </div>
      </div>

      <div className="research-grid">
        <div className="panel research-chart">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">PRODUCTION RISK DISTRIBUTION</span>
              <h3>Records by ThreatLens category</h3>
            </div>
          </div>

          <div className="chart-area research-chart-area">
            <Suspense
              fallback={
                <div className="chart-loading">Loading risk distribution…</div>
              }
            >
              <RiskDistributionChart data={riskDistribution} />
            </Suspense>
          </div>
        </div>

        <div className="panel research-notes">
          <span className="panel-kicker">LIVE MODEL SUMMARY</span>

          <div className="research-note">
            <Database size={17} />
            <div>
              <strong>Prediction dataset</strong>
              <p>
                {formatNumber(summary?.total_records)} vulnerability records
                are currently available to the dashboard.
              </p>
            </div>
          </div>

          <div className="research-note">
            <AlertTriangle size={17} />
            <div>
              <strong>Known exploited vulnerabilities</strong>
              <p>
                {formatNumber(summary?.known_exploited)} records are marked
                known exploited in the production dataset.
              </p>
            </div>
          </div>

          <div className="research-note">
            <Gauge size={17} />
            <div>
              <strong>Average ThreatLens score</strong>
              <p>
                {summary?.average_threatlens_score == null
                  ? "Unavailable from the application API."
                  : formatScore(summary.average_threatlens_score)}
                {" "}from the live dashboard summary.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function IntelligencePage({
  title,
  icon: Icon,
  description,
  items,
  endpoint,
  exampleCveId,
}) {
  const [cveId, setCveId] = useState(exampleCveId || "");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submitLookup(event) {
    event.preventDefault();
    const normalizedId = cveId.trim().toUpperCase();

    if (!normalizedId) {
      setError("Enter a CVE identifier to continue.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      const response = await axios.get(
        `${API}/vulnerabilities/${encodeURIComponent(normalizedId)}/${endpoint}`
      );
      setResult(response.data);
    } catch (err) {
      setResult(null);
      setError(
        apiErrorMessage(
          err,
          `Unable to load ${title.toLowerCase()} results.`
        )
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <section className="intelligence-hero">
        <div className="intelligence-icon">
          <Icon size={30} />
        </div>

        <div>
          <span className="panel-kicker">THREATLENS INTELLIGENCE</span>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </section>

      <section className="panel intelligence-lookup">
        <div className="panel-heading">
          <div>
            <span className="panel-kicker">LIVE API LOOKUP</span>
            <h3>{title} for a CVE</h3>
          </div>
          <Icon size={19} />
        </div>

        <form className="graph-search-form" onSubmit={submitLookup}>
          <label htmlFor={`${endpoint}-cve`}>CVE identifier</label>
          <div className="graph-search-controls">
            <input
              id={`${endpoint}-cve`}
              value={cveId}
              onChange={(event) => setCveId(event.target.value)}
              placeholder="CVE-2025-9242"
              autoComplete="off"
            />
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Loading…" : `Run ${title}`}
            </button>
          </div>
          <small>
            {exampleCveId
              ? `Pre-filled with ${exampleCveId} from the priority queue.`
              : "Enter a CVE identifier from the production dataset."}
          </small>
        </form>

        {error && (
          <div className="inline-error" role="alert">
            <AlertTriangle size={15} />
            {error}
          </div>
        )}

        {result && (
          <IntelligenceResult endpoint={endpoint} result={result} />
        )}
      </section>

      <div className="intelligence-grid">
        {items.map((item, index) => (
          <div className="intelligence-card" key={item}>
            <div className="intelligence-number">
              {String(index + 1).padStart(2, "0")}
            </div>

            <div>
              <strong>{item}</strong>
              <small>
                {title === "AI Investigator"
                  ? "Structured security intelligence"
                  : "Evidence-backed security context"}
              </small>
            </div>
          </div>
        ))}
      </div>

      <div className="panel architecture-panel">
        <span className="panel-kicker">PIPELINE</span>

        <div className="architecture-flow">
          <span>DATA</span>
          <ChevronRight size={16} />
          <span>NORMALIZE</span>
          <ChevronRight size={16} />
          <span>RANK</span>
          <ChevronRight size={16} />
          <span>EVIDENCE</span>
          <ChevronRight size={16} />
          <span>INVESTIGATE</span>
        </div>
      </div>
    </div>
  );
}

function IntelligenceResult({ endpoint, result }) {
  if (endpoint === "investigation") {
    const risk = result.risk || {};

    return (
      <div className="intelligence-result">
        <div className="investigation-summary">
          <div>
            <span className="panel-kicker">
              {result.status || "INVESTIGATION"}
            </span>
            <h3>{result.cve_id}</h3>
            <p>{result.finding || "No finding was returned."}</p>
          </div>
          <div className="confidence-chip">
            {String(result.confidence || "Confidence unavailable").replaceAll(
              "_",
              " "
            )}
          </div>
        </div>

        {risk.category && (
          <div className="investigation-risk">
            <span>THREATLENS RISK</span>
            <strong>{formatScore(risk.score)}</strong>
            <CategoryBadge category={risk.category} />
          </div>
        )}

        <div className="investigation-facts">
          {(result.facts || []).map((fact) => (
            <article className="investigation-fact" key={fact.id}>
              <div>
                <span>{fact.id} · {fact.type?.replaceAll("_", " ")}</span>
                <strong>{fact.statement}</strong>
                <small>{fact.source}</small>
              </div>
              <span className="evidence-status">
                {fact.evidence_status || "UNKNOWN"}
              </span>
            </article>
          ))}
        </div>

        {result.limitations?.length > 0 && (
          <div className="result-limitations">
            <strong>Limitations</strong>
            <ul>
              {result.limitations.map((limitation, index) => (
                <li key={`${limitation}-${index}`}>{limitation}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  const sections = Object.entries(result.evidence || {});

  return (
    <div className="intelligence-result">
      <div className="investigation-summary">
        <div>
          <span className="panel-kicker">
            {result.status || "EVIDENCE"}
          </span>
          <h3>{result.cve_id}</h3>
          <p>
            {formatNumber(result.summary?.known)} known ·{" "}
            {formatNumber(result.summary?.inferred)} inferred ·{" "}
            {formatNumber(result.summary?.unknown)} unknown evidence groups
          </p>
        </div>
      </div>

      <div className="evidence-grid">
        {sections.map(([sectionName, fields]) => (
          <article className="evidence-card" key={sectionName}>
            <div className="evidence-card-header">
              <strong>{sectionName.replaceAll("_", " ")}</strong>
              <span>{fields.status || "UNKNOWN"}</span>
            </div>
            {Object.entries(fields)
              .filter(([fieldName]) => fieldName !== "status")
              .map(([fieldName, value]) => (
                <EvidenceField
                  key={fieldName}
                  label={fieldName}
                  value={value}
                />
              ))}
          </article>
        ))}
      </div>
    </div>
  );
}

function EvidenceField({ label, value }) {
  if (value == null || typeof value === "boolean") {
    return (
      <div className="evidence-field">
        <span>{label.replaceAll("_", " ")}</span>
        <strong>{value == null ? "Unavailable" : value ? "Yes" : "No"}</strong>
      </div>
    );
  }

  if (Array.isArray(value)) {
    const isReferenceList = value.some(
      (item) => item && typeof item === "object" && item.url
    );
    const visibleItems = value.slice(0, isReferenceList ? 4 : 5);

    return (
      <div className="evidence-field evidence-array">
        <span>{label.replaceAll("_", " ")} · {value.length}</span>
        <div>
          {visibleItems.map((item, index) => {
            if (item && typeof item === "object" && item.url) {
              return (
                <a
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  key={`${item.url}-${index}`}
                >
                  <ExternalLink size={12} />
                  {item.label || item.url}
                </a>
              );
            }

            return (
              <code key={`${String(item)}-${index}`}>
                {typeof item === "object"
                  ? JSON.stringify(item)
                  : String(item)}
              </code>
            );
          })}
          {value.length > visibleItems.length && (
            <small>+ {value.length - visibleItems.length} more</small>
          )}
        </div>
      </div>
    );
  }

  const displayValue =
    typeof value === "object" ? JSON.stringify(value) : String(value);

  return (
    <div className="evidence-field">
      <span>{label.replaceAll("_", " ")}</span>
      <strong>{displayValue}</strong>
    </div>
  );
}

function DetailDrawer({ vulnerability, onExplore, onClose }) {
  const [evidence, setEvidence] = useState(null);
  const [evidenceLoading, setEvidenceLoading] = useState(true);
  const [evidenceError, setEvidenceError] = useState("");
  const drawerRef = useRef(null);
  const closeButtonRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadEvidence() {
      try {
        setEvidenceLoading(true);
        setEvidenceError("");
        const response = await axios.get(
          `${API}/vulnerabilities/${encodeURIComponent(vulnerability.cve_id)}/evidence`,
          { signal: controller.signal }
        );
        setEvidence(response.data);
      } catch (error) {
        if (!axios.isCancel(error)) {
          setEvidenceError(
            apiErrorMessage(error, "Evidence details are unavailable.")
          );
        }
      } finally {
        if (!controller.signal.aborted) setEvidenceLoading(false);
      }
    }

    void loadEvidence();
    return () => controller.abort();
  }, [vulnerability.cve_id]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const previouslyFocused = document.activeElement;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        onClose();
        return;
      }

      if (event.key !== "Tab" || !drawerRef.current) return;
      const focusable = drawerRef.current.querySelectorAll(
        'button, a[href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
      if (previouslyFocused instanceof HTMLElement && previouslyFocused.isConnected) {
        previouslyFocused.focus({ preventScroll: true });
      }
    };
  }, [onClose]);

  const sections = evidence?.evidence || {};
  const cweItems = sections.cwe?.cwes || vulnerability.cwe_ids || [];
  const products = sections.affected_products?.products || [];
  const references =
    sections.references?.references ||
    (vulnerability.references || []).map((reference) =>
      typeof reference === "string"
        ? { url: reference, label: reference }
        : reference
    );
  const affectedCount =
    sections.affected_products?.count ??
    (products.length || vulnerability.affected_cpe_count);
  const referenceCount =
    sections.references?.count ??
    (references.length || vulnerability.reference_count);
  const kev = sections.kev?.known_exploited;

  return (
    <div
      className="drawer-backdrop"
      onClick={onClose}
      role="presentation"
    >
      <aside
        className="detail-drawer"
        ref={drawerRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="vulnerability-detail-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="drawer-header">
          <div>
            <span className="panel-kicker">VULNERABILITY DETAIL</span>
            <h3 id="vulnerability-detail-title">{vulnerability.cve_id}</h3>
          </div>

          <button
            className="close-button"
            ref={closeButtonRef}
            onClick={onClose}
            aria-label="Close vulnerability details"
            title="Close"
          >
            ×
          </button>
        </div>

        <div className="drawer-score">
          <div>
            <span>THREATLENS SCORE</span>
            <strong>{formatScore(vulnerability.threatlens_score)}</strong>
          </div>

          <CategoryBadge category={vulnerability.risk_category} />
        </div>

        <div className="detail-shortcuts">
          <button onClick={() => onExplore("graph", vulnerability.cve_id)}>
            <Network size={14} />
            Graph
          </button>
          <button onClick={() => onExplore("evidence", vulnerability.cve_id)}>
            <FileSearch size={14} />
            Evidence
          </button>
          <button
            onClick={() => onExplore("investigator", vulnerability.cve_id)}
          >
            <BrainCircuit size={14} />
            Investigate
          </button>
        </div>

        <div className="detail-list">
          <DetailRow
            label="CVSS"
            value={
              vulnerability.cvss_score == null
                ? "Unavailable"
                : Number(vulnerability.cvss_score).toFixed(1)
            }
          />

          <DetailRow
            label="Published"
            value={
              sections.metadata?.published_date ||
              vulnerability.published_date ||
              "Unavailable"
            }
          />

          <DetailRow
            label="CISA KEV"
            value={
              kev == null
                ? evidenceLoading
                  ? "Loading…"
                  : vulnerability.known_exploited
                    ? "Known exploited"
                    : "Not listed"
                : kev
                  ? `Known exploited${
                      sections.kev?.kev_date_added
                        ? ` · added ${sections.kev.kev_date_added}`
                        : ""
                    }`
                  : "Not listed"
            }
          />

          <DetailRow
            label="CWE"
            value={
              cweItems.length
                ? cweItems.join(", ")
                : evidenceLoading
                  ? "Loading…"
                  : "Unavailable"
            }
          />

          <DetailRow
            label="Affected products"
            value={
              affectedCount == null
                ? evidenceLoading
                  ? "Loading…"
                  : "Unavailable"
                : formatNumber(affectedCount)
            }
          />

          <DetailRow
            label="References"
            value={
              referenceCount == null
                ? evidenceLoading
                  ? "Loading…"
                  : "Unavailable"
                : formatNumber(referenceCount)
            }
          />
        </div>

        {evidenceError && (
          <div className="inline-error" role="alert">
            <AlertTriangle size={15} />
            {evidenceError}
          </div>
        )}

        {products.length > 0 && (
          <div className="references-block">
            <span>AFFECTED PRODUCTS</span>
            {products.slice(0, 5).map((product) => (
              <code key={product}>{product}</code>
            ))}
            {products.length > 5 && (
              <small>+ {products.length - 5} more products</small>
            )}
          </div>
        )}

        {vulnerability.description && (
          <div className="description-block">
            <span>DESCRIPTION</span>
            <p>{vulnerability.description}</p>
          </div>
        )}

        {references.length > 0 && (
          <div className="references-block">
            <span>REFERENCES</span>

            {references.slice(0, 8).map((reference, index) => (
              <a
                href={reference.url}
                target="_blank"
                rel="noreferrer"
                key={`${reference.url}-${index}`}
              >
                <ExternalLink size={13} />
                {reference.label || `Reference ${index + 1}`}
              </a>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
