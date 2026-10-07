import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Download,
  FileSearch,
  FileText,
  HardDrive,
  Info,
  LoaderCircle,
  LockKeyhole,
  ShieldAlert,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import "./LogAnalysis.css";

const MAX_FILE_BYTES = 25 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = new Set(["csv", "json", "log", "txt"]);
const severityNames = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
const severityLabels = {
  CRITICAL: "Critical",
  HIGH: "High",
  MEDIUM: "Medium",
  LOW: "Low",
};
const numberFormat = new Intl.NumberFormat("en-US");

function formatDate(value, includeTime = false) {
  if (!value) return "Time unavailable";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "Time unavailable";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    ...(includeTime ? { timeStyle: "short" } : {}),
  }).format(date);
}

function formatBytes(bytes) {
  return new Intl.NumberFormat(undefined, {
    style: "unit",
    unit: bytes >= 1_000_000 ? "megabyte" : "kilobyte",
    maximumFractionDigits: 1,
  }).format(bytes / (bytes >= 1_000_000 ? 1_000_000 : 1_000));
}

function entityList(values) {
  return values.length ? values.join(", ") : "Not identified in matched events";
}

export default function LogAnalysis() {
  const [file, setFile] = useState(null);
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [expandedFinding, setExpandedFinding] = useState(null);
  const inputRef = useRef(null);
  const workerRef = useRef(null);

  useEffect(
    () => () => {
      workerRef.current?.terminate();
    },
    []
  );

  function selectFile(candidate) {
    setError("");
    setReport(null);
    setExpandedFinding(null);
    if (!candidate) {
      setFile(null);
      return;
    }

    const extension = candidate.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.has(extension)) {
      setFile(null);
      setError("Choose a supported .csv, .json, .log, or .txt file.");
      return;
    }
    if (candidate.size === 0) {
      setFile(null);
      setError("The selected file is empty.");
      return;
    }
    if (candidate.size > MAX_FILE_BYTES) {
      setFile(null);
      setError("Files must be 25 MB or smaller. Split larger logs into smaller time windows.");
      return;
    }

    setFile(candidate);
    setStatus("ready");
  }

  async function analyzeSelectedFile() {
    if (!file || status === "analyzing") return;
    setError("");
    setReport(null);
    setStatus("reading");

    try {
      const contents = await file.text();
      setStatus("analyzing");
      workerRef.current?.terminate();
      const worker = new Worker(
        new URL("./logAnalysis.worker.js", import.meta.url),
        { type: "module" }
      );
      workerRef.current = worker;
      worker.onmessage = (event) => {
        worker.terminate();
        workerRef.current = null;
        if (event.data.type === "error") {
          setError(event.data.message);
          setStatus("error");
          return;
        }
        setReport(event.data.report);
        setStatus("complete");
      };
      worker.onerror = () => {
        worker.terminate();
        workerRef.current = null;
        setError("Local analysis could not be completed. Check the file format and try again.");
        setStatus("error");
      };
      worker.postMessage({ text: contents, fileName: file.name });
    } catch (readError) {
      setError(
        readError instanceof Error
          ? readError.message
          : "The selected file could not be read."
      );
      setStatus("error");
    }
  }

  function clearAnalysis() {
    workerRef.current?.terminate();
    workerRef.current = null;
    setFile(null);
    setReport(null);
    setError("");
    setStatus("idle");
    setExpandedFinding(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function downloadReport() {
    if (!report) return;
    const exportReport = {
      reportType: "ThreatLens optional local log analysis",
      generatedAt: report.analyzedAt,
      analysisMethod:
        "Deterministic local pattern checks. Results are investigative leads, not a verdict that activity is malicious.",
      rawUploadedLogIncluded: false,
      report,
    };
    const blob = new Blob([JSON.stringify(exportReport, null, 2)], {
      type: "application/json",
    });
    const objectUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = `threatlens-log-analysis-${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  }

  const isBusy = status === "reading" || status === "analyzing";

  return (
    <section className="log-analysis panel" aria-labelledby="log-analysis-title">
      <div className="log-analysis-heading">
        <div className="log-analysis-heading-icon">
          <FileSearch size={20} />
        </div>
        <div>
          <span className="panel-kicker">OPTIONAL · LOCAL ANALYSIS</span>
          <h3 id="log-analysis-title">Analyze Your Logs</h3>
          <p>
            Review an exported log file for grouped activity patterns without
            connecting it to your existing telemetry.
          </p>
        </div>
        <span className="local-only-badge">
          <LockKeyhole size={12} />
          Processed in this browser
        </span>
      </div>

      <div
        className={`log-upload-zone ${dragging ? "is-dragging" : ""} ${
          file ? "has-file" : ""
        }`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget)) {
            setDragging(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          selectFile(event.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          className="log-file-input"
          type="file"
          accept=".csv,.json,.log,.txt,text/csv,application/json,text/plain"
          aria-label="Choose a security log file"
          onChange={(event) => selectFile(event.target.files?.[0])}
        />
        <div className="log-upload-icon">
          {file ? <FileText size={20} /> : <Upload size={20} />}
        </div>
        <div className="log-upload-copy">
          <strong>{file ? file.name : "Drop a log file here or browse"}</strong>
          <span>
            {file
              ? `${formatBytes(file.size)} · Ready for local analysis`
              : "CSV, JSON, NDJSON, .log, or plain text · up to 25 MB"}
          </span>
        </div>
        <button
          className="log-secondary-button"
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={isBusy}
        >
          <Upload size={14} />
          Upload Logs
        </button>
      </div>

      <div className="log-privacy-note">
        <LockKeyhole size={14} />
        <span>
          Your file is read and analyzed locally. It is not sent to ThreatLens
          servers or saved by this feature. Clear the analysis when finished.
        </span>
      </div>

      <div className="log-workflow">
        <div className={`log-step ${file ? "is-ready" : "is-current"}`}>
          <span>{file ? <CheckCircle2 size={14} /> : "1"}</span>
          <div>
            <strong>Select a file</strong>
            <small>Supported formats only</small>
          </div>
        </div>
        <div className={`log-step ${isBusy || report ? "is-ready" : ""}`}>
          <span>{report ? <CheckCircle2 size={14} /> : "2"}</span>
          <div>
            <strong>Parse and analyze</strong>
            <small>Normalize fields and group patterns</small>
          </div>
        </div>
        <div className={`log-step ${report ? "is-ready" : ""}`}>
          <span>{report ? <CheckCircle2 size={14} /> : "3"}</span>
          <div>
            <strong>Review findings</strong>
            <small>Evidence, context, and next steps</small>
          </div>
        </div>
      </div>

      <div className="log-actions">
        <button
          className="primary-button"
          type="button"
          onClick={analyzeSelectedFile}
          disabled={!file || isBusy}
        >
          {isBusy ? (
            <LoaderCircle size={15} className="log-spin" />
          ) : (
            <ShieldAlert size={15} />
          )}
          {status === "reading"
            ? "Reading locally..."
            : status === "analyzing"
              ? "Analyzing patterns..."
              : "Upload & Analyze"}
        </button>
        {report && (
          <button className="log-secondary-button" type="button" onClick={downloadReport}>
            <Download size={15} />
            Download report
          </button>
        )}
        {(file || report || error) && (
          <button
            className="log-clear-button"
            type="button"
            onClick={clearAnalysis}
            disabled={isBusy}
          >
            <Trash2 size={14} />
            Clear
          </button>
        )}
        {isBusy && (
          <span className="log-progress" role="status" aria-live="polite">
            <LoaderCircle size={14} className="log-spin" />
            {status === "reading"
              ? "Reading file in browser memory…"
              : "Detecting format, normalizing events, and grouping patterns…"}
          </span>
        )}
      </div>

      {error && (
        <div className="log-error" role="alert">
          <AlertTriangle size={16} />
          <span>{error}</span>
          <button type="button" onClick={() => setError("")} aria-label="Dismiss log analysis error">
            <X size={15} />
          </button>
        </div>
      )}

      {report && (
        <AnalysisReport
          report={report}
          expandedFinding={expandedFinding}
          setExpandedFinding={setExpandedFinding}
        />
      )}
      <p className="log-method-note">
        Pattern checks are heuristic and depend on the fields present. They do
        not replace a SIEM, endpoint investigation, threat-intelligence
        enrichment, or analyst validation. No matched pattern does not mean a
        log is safe.
      </p>
    </section>
  );
}

function AnalysisReport({ report, expandedFinding, setExpandedFinding }) {
  const timelineMaximum = Math.max(1, ...report.timeline.map((item) => item.count));

  return (
    <div className="log-report" aria-live="polite">
      <div className="log-report-header">
        <div>
          <span className="panel-kicker">LOCAL ANALYSIS COMPLETE</span>
          <h4>Investigation overview</h4>
          <p>
            {report.detectedFormat} detected · {report.fields.length} available
            fields · {formatDate(report.analyzedAt, true)}
          </p>
        </div>
        <div className={`log-risk-score ${
          report.riskScore >= 70 ? "risk-high" : report.riskScore >= 40 ? "risk-medium" : "risk-low"
        }`}>
          <span>HEURISTIC RISK</span>
          <strong>{report.riskScore}<small>/100</small></strong>
        </div>
      </div>

      <div className="log-summary-grid">
        <SummaryMetric label="Events analyzed" value={numberFormat.format(report.totalEvents)} />
        <SummaryMetric label="Grouped findings" value={numberFormat.format(report.findingCount)} />
        {severityNames.map((severity) => (
          <SummaryMetric
            key={severity}
            label={severityLabels[severity]}
            value={numberFormat.format(report.severityCounts[severity])}
            severity={severity.toLowerCase()}
          />
        ))}
      </div>

      <div className="log-report-grid">
        <section className="log-report-card">
          <div className="log-card-heading">
            <div>
              <span className="panel-kicker">DETECTED PATTERNS</span>
              <h4>Potential incidents</h4>
            </div>
            <span className="log-count-badge">{report.findingCount}</span>
          </div>

          {report.findings.length ? (
            <div className="log-findings-list">
              {report.findings.map((finding) => {
                const expanded = expandedFinding === finding.id;
                return (
                  <article className="log-finding" key={finding.id}>
                    <button
                      className="log-finding-toggle"
                      type="button"
                      aria-expanded={expanded}
                      onClick={() => setExpandedFinding(expanded ? null : finding.id)}
                    >
                      <span className={`log-severity-dot severity-${finding.severity.toLowerCase()}`} />
                      <span className="log-finding-main">
                        <strong>{finding.title}</strong>
                        <small>{finding.eventCount} related event{finding.eventCount === 1 ? "" : "s"} · risk {finding.riskScore}/100</small>
                      </span>
                      <span className={`log-severity-label severity-${finding.severity.toLowerCase()}`}>
                        {severityLabels[finding.severity]}
                      </span>
                      <ChevronDown size={15} className={expanded ? "is-expanded" : ""} />
                    </button>
                    {expanded && (
                      <div className="log-finding-details">
                        <p>{finding.detail}</p>
                        <EvidenceLabel text={finding.evidence} />
                        {finding.technique && (
                          <a
                            className="log-technique-link"
                            href={`https://attack.mitre.org/techniques/${finding.technique.id}/`}
                            target="_blank"
                            rel="noreferrer"
                          >
                            MITRE ATT&CK {finding.technique.id}: {finding.technique.name}
                          </a>
                        )}
                        <dl className="log-entity-list">
                          <dt>Source / destination IPs</dt>
                          <dd>{entityList(finding.ips)}</dd>
                          <dt>Users</dt>
                          <dd>{entityList(finding.users)}</dd>
                          <dt>Hosts</dt>
                          <dd>{entityList(finding.hosts)}</dd>
                          <dt>Observed time range</dt>
                          <dd>{formatDate(finding.firstSeen, true)} – {formatDate(finding.lastSeen, true)}</dd>
                        </dl>
                        <div className="log-recommendation">
                          <strong>Recommended validation</strong>
                          <p>{finding.recommendation}</p>
                        </div>
                      </div>
                    )}
                  </article>
                );
              })}
              {report.findingsOmitted > 0 && (
                <p className="log-findings-limit">
                  Showing the first {numberFormat.format(report.findings.length)}{" "}
                  grouped findings. {numberFormat.format(report.findingsOmitted)}{" "}
                  additional groups are included in the severity totals but
                  omitted from this view.
                </p>
              )}
            </div>
          ) : (
            <div className="log-no-findings">
              <CheckCircle2 size={18} />
              <div>
                <strong>No configured pattern matched</strong>
                <p>This is not proof that the events are benign. Review field coverage and use your normal detection workflow.</p>
              </div>
            </div>
          )}
        </section>

        <div className="log-side-column">
          <section className="log-report-card">
            <div className="log-card-heading">
              <div>
                <span className="panel-kicker">OBSERVED ENTITIES</span>
                <h4>Items in matched activity</h4>
              </div>
            </div>
            <EntityGroup label="IP addresses" values={report.suspicious.ips} />
            <EntityGroup label="Users" values={report.suspicious.users} />
            <EntityGroup label="Hosts" values={report.suspicious.hosts} />
            <p className="log-entity-disclaimer">
              These entities appear in matched events; their presence alone does
              not establish malicious intent or IOC status.
            </p>
          </section>

          <section className="log-report-card log-timeline-card">
            <div className="log-card-heading">
              <div>
                <span className="panel-kicker">EVENT TIMELINE</span>
                <h4>Volume by hour</h4>
              </div>
              <Clock3 size={16} />
            </div>
            {report.timeline.length ? (
              <div className="log-timeline">
                {report.timeline.map((item) => (
                  <div className="log-timeline-row" key={item.timestamp}>
                    <time dateTime={item.timestamp}>
                      {formatDate(item.timestamp, true)}
                    </time>
                    <span className="log-timeline-track">
                      <span style={{ width: `${Math.max(3, (item.count / timelineMaximum) * 100)}%` }} />
                    </span>
                    <strong>{numberFormat.format(item.count)}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <p className="log-empty-timeline">No parseable timestamps were found. Event totals and pattern checks are still available.</p>
            )}
          </section>
        </div>
      </div>

      <section className="log-report-card log-field-card">
        <div className="log-card-heading">
          <div>
            <span className="panel-kicker">INPUT COVERAGE</span>
            <h4>Available fields</h4>
          </div>
          <span>{report.eventsWithTimestamps} events with parseable timestamps</span>
        </div>
        <div className="log-field-chips">
          {report.fields.map((field) => <code key={field}>{field}</code>)}
        </div>
      </section>

      <div className="log-evidence-notice">
        <Info size={15} />
        <span>
          Findings summarize observed patterns and anomalies in this file. They
          are not confirmed incidents, verified indicators of compromise, or
          proof that activity is malicious. Confirm with source telemetry and
          analyst review before responding.
        </span>
      </div>
    </div>
  );
}

function SummaryMetric({ label, value, severity }) {
  return (
    <div className={`log-summary-metric ${severity ? `metric-${severity}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EntityGroup({ label, values }) {
  return (
    <div className="log-entity-group">
      <strong>{label}</strong>
      <span>{entityList(values)}</span>
    </div>
  );
}

function EvidenceLabel({ text }) {
  return (
    <span className="log-evidence-label">
      <HardDrive size={12} />
      {text}
    </span>
  );
}
