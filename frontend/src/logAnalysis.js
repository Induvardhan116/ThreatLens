const MAX_EVENTS = 50_000;
const MAX_DISPLAYED_FINDINGS = 100;

const aliases = {
  timestamp: [
    "@timestamp",
    "timestamp",
    "time",
    "datetime",
    "date",
    "event_time",
    "created_at",
  ],
  sourceIp: [
    "source.ip",
    "src_ip",
    "source_ip",
    "client_ip",
    "remote_addr",
    "ip_address",
    "src",
  ],
  destinationIp: [
    "destination.ip",
    "dest_ip",
    "destination_ip",
    "server_ip",
    "dst",
  ],
  sourcePort: ["source.port", "src_port", "source_port"],
  destinationPort: [
    "destination.port",
    "dest_port",
    "destination_port",
    "dst_port",
    "port",
  ],
  username: ["user.name", "username", "user", "account", "principal"],
  host: ["host.name", "hostname", "host", "computer", "device_name"],
  eventType: ["event.action", "event_type", "action", "event", "type", "eventid"],
  message: ["message", "msg", "description", "command_line", "process.command_line"],
  status: ["status", "result", "outcome", "action_result"],
  bytes: ["network.bytes", "bytes", "bytes_out", "sent_bytes", "size"],
};

const techniques = {
  bruteForce: { id: "T1110", name: "Brute Force" },
  validAccounts: { id: "T1078", name: "Valid Accounts" },
  discovery: { id: "T1046", name: "Network Service Discovery" },
  command: { id: "T1059", name: "Command and Scripting Interpreter" },
  credentials: { id: "T1003", name: "OS Credential Dumping" },
  privilege: { id: "T1068", name: "Exploitation for Privilege Escalation" },
  exfiltration: { id: "T1041", name: "Exfiltration Over C2 Channel" },
};

const rules = [
  {
    key: "authentication-failures",
    title: "Repeated authentication failures",
    technique: techniques.bruteForce,
    severity: (count) =>
      count >= 25 ? "CRITICAL" : count >= 10 ? "HIGH" : "MEDIUM",
    recommendation:
      "Review the source address and targeted accounts, then verify MFA, lockout, and identity-provider telemetry.",
  },
  {
    key: "port-sweep",
    title: "Multi-port connection sweep",
    technique: techniques.discovery,
    severity: (count) => (count >= 20 ? "HIGH" : "MEDIUM"),
    recommendation:
      "Check whether the source is an authorized scanner and correlate the destinations with firewall and endpoint records.",
  },
  {
    key: "suspicious-command",
    title: "Command-line indicators require review",
    technique: techniques.command,
    severity: () => "MEDIUM",
    recommendation:
      "Review the matched process command, parent process, signer, host, and user before taking containment action.",
  },
  {
    key: "credential-access",
    title: "Credential-access indicators require review",
    technique: techniques.credentials,
    severity: () => "HIGH",
    recommendation:
      "Validate the process and endpoint context, inspect EDR alerts, and follow credential exposure procedures if corroborated.",
  },
  {
    key: "privilege-activity",
    title: "Privilege-change indicators require review",
    technique: techniques.privilege,
    severity: () => "MEDIUM",
    recommendation:
      "Confirm the change against approved administrative activity and inspect the initiating identity and host.",
  },
  {
    key: "high-volume-transfer",
    title: "High-volume outbound transfer",
    technique: techniques.exfiltration,
    severity: () => "HIGH",
    recommendation:
      "Validate units and direction for the source byte field, then review destination, business context, and egress controls.",
  },
];

function fieldKey(value) {
  return String(value).trim().toLowerCase().replace(/[\s-]+/g, "_");
}

function flattenRecord(record, prefix = "", output = {}) {
  if (!record || typeof record !== "object" || Array.isArray(record)) return output;
  for (const [key, value] of Object.entries(record)) {
    const name = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      flattenRecord(value, name, output);
    } else {
      output[fieldKey(name)] = Array.isArray(value) ? value.join(", ") : value;
      output[fieldKey(key)] ??= Array.isArray(value) ? value.join(", ") : value;
    }
  }
  return output;
}

function readField(record, names) {
  for (const name of names) {
    const value = record[fieldKey(name)];
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      return value;
    }
  }
  return "";
}

function normalizeEvent(record) {
  const fields = flattenRecord(record);
  const timestampValue = readField(fields, aliases.timestamp);
  const parsedTime = timestampValue ? Date.parse(timestampValue) : NaN;
  const message = String(readField(fields, aliases.message) || "");
  const eventType = String(readField(fields, aliases.eventType) || "");
  const status = String(readField(fields, aliases.status) || "");
  const combined = `${eventType} ${status} ${message}`.toLowerCase();
  const bytesValue = Number(String(readField(fields, aliases.bytes)).replace(/,/g, ""));

  return {
    timestamp: Number.isFinite(parsedTime) ? new Date(parsedTime).toISOString() : null,
    timeMs: Number.isFinite(parsedTime) ? parsedTime : null,
    sourceIp: String(readField(fields, aliases.sourceIp) || ""),
    destinationIp: String(readField(fields, aliases.destinationIp) || ""),
    sourcePort: Number(readField(fields, aliases.sourcePort)) || null,
    destinationPort: Number(readField(fields, aliases.destinationPort)) || null,
    username: String(readField(fields, aliases.username) || ""),
    host: String(readField(fields, aliases.host) || ""),
    eventType,
    message,
    status,
    bytes: Number.isFinite(bytesValue) && bytesValue >= 0 ? bytesValue : null,
    failedAuth: /fail(?:ed|ure)?|invalid password|authentication error|login denied/.test(combined),
    successfulAuth: /login success|successful login|authenticated successfully|authentication success/.test(combined),
    outbound: /outbound|upload|sent|egress|transmit/.test(combined),
    commandText: `${eventType} ${message}`.toLowerCase(),
  };
}

function parseCsv(text) {
  const delimiter = (text.split(/\r?\n/, 1)[0].match(/\t/g) || []).length >
    (text.split(/\r?\n/, 1)[0].match(/,/g) || []).length
    ? "\t"
    : ",";
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === delimiter && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += character;
    }
  }

  if (quoted) throw new Error("The CSV file contains an unterminated quoted field.");
  row.push(cell);
  if (row.some((value) => value.trim())) rows.push(row);
  if (rows.length < 2) throw new Error("The CSV file needs a header and at least one event row.");

  const headers = rows[0].map(fieldKey);
  const records = rows.slice(1).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]))
  );
  return { records, fields: rows[0].map((field) => String(field).trim()).filter(Boolean) };
}

function parseJson(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    const lines = text.split(/\r?\n/).filter((line) => line.trim());
    if (!lines.length) throw new Error("The file is empty.");
    try {
      parsed = lines.map((line) => JSON.parse(line));
    } catch {
      throw new Error("The JSON file is invalid. Use a JSON array, an object with an events array, or newline-delimited JSON.");
    }
  }

  const records = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed?.events)
      ? parsed.events
      : parsed && typeof parsed === "object"
        ? [parsed]
        : [];

  if (!records.length || records.some((record) => !record || typeof record !== "object" || Array.isArray(record))) {
    throw new Error("JSON logs must contain event objects.");
  }

  const fields = [...new Set(records.flatMap((record) => Object.keys(flattenRecord(record))))];
  return { records, fields };
}

function parseTextLog(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  const ipv4 = /(?:\d{1,3}\.){3}\d{1,3}/;
  const records = lines.map((line) => {
    const timestamp =
      line.match(/\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?/)?.[0] ||
      line.match(/\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b/)?.[0] ||
      "";
    const sourceIp = line.match(ipv4)?.[0] || "";
    const user = line.match(/\b(?:user|username|account)=["']?([^\s,"']+)/i)?.[1] || "";
    const host = line.match(/\b(?:host|hostname|computer)=["']?([^\s,"']+)/i)?.[1] || "";
    const port =
      Number(line.match(/\b(?:dst_port|dest_port|dport|port)=["']?(\d{1,5})/i)?.[1]) ||
      null;

    return {
      timestamp,
      source_ip: sourceIp,
      username: user,
      host,
      destination_port: port,
      message: line,
    };
  });

  if (!records.length) throw new Error("The text log contains no events.");
  return {
    records,
    fields: ["timestamp", "source_ip", "username", "host", "destination_port", "message"],
  };
}

function detectAndParse(text, fileName) {
  const trimmed = text.replace(/^\uFEFF/, "").trim();
  if (!trimmed) throw new Error("The selected file is empty.");

  const extension = fileName.split(".").pop()?.toLowerCase();
  if (!["csv", "json", "log", "txt"].includes(extension)) {
    throw new Error("Choose a supported .csv, .json, .log, or .txt file.");
  }

  if (extension === "csv") {
    const parsed = parseCsv(trimmed);
    return { ...parsed, format: "CSV" };
  }

  if (extension === "json") {
    const parsed = parseJson(trimmed);
    return { ...parsed, format: "JSON / NDJSON" };
  }

  if (/^\s*(?:\[|\{)/.test(trimmed)) {
    try {
      const parsed = parseJson(trimmed);
      return { ...parsed, format: "JSON / NDJSON (detected from content)" };
    } catch {
      if (extension === "json") throw new Error("The JSON content could not be parsed.");
    }
  }

  return { ...parseTextLog(trimmed), format: "Plain-text log" };
}

function capValues(values) {
  return [...new Set(values.filter(Boolean).map(String))].slice(0, 8);
}

function makeFinding(rule, events, detail, score) {
  const severity = rule.severity(events.length);
  const ips = capValues(events.flatMap((event) => [event.sourceIp, event.destinationIp]));
  const users = capValues(events.map((event) => event.username));
  const hosts = capValues(events.map((event) => event.host));
  const times = events.map((event) => event.timeMs).filter(Number.isFinite);

  return {
    id: rule.key,
    title: rule.title,
    severity,
    riskScore: score,
    eventCount: events.length,
    detail,
    evidence: "Observed log-event pattern",
    ips,
    users,
    hosts,
    firstSeen: times.length ? new Date(Math.min(...times)).toISOString() : null,
    lastSeen: times.length ? new Date(Math.max(...times)).toISOString() : null,
    technique: rule.technique,
    recommendation: rule.recommendation,
  };
}

function analyzeEvents(events) {
  const findings = [];
  const failedAuth = events.filter((event) => event.failedAuth);
  if (failedAuth.length >= 5) {
    const bySource = Map.groupBy
      ? Map.groupBy(failedAuth, (event) => event.sourceIp || "Source not recorded")
      : failedAuth.reduce((groups, event) => {
          const key = event.sourceIp || "Source not recorded";
          groups.set(key, [...(groups.get(key) || []), event]);
          return groups;
        }, new Map());

    for (const [source, grouped] of bySource) {
      if (grouped.length >= 5) {
        findings.push(
          makeFinding(
            rules[0],
            grouped,
            `${grouped.length} failed-authentication events were recorded for source ${source}. This may indicate repeated attempts; it is not proof of compromise.`,
            grouped.length >= 25 ? 88 : grouped.length >= 10 ? 72 : 55
          )
        );
      }
    }
  }

  const portsBySource = new Map();
  for (const event of events) {
    if (!event.sourceIp || !event.destinationPort) continue;
    const current = portsBySource.get(event.sourceIp) || new Map();
    current.set(event.destinationPort, [...(current.get(event.destinationPort) || []), event]);
    portsBySource.set(event.sourceIp, current);
  }
  for (const [source, ports] of portsBySource) {
    if (ports.size >= 8) {
      const grouped = [...ports.values()].flat();
      findings.push(
        makeFinding(
          rules[1],
          grouped,
          `Source ${source} contacted ${ports.size} distinct destination ports. Confirm whether this was authorized discovery activity.`,
          ports.size >= 20 ? 70 : 52
        )
      );
    }
  }

  const commandPattern =
    /\b(?:powershell(?:\.exe)?\s+-enc(?:odedcommand)?|curl\s+[^|]+\|\s*(?:sh|bash)|wget\s+[^|]+\|\s*(?:sh|bash)|certutil\s+-decode|base64\s+-d|invoke-expression|downloadstring)\b/i;
  const commandEvents = events.filter((event) => commandPattern.test(event.commandText));
  if (commandEvents.length) {
    findings.push(
      makeFinding(
        rules[2],
        commandEvents,
        `${commandEvents.length} event(s) contain command-line strings commonly worth review. Validate the complete process context.`,
        58
      )
    );
  }

  const credentialPattern = /\b(?:mimikatz|lsass(?:\.exe)?|sekurlsa|samdump|shadow copy|ntds\.dit)\b/i;
  const credentialEvents = events.filter((event) => credentialPattern.test(event.commandText));
  if (credentialEvents.length) {
    findings.push(
      makeFinding(
        rules[3],
        credentialEvents,
        `${credentialEvents.length} event(s) contain credential-access-related strings. These are indicators to investigate, not confirmation of credential theft.`,
        76
      )
    );
  }

  const privilegePattern = /\b(?:sudo\s+|privilege escalation|added to (?:the )?(?:administrators|sudoers)|useradd|net localgroup administrators)\b/i;
  const privilegeEvents = events.filter((event) => privilegePattern.test(event.commandText));
  if (privilegeEvents.length) {
    findings.push(
      makeFinding(
        rules[4],
        privilegeEvents,
        `${privilegeEvents.length} event(s) match privilege-change or elevated-command indicators. Validate against approved administrative work.`,
        54
      )
    );
  }

  const transferred = events.filter(
    (event) => event.outbound && event.bytes != null && event.bytes >= 100_000_000
  );
  if (transferred.length) {
    findings.push(
      makeFinding(
        rules[5],
        transferred,
        `${transferred.length} outbound-labelled event(s) report at least 100 MB in the detected byte field. Verify the field's units and direction.`,
        68
      )
    );
  }

  const eventThreshold = Math.max(100, Math.ceil(events.length * 0.1));
  const sourceCounts = new Map();
  for (const event of events) {
    if (event.sourceIp) sourceCounts.set(event.sourceIp, (sourceCounts.get(event.sourceIp) || 0) + 1);
  }
  for (const [source, count] of sourceCounts) {
    if (count >= eventThreshold && events.length >= 100) {
      findings.push({
        ...makeFinding(
          {
            title: "Concentrated source activity anomaly",
            severity: () => "LOW",
            technique: null,
            recommendation:
              "Compare this source's event volume with an appropriate time-based baseline and expected scanner or service activity.",
          },
          events.filter((event) => event.sourceIp === source),
          `${count} of ${events.length} supplied events (${Math.round((count / events.length) * 100)}%) reference source ${source}. This is a simple concentration check, not a learned baseline.`,
          28
        ),
        id: `source-volume-${source}`,
      });
    }
  }

  const severityOrder = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
  findings.sort(
    (left, right) =>
      severityOrder[left.severity] - severityOrder[right.severity] ||
      right.riskScore - left.riskScore
  );

  const severityCounts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
  for (const finding of findings) severityCounts[finding.severity] += 1;
  const riskScore = Math.min(
    100,
    findings.length
      ? Math.round(
          findings.reduce(
            (total, finding) =>
              total + finding.riskScore * Math.max(1, Math.min(finding.eventCount, 10)) / 10,
            0
          )
        )
      : 0
  );

  const timelineCounts = new Map();
  for (const event of events) {
    if (event.timeMs == null) continue;
    const hour = new Date(event.timeMs);
    hour.setMinutes(0, 0, 0);
    const key = hour.toISOString();
    timelineCounts.set(key, (timelineCounts.get(key) || 0) + 1);
  }

  const suspicious = {
    ips: capValues(findings.flatMap((finding) => finding.ips)),
    users: capValues(findings.flatMap((finding) => finding.users)),
    hosts: capValues(findings.flatMap((finding) => finding.hosts)),
  };

  return {
    totalEvents: events.length,
    severityCounts,
    findingCount: findings.length,
    findingsOmitted: Math.max(0, findings.length - MAX_DISPLAYED_FINDINGS),
    riskScore,
    findings: findings.slice(0, MAX_DISPLAYED_FINDINGS),
    suspicious,
    timeline: [...timelineCounts.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .slice(-24)
      .map(([timestamp, count]) => ({ timestamp, count })),
  };
}

export function analyzeLogFile(text, fileName) {
  const parsed = detectAndParse(text, fileName);
  if (parsed.records.length > MAX_EVENTS) {
    throw new Error(`The file contains more than ${MAX_EVENTS.toLocaleString()} events. Split it into smaller time windows and analyze them separately.`);
  }

  const events = parsed.records.map(normalizeEvent);
  const analysis = analyzeEvents(events);
  return {
    ...analysis,
    fileName,
    detectedFormat: parsed.format,
    fields: parsed.fields.slice(0, 40),
    eventsWithTimestamps: events.filter((event) => event.timestamp).length,
    analyzedAt: new Date().toISOString(),
  };
}
