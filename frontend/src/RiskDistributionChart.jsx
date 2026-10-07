import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const categoryColors = {
  CRITICAL: "#c77970",
  HIGH: "#bd8a6c",
  MEDIUM: "#b89b63",
  LOW: "#789a88",
  MINIMAL: "#718091",
};

export default function RiskDistributionChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data}>
        <CartesianGrid
          strokeDasharray="3 3"
          vertical={false}
          stroke="rgba(148,163,184,.12)"
        />
        <XAxis
          dataKey="shortName"
          tick={{ fill: "#a7b3bf", fontSize: 12 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: "#97a4b1", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,.03)" }}
          contentStyle={{
            background: "#111923",
            border: "1px solid rgba(255,255,255,.08)",
            borderRadius: "10px",
            color: "#e5edf7",
          }}
        />
        <Bar dataKey="value" radius={[5, 5, 0, 0]}>
          {data.map((entry) => (
            <Cell
              key={entry.name}
              fill={categoryColors[entry.name] || "#64748b"}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
