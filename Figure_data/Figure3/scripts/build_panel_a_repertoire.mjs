import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const figureDir = path.resolve(scriptDir, "..");
const dataPath = path.join(figureDir, "data", "panel_a_repertoire.csv");
const outDir = path.join(figureDir, "plots", "panel_a_repertoire_reproduced");
fs.mkdirSync(outDir, { recursive: true });

const svgPath = path.join(
  outDir,
  "panel_a_capsid_repertoire_landscape_log_axis.svg",
);

const W = 1200;
const H = 500;
const blue = "#49A3F5";
const blueDark = "#1D6EA5";
const gray = "#CFCFCF";
const grayStroke = "#8B8B8B";
const black = "#111111";
const lightGrid = "#DADADA";

const fmt = new Intl.NumberFormat("en-US");

function parseCsvLine(line) {
  const fields = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        field += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      fields.push(field);
      field = "";
    } else {
      field += char;
    }
  }
  fields.push(field);
  return fields;
}

function readCsv(filePath) {
  const lines = fs.readFileSync(filePath, "utf8").trim().split(/\r?\n/);
  const header = parseCsvLine(lines[0]);
  return lines
    .slice(1)
    .map((line) =>
      Object.fromEntries(
        parseCsvLine(line).map((value, index) => [header[index], value]),
      ),
    );
}

const rows = readCsv(dataPath);

function rowFor(metric, category) {
  const row = rows.find(
    (item) => item.metric === metric && item.category === category,
  );
  if (!row) {
    throw new Error(`Missing ${metric}/${category} in ${dataPath}`);
  }
  return row;
}

function numericValue(metric, category) {
  const value = Number(rowFor(metric, category).value);
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid numeric value for ${metric}/${category}`);
  }
  return value;
}

const seq = rows
  .filter(
    (row) =>
      row.metric === "capsid_sequence_repertoire" && row.plot_element === "bar",
  )
  .map((row) => ({
    label: row.category,
    value: Number(row.value),
    color: row.category === "This study" ? blue : gray,
    stroke: row.category === "This study" ? blueDark : grayStroke,
  }));

if (seq.length !== 3 || seq.some((row) => !Number.isFinite(row.value))) {
  throw new Error(
    `Expected three numeric capsid repertoire bars in ${dataPath}`,
  );
}

const foldVsUniProt = numericValue("fold_expansion", "This study vs UniProt");
const foldVsNcbi = numericValue("fold_expansion", "This study vs NCBI Virus");
const hmmPct = numericValue("hmm_recovery", "HMM-recovered");
const hmmMissedPct = numericValue("hmm_recovery", "HMM-missed");
const totalClusters = numericValue("predicted_capsid_aai50_clusters", "Total");

if (Math.abs(hmmPct + hmmMissedPct - 100) > 1e-6) {
  throw new Error("HMM-recovered and HMM-missed percentages must sum to 100");
}

function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function text(x, y, content, opts = {}) {
  const {
    size = 20,
    weight = 500,
    fill = black,
    anchor = "middle",
    rotate = null,
    style = "",
  } = opts;
  const tr = rotate ? ` transform="rotate(${rotate} ${x} ${y})"` : "";
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}" fill="${fill}"${tr} style="${style}">${esc(content)}</text>`;
}

function polar(cx, cy, r, angleDeg) {
  const a = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

function donutSlice(cx, cy, ro, ri, startAngle, endAngle) {
  const outerStart = polar(cx, cy, ro, endAngle);
  const outerEnd = polar(cx, cy, ro, startAngle);
  const innerStart = polar(cx, cy, ri, startAngle);
  const innerEnd = polar(cx, cy, ri, endAngle);
  const large = endAngle - startAngle > 180 ? 1 : 0;
  return [
    `M ${outerStart.x.toFixed(2)} ${outerStart.y.toFixed(2)}`,
    `A ${ro} ${ro} 0 ${large} 0 ${outerEnd.x.toFixed(2)} ${outerEnd.y.toFixed(2)}`,
    `L ${innerStart.x.toFixed(2)} ${innerStart.y.toFixed(2)}`,
    `A ${ri} ${ri} 0 ${large} 1 ${innerEnd.x.toFixed(2)} ${innerEnd.y.toFixed(2)}`,
    "Z",
  ].join(" ");
}

const parts = [];
parts.push(
  `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`,
);
parts.push(`<rect width="${W}" height="${H}" fill="#FFFFFF"/>`);
parts.push(`<style>
  text { font-family: Arial, Helvetica, sans-serif; dominant-baseline: middle; }
  .axis { stroke: #222222; stroke-width: 1.8; shape-rendering: crispEdges; }
  .tick { stroke: #222222; stroke-width: 1.2; shape-rendering: crispEdges; }
  .grid { stroke: ${lightGrid}; stroke-width: 1.1; shape-rendering: crispEdges; }
</style>`);

parts.push(text(30, 32, "A", { size: 36, weight: 700, anchor: "start" }));

const chart = { x: 112, y: 78, w: 520, h: 320 };
const yMin = 10000;
const yMax = 3000000;
const logMin = Math.log10(yMin);
const logMax = Math.log10(yMax);
const yFor = (v) =>
  chart.y + chart.h - ((Math.log10(v) - logMin) / (logMax - logMin)) * chart.h;
const baseY = yFor(yMin);

parts.push(
  `<line class="axis" x1="${chart.x}" y1="${chart.y}" x2="${chart.x}" y2="${chart.y + chart.h}"/>`,
);
parts.push(
  `<line class="axis" x1="${chart.x}" y1="${chart.y + chart.h}" x2="${chart.x + chart.w}" y2="${chart.y + chart.h}"/>`,
);

for (const t of [10000, 100000, 1000000, 3000000]) {
  const y = yFor(t);
  if (t > yMin)
    parts.push(
      `<line class="grid" x1="${chart.x}" y1="${y.toFixed(2)}" x2="${chart.x + chart.w}" y2="${y.toFixed(2)}"/>`,
    );
  parts.push(
    `<line class="tick" x1="${chart.x - 7}" y1="${y.toFixed(2)}" x2="${chart.x}" y2="${y.toFixed(2)}"/>`,
  );
  parts.push(text(chart.x - 14, y, fmt.format(t), { size: 16, anchor: "end" }));
}
parts.push(
  text(28, chart.y + chart.h / 2, "Capsid protein sequences (log10)", {
    size: 18,
    anchor: "middle",
    rotate: -90,
  }),
);

const barW = 66;
const xs = [chart.x + 118, chart.x + 268, chart.x + 420];
seq.forEach((d, i) => {
  const topY = yFor(d.value);
  const h = baseY - topY;
  parts.push(
    `<rect x="${(xs[i] - barW / 2).toFixed(2)}" y="${topY.toFixed(2)}" width="${barW}" height="${h.toFixed(2)}" fill="${d.color}" stroke="${d.stroke}" stroke-width="1.3"/>`,
  );
  parts.push(
    text(xs[i], chart.y + chart.h + 30, d.label, { size: 18, weight: 700 }),
  );
  parts.push(
    text(xs[i], topY - 20, fmt.format(d.value), { size: 16, weight: 600 }),
  );
});

parts.push(
  text(chart.x + 225, chart.y + 84, `${foldVsUniProt.toFixed(2)}x vs UniProt`, {
    size: 19,
    weight: 700,
    fill: blue,
  }),
);
parts.push(
  text(
    chart.x + 225,
    chart.y + 124,
    `${foldVsNcbi.toFixed(2)}x vs NCBI Virus`,
    { size: 19, weight: 700, fill: blue },
  ),
);

const cx = 850;
const cy = 238;
const ro = 112;
const ri = 55;
const hmmAngle = (hmmPct / 100) * 360;
parts.push(
  `<path d="${donutSlice(cx, cy, ro, ri, 0, hmmAngle)}" fill="${blue}" stroke="#FFFFFF" stroke-width="3"/>`,
);
parts.push(
  `<path d="${donutSlice(cx, cy, ro, ri, hmmAngle, 360)}" fill="${gray}" stroke="#FFFFFF" stroke-width="3"/>`,
);

const hmmLabel = polar(cx, cy, ro * 0.72, hmmAngle / 2);
const missLabel = polar(cx, cy, ro * 0.72, hmmAngle + (360 - hmmAngle) / 2);
parts.push(
  text(hmmLabel.x, hmmLabel.y, `${hmmPct.toFixed(2)}%`, {
    size: 20,
    weight: 700,
    fill: "#222222",
  }),
);
parts.push(
  text(missLabel.x, missLabel.y, `${hmmMissedPct.toFixed(2)}%`, {
    size: 20,
    weight: 700,
    fill: "#333333",
  }),
);

const lx = 1010;
const ly = 193;
parts.push(`<rect x="${lx}" y="${ly}" width="17" height="17" fill="${blue}"/>`);
parts.push(
  text(lx + 27, ly + 8.5, "HMM-recovered", {
    size: 17,
    weight: 700,
    anchor: "start",
  }),
);
parts.push(
  `<rect x="${lx}" y="${ly + 32}" width="17" height="17" fill="${gray}"/>`,
);
parts.push(
  text(lx + 27, ly + 40.5, "HMM-missed", {
    size: 17,
    weight: 700,
    anchor: "start",
  }),
);
parts.push(
  text(
    cx,
    cy + ro + 38,
    `${fmt.format(totalClusters)} predicted capsid AAI50 clusters`,
    { size: 18, weight: 600 },
  ),
);

parts.push("</svg>");

fs.writeFileSync(svgPath, parts.join("\n"));
console.log(`input: ${dataPath}`);
console.log(svgPath);
