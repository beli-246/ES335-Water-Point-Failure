/* Site logic. Every number comes from data/*.json, written by src/export_site.py. Charts are hand-rolled SVG. */
"use strict";

const COUNTRIES = ["Ethiopia", "Kenya", "Tanzania", "Uganda"];
const MODEL_ORDER = ["majority", "stump", "logreg", "tree", "rf", "xgb", "svm_linear", "svm_rbf"];
const MODEL_NAME = { majority: "Majority class", stump: "One-rule stump", logreg: "Logistic regression", tree: "Decision tree",
  rf: "Random forest", xgb: "XGBoost", svm_linear: "Linear SVM", svm_rbf: "RBF SVM", stack: "Stacked ensemble" };
const REGIME = { A: "A · pooled (foreign + local)", B: "B · foreign only", C: "C · local only" };
const REGIME_COLOR = { A: "--s1", B: "--s2", C: "--s3" };
const FEATURE_NAME = { water_source_clean: "Water source", water_tech_clean: "Technology", management_clean: "Management",
  local_population: "Local population", distance_to_primary: "Distance to primary road", distance_to_secondary: "Distance to secondary road",
  distance_to_tertiary: "Distance to tertiary road", distance_to_city: "Distance to city", distance_to_town: "Distance to town",
  age: "Age (years)", is_urban: "Urban flag" };
const VARIANT_NAME = { main: "Main", no_missing: "No missing signal", no_remoteness: "No remoteness / population", with_pay: "+ fee payment",
  uganda_sub: "Uganda cut to 21,000 rows" };

const $ = (s, r = document) => r.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const f3 = (v) => (v == null || isNaN(v) ? "–" : (+v).toFixed(3));
const f2 = (v) => (v == null || isNaN(v) ? "–" : (+v).toFixed(2));
const pct = (v, d = 0) => (v == null || isNaN(v) ? "–" : (100 * v).toFixed(d) + "%");
const fmtInt = (v) => Math.round(v).toLocaleString("en-US");
const prettyFeat = (f) => f.replace(/^(cat|log|plain)__/, "").replace("water_source_clean_", "Source: ").replace("water_tech_clean_", "Tech: ")
  .replace("management_clean_", "Management: ").replace("missingindicator_", "missing: ").replace("_sklearn", "");
const skill = (ap, prev) => (ap - prev) / (1 - prev); // 0 = random, 1 = perfect; comparable across base rates
const signed = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const candName = (c) => { if (c === "stack") return MODEL_NAME.stack; const [m, r] = c.split("__"); return `${MODEL_NAME[m] || m} (${r})`; };

function el(tag, attrs = {}, parent) {
  const svgTags = ["svg", "g", "line", "rect", "circle", "path", "text", "polyline"];
  const e = svgTags.includes(tag) ? document.createElementNS("http://www.w3.org/2000/svg", tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v; else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (v != null) e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}

/* ---------- tooltip ---------- */
const tip = el("div", { class: "tip", role: "status" }, document.body);
function showTip(evt, title, rows) {
  tip.replaceChildren();
  if (title) el("div", { class: "t", text: title }, tip);
  for (const r of rows) {
    const row = el("div", { class: "row" }, tip);
    if (r.color) el("span", { class: "k", style: { background: r.color } }, row);
    el("b", { text: r.value }, row);
    el("span", { text: r.label }, row);
  }
  tip.style.display = "block";
  const x = evt.clientX ?? evt.target.getBoundingClientRect().right, y = evt.clientY ?? evt.target.getBoundingClientRect().top;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + "px";
  tip.style.top = Math.max(8, Math.min(window.innerHeight - h - 8, y - h - 10)) + "px";
}
const hideTip = () => (tip.style.display = "none");

/* ---------- scales & axes ---------- */
function scale(d0, d1, r0, r1, log = false) {
  const t = log ? Math.log10 : (v) => v;
  const a = t(d0), b = t(d1);
  const s = (v) => r0 + ((t(v) - a) / (b - a || 1)) * (r1 - r0);
  s.domain = [d0, d1];
  return s;
}
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function logTicks(lo, hi) { const out = []; for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) { const v = 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); } return out; }
function frame(container, { h = 260, ml = 48, mr = 16, mt = 12, mb = 40 } = {}) {
  container.replaceChildren();
  const W = Math.max(300, container.clientWidth || 600);
  const svg = el("svg", { viewBox: `0 0 ${W} ${h}`, role: "img" }, container);
  return { svg, W, H: h, x0: ml, x1: W - mr, y0: h - mb, y1: mt };
}
function yAxis(f, ys, ticks, fmt, title) {
  const g = el("g", { class: "grid" }, f.svg);
  for (const t of ticks) {
    el("line", { x1: f.x0, x2: f.x1, y1: ys(t), y2: ys(t) }, g);
    el("text", { x: f.x0 - 6, y: ys(t) + 4, "text-anchor": "end", text: fmt(t) }, f.svg);
  }
  if (title) el("text", { class: "axis-title", x: 12, y: (f.y0 + f.y1) / 2, transform: `rotate(-90 12 ${(f.y0 + f.y1) / 2})`, "text-anchor": "middle", text: title }, f.svg);
}
function xAxis(f, xs, ticks, fmt, title) {
  el("line", { class: "baseline", x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0 }, f.svg);
  for (const t of ticks) el("text", { x: xs(t), y: f.y0 + 16, "text-anchor": "middle", text: fmt(t) }, f.svg);
  if (title) el("text", { class: "axis-title", x: (f.x0 + f.x1) / 2, y: f.H - 6, "text-anchor": "middle", text: title }, f.svg);
}
function legend(container, items) {
  const L = el("div", { class: "legend" }, container);
  for (const it of items) {
    const s = el("span", {}, L);
    el("i", { class: it.kind || "", style: { background: it.color, borderColor: it.color } }, s);
    s.appendChild(document.createTextNode(it.label));
  }
  return L;
}

/* ---------- line chart with crosshair (all series in one tooltip) ---------- */
function lineChart(container, o) {
  const f = frame(container, { h: o.h || 260, ml: o.ml || 52 });
  const all = o.series.flatMap((s) => s.pts);
  const xlo = o.xmin ?? Math.min(...all.map((p) => p.x)), xhi = o.xmax ?? Math.max(...all.map((p) => p.x));
  let ylo = o.ymin ?? Math.min(...all.map((p) => p.y)), yhi = o.ymax ?? Math.max(...all.map((p) => p.y));
  if (o.ymin == null) ylo -= (yhi - ylo) * 0.05;
  if (o.ymax == null) yhi += (yhi - ylo) * 0.05;
  const xs = scale(xlo, xhi, f.x0, f.x1, o.xlog), ys = scale(ylo, yhi, f.y0, f.y1, o.ylog);
  yAxis(f, ys, o.ylog ? logTicks(ylo, yhi) : niceTicks(ylo, yhi, 4), o.yfmt || f2, o.ylabel);
  const xt = o.xticks || (o.xlog ? logTicks(xlo, xhi) : niceTicks(xlo, xhi, 6));
  xAxis(f, xs, xt, o.xfmt || ((v) => v), o.xlabel);
  for (const r of o.refs || []) {
    if (r.y != null) el("line", { x1: f.x0, x2: f.x1, y1: ys(r.y), y2: ys(r.y), stroke: css("--neutral"), "stroke-width": 1, "stroke-dasharray": "4 4" }, f.svg);
    if (r.diag) el("line", { x1: xs(xlo), y1: ys(xlo), x2: xs(Math.min(xhi, yhi)), y2: ys(Math.min(xhi, yhi)), stroke: css("--neutral"), "stroke-width": 1, "stroke-dasharray": "4 4" }, f.svg);
    if (r.label && r.y != null) el("text", { x: f.x1, y: ys(r.y) - 5, "text-anchor": "end", text: r.label }, f.svg);
  }
  for (const s of o.series) {
    const pts = s.pts.filter((p) => p.y != null && isFinite(p.y));
    el("polyline", { points: pts.map((p) => `${xs(p.x)},${ys(p.y)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": s.width || 2,
      "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash ? "5 4" : null, opacity: s.opacity ?? 1 }, f.svg);
    if (s.markers) for (const p of pts) el("circle", { cx: xs(p.x), cy: ys(p.y), r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, f.svg);
  }
  const api = { xs, ys, f };
  if (o.notip) return api;
  const xsAll = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b);
  const cross = el("line", { y1: f.y1, y2: f.y0, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" }, f.svg);
  const hit = el("rect", { x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, fill: "transparent" }, f.svg);
  hit.addEventListener("pointermove", (e) => {
    const r = f.svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * f.W;
    let best = xsAll[0];
    for (const x of xsAll) if (Math.abs(xs(x) - px) < Math.abs(xs(best) - px)) best = x;
    cross.setAttribute("x1", xs(best)); cross.setAttribute("x2", xs(best)); cross.setAttribute("visibility", "visible");
    const rows = o.series.filter((s) => !s.notip).map((s) => {
      const p = s.pts.reduce((a, b) => (Math.abs(b.x - best) < Math.abs(a.x - best) ? b : a));
      return { color: s.color, value: (o.tipfmt || o.yfmt || f3)(p.y), label: s.name };
    });
    showTip(e, (o.xtipfmt || o.xfmt || ((v) => v))(best), rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  return api;
}

/* ---------- dot plot: rows x series, optional CI whiskers ---------- */
function dotPlot(container, o) {
  const rowH = o.rowH || 30, h = o.rows.length * rowH + 52;
  const f = frame(container, { h: h + 12, ml: o.ml || 150, mb: 40, mt: 20 });
  const vals = o.rows.flatMap((r) => o.series.flatMap((s) => [r[s.key], r[s.key + "_lo"], r[s.key + "_hi"]])).filter((v) => v != null && isFinite(v));
  o.refs = (o.refs || []).filter((r) => r.x != null && isFinite(r.x));
  if (!vals.length) { container.replaceChildren(); container.textContent = "No results yet."; return; }
  let lo = Math.min(...vals, ...o.refs.map((r) => r.x)), hi = Math.max(...vals, ...o.refs.map((r) => r.x));
  if (o.xmin != null) lo = o.xmin;
  const pad = (hi - lo) * 0.06; lo = o.xmin ?? Math.max(0, lo - pad); hi += pad;
  if (o.xmin != null && lo > Math.min(...vals)) lo = Math.min(...vals) - pad;
  const xs = scale(lo, hi, f.x0, f.x1);
  const g = el("g", { class: "grid" }, f.svg);
  for (const t of niceTicks(lo, hi, 5)) { el("line", { x1: xs(t), x2: xs(t), y1: f.y1, y2: f.y0 }, g); el("text", { x: xs(t), y: f.y0 + 16, "text-anchor": "middle", text: (o.xfmt || f2)(t) }, f.svg); }
  el("text", { class: "axis-title", x: (f.x0 + f.x1) / 2, y: f.H - 6, "text-anchor": "middle", text: o.xlabel }, f.svg);
  for (const r of o.refs || []) {
    el("line", { x1: xs(r.x), x2: xs(r.x), y1: f.y1, y2: f.y0, stroke: css("--neutral"), "stroke-dasharray": "4 4" }, f.svg);
    if (r.label) el("text", { x: xs(r.x), y: f.y1 - 6, "text-anchor": "middle", text: r.label }, f.svg);
  }
  const n = o.series.length;
  o.rows.forEach((r, i) => {
    const yc = f.y1 + i * rowH + rowH / 2;
    el("text", { x: f.x0 - 8, y: yc + 4, "text-anchor": "end", class: "axis-title", text: r.label }, f.svg);
    o.series.forEach((s, j) => {
      const v = r[s.key]; if (v == null || !isFinite(v)) return;
      const y = yc + (j - (n - 1) / 2) * Math.min(8, rowH / (n + 1));
      if (r[s.key + "_lo"] != null) el("line", { x1: xs(r[s.key + "_lo"]), x2: xs(r[s.key + "_hi"]), y1: y, y2: y, stroke: s.color, "stroke-width": 2, opacity: 0.45, "stroke-linecap": "round" }, f.svg);
      el("circle", { cx: xs(v), cy: y, r: 5, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, f.svg);
      const hit = el("circle", { cx: xs(v), cy: y, r: 12, fill: "transparent", tabindex: 0 }, f.svg);
      const show = (e) => showTip(e, `${r.label} · ${s.name}`, [{ color: s.color, value: (o.tipfmt || f3)(v), label: r[s.key + "_lo"] != null ? `95% CI ${(o.tipfmt || f3)(r[s.key + "_lo"])}–${(o.tipfmt || f3)(r[s.key + "_hi"])}` : "" },
        ...(o.extraTip ? o.extraTip(r, s) : [])]);
      hit.addEventListener("pointermove", show); hit.addEventListener("focus", show);
      hit.addEventListener("pointerleave", hideTip); hit.addEventListener("blur", hideTip);
    });
  });
}

/* ---------- horizontal bars ---------- */
function hbars(container, o) {
  const rowH = 26, f = frame(container, { h: o.rows.length * rowH + 60, ml: o.ml || 180, mt: 20 });
  const vals = o.rows.flatMap((r) => [r.value, r.value + (r.err || 0)]);
  const lo = Math.min(0, ...vals), hi = Math.max(...vals, ...(o.refs || []).map((r) => r.x)) * 1.08;
  const xs = scale(lo, hi, f.x0, f.x1);
  const g = el("g", { class: "grid" }, f.svg);
  for (const t of niceTicks(lo, hi, 5)) { el("line", { x1: xs(t), x2: xs(t), y1: f.y1, y2: f.y0 }, g); el("text", { x: xs(t), y: f.y0 + 16, "text-anchor": "middle", text: (o.xfmt || f2)(t) }, f.svg); }
  el("text", { class: "axis-title", x: (f.x0 + f.x1) / 2, y: f.H - 6, "text-anchor": "middle", text: o.xlabel }, f.svg);
  o.rows.forEach((r, i) => {
    const y = f.y1 + i * rowH + 5, bh = Math.min(16, rowH - 8), x0 = xs(Math.min(0, r.value)), x1 = xs(Math.max(0, r.value));
    el("text", { x: f.x0 - 8, y: y + bh / 2 + 4, "text-anchor": "end", class: "axis-title", text: r.label }, f.svg);
    const w = Math.max(1, x1 - x0), rr = Math.min(4, w / 2);
    const right = r.value >= 0;
    const d = right ? `M${x0},${y} h${w - rr} q${rr},0 ${rr},${rr} v${bh - 2 * rr} q0,${rr} -${rr},${rr} h-${w - rr} z`
      : `M${x1},${y} h-${w - rr} q-${rr},0 -${rr},${rr} v${bh - 2 * rr} q0,${rr} ${rr},${rr} h${w - rr} z`;
    el("path", { d, fill: r.color || css("--s1") }, f.svg);
    if (r.err) el("line", { x1: xs(r.value - r.err), x2: xs(r.value + r.err), y1: y + bh / 2, y2: y + bh / 2, stroke: css("--ink-2"), "stroke-width": 1 }, f.svg);
    el("text", { x: Math.max(x1, r.err ? xs(r.value + r.err) : x1, xs(0)) + 6, y: y + bh / 2 + 4, "text-anchor": "start", text: (o.vfmt || f3)(r.value) }, f.svg);
    const hit = el("rect", { x: f.x0, y: y - 4, width: f.x1 - f.x0, height: rowH, fill: "transparent" }, f.svg);
    hit.addEventListener("pointermove", (e) => showTip(e, r.label, [{ color: r.color || css("--s1"), value: (o.vfmt || f3)(r.value), label: r.tip || "" }]));
    hit.addEventListener("pointerleave", hideTip);
  });
  el("line", { class: "baseline", x1: xs(0), x2: xs(0), y1: f.y1, y2: f.y0 }, f.svg);
  for (const r of o.refs || []) { el("line", { x1: xs(r.x), x2: xs(r.x), y1: f.y1, y2: f.y0, stroke: css("--neutral"), "stroke-dasharray": "4 4" }, f.svg); el("text", { x: xs(r.x), y: f.y1 - 6, "text-anchor": "middle", text: r.label }, f.svg); }
}

/* ---------- heatmap (sequential, one hue) ---------- */
function mix(a, b, t) {
  const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const A = p(a), B = p(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`;
}
function heatmap(container, o) {
  const f = frame(container, { h: o.rows.length * 40 + 56, ml: 70, mt: 8, mb: 44 });
  const cw = (f.x1 - f.x0) / o.cols.length, ch = (f.y0 - f.y1) / o.rows.length;
  const flat = o.values.flat(), lo = Math.min(...flat), hi = Math.max(...flat);
  const best = Math.max(...flat);
  o.rows.forEach((r, i) => o.cols.forEach((c, j) => {
    const v = o.values[i][j], t = (v - lo) / (hi - lo || 1);
    const x = f.x0 + j * cw, y = f.y0 - (i + 1) * ch;
    const fill = mix(css("--seq-lo"), css("--seq-hi"), t);
    const cell = el("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 4, fill, tabindex: 0 }, f.svg);
    el("text", { x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", style: { fill: t > 0.55 ? "#ffffff" : "#0b0b0b", fontWeight: v === best ? 700 : 400 }, text: (o.fmt || f3)(v) }, f.svg);
    const show = (e) => showTip(e, `${o.ylabel} ${r} · ${o.xlabel} ${c}`, [{ value: (o.fmt || f3)(v), label: o.vlabel }]);
    cell.addEventListener("pointermove", show); cell.addEventListener("focus", show);
    cell.addEventListener("pointerleave", hideTip); cell.addEventListener("blur", hideTip);
  }));
  o.cols.forEach((c, j) => el("text", { x: f.x0 + (j + 0.5) * cw, y: f.y0 + 16, "text-anchor": "middle", text: c }, f.svg));
  o.rows.forEach((r, i) => el("text", { x: f.x0 - 8, y: f.y0 - (i + 0.5) * ch + 4, "text-anchor": "end", text: r }, f.svg));
  el("text", { class: "axis-title", x: (f.x0 + f.x1) / 2, y: f.H - 6, "text-anchor": "middle", text: o.xlabel }, f.svg);
  el("text", { class: "axis-title", x: 12, y: (f.y0 + f.y1) / 2, transform: `rotate(-90 12 ${(f.y0 + f.y1) / 2})`, "text-anchor": "middle", text: o.ylabel }, f.svg);
}

/* ---------- small helpers ---------- */
function segmented(container, options, value, onChange) {
  const s = el("div", { class: "seg", role: "group" }, container);
  const btns = options.map(([v, label]) => {
    const b = el("button", { type: "button", "aria-pressed": v === value ? "true" : "false", text: label }, s);
    b.addEventListener("click", () => { btns.forEach((x) => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); onChange(v); });
    return b;
  });
  return s;
}
function table(container, cols, rows, rowClass) {
  const wrap = el("div", { class: "tablewrap" }, container), t = el("table", {}, wrap);
  const hr = el("tr", {}, el("thead", {}, t));
  for (const c of cols) el("th", { class: c.num ? "num" : "", text: c.label }, hr);
  const tb = el("tbody", {}, t);
  for (const r of rows) {
    const tr = el("tr", { class: rowClass ? rowClass(r) : "" }, tb);
    for (const c of cols) { const v = c.get(r); const td = el("td", { class: (c.num ? "num " : "") + (c.cls ? c.cls(r) : ""), text: v }, tr); }
  }
  return wrap;
}
const state = { country: "Uganda" };
let R, C, O;
const points = {};
async function loadPoints(c) {
  if (!points[c]) points[c] = await fetch(`data/points_${c}.json`).then((r) => (r.ok ? r.json() : null));
  return points[c];
}
const metricRows = (variant) => R.metrics.filter((m) => m.variant === variant);

/* ================= sections ================= */
function renderHero() {
  const box = $("#hero-tiles"); if (!box) return;
  box.replaceChildren();
  const ch = R.final_choice || {};
  const tiles = [];
  for (const c of COUNTRIES) {
    const t = R.final_test.find((r) => r.target === c && r.chosen);
    if (!t || !ch[c]) continue;
    tiles.push([c, pct(t["prec@10%"]), `vs ${pct(t.prevalence)} of all test points · ${candName(ch[c].chosen)}`]);
  }
  for (const [label, value, sub] of tiles) {
    const d = el("div", { class: "tile" }, box);
    el("div", { class: "label", text: `${label}: no water among the top 10% flagged` }, d); el("div", { class: "value", text: value }, d); el("div", { class: "sub", text: sub }, d);
  }
}

function renderDataTable() {
  const box = $("#data-table"); if (!box || !O) return;
  const pc = O.cleaning.per_country, sp = O.splits;
  table(box, [
    { label: "Country", get: (c) => c },
    { label: "Water points", num: true, get: (c) => fmtInt(pc[c].rows) },
    { label: "No water", num: true, get: (c) => pct(pc[c].no_water, 1) },
    { label: "Districts", num: true, get: (c) => pc[c].districts },
    { label: "Test districts", num: true, get: (c) => `${sp[c].n_test_districts} (${pct(sp[c].test_share, 1)} of rows)` },
    { label: "Test no-water rate", num: true, get: (c) => pct(sp[c].test_no_water, 1) },
  ], COUNTRIES);
  const cl = O.cleaning;
  $("#cleaning-steps").textContent = `${fmtInt(cl.raw_rows)} raw rows → ${fmtInt(cl.unknown_status_dropped)} with Unknown status dropped → ${fmtInt(cl.duplicates_dropped)} WPdx-flagged duplicates dropped → ${fmtInt(cl.install_year_set_missing)} impossible install years set to missing → ${fmtInt(cl.final_rows)} rows.`;
}

function regimeRows(c, variant = "main") {
  const ms = metricRows(variant).filter((m) => m.target === c);
  const prev = ms[0]?.prevalence;
  return { prev, rows: MODEL_ORDER.filter((m) => m !== "majority" && ms.some((x) => x.model === m)).map((m) => {
    const row = { label: MODEL_NAME[m], model: m };
    for (const r of ["A", "B", "C"]) {
      const x = ms.find((k) => k.model === m && k.regime === r);
      if (x) row[r] = x.pr_auc;
      // the regime's own 95% CI: from any paired comparison it takes part in (same resamples, so all agree)
      const gs = variant === "main" ? R.gaps.filter((k) => k.target === c && k.model === m && k.metric === "pr_auc") : [];
      const asA = gs.find((k) => k.comparison.startsWith(r + "-")), asB = gs.find((k) => k.comparison.endsWith("-" + r));
      if (asA) { row[r + "_lo"] = asA.a_lo; row[r + "_hi"] = asA.a_hi; } else if (asB) { row[r + "_lo"] = asB.b_lo; row[r + "_hi"] = asB.b_hi; }
    }
    return row;
  }) };
}

function renderRegimes() {
  const box = $("#regime-chart"); if (!box) return;
  const ctr = $("#regime-controls"); ctr.replaceChildren();
  segmented(ctr, COUNTRIES.map((c) => [c, c]), state.country, (c) => { state.country = c; renderRegimes(); renderGapTable(); });
  const { prev, rows } = regimeRows(state.country);
  const series = ["A", "B", "C"].filter((r) => rows.some((x) => x[r] != null)).map((r) => ({ key: r, name: REGIME[r], color: css(REGIME_COLOR[r]) }));
  box.replaceChildren();
  const leg = el("div", {}, box); legend(leg, [...series.map((s) => ({ label: s.name, color: s.color })), { label: `random ranking = no-water rate (${pct(prev, 1)})`, color: css("--neutral"), kind: "dash" }]);
  const ch = el("div", { class: "chart" }, box);
  dotPlot(ch, { rows, series, refs: [{ x: prev, label: "random" }], xlabel: "Test PR-AUC · whiskers: 95% CI",
    extraTip: (r) => [{ value: f2(r.A / prev) + "× / " + f2(r.B / prev) + "× / " + f2(r.C / prev) + "×", label: "lift A / B / C" }] });
  renderGapTable();
}

function renderGapTable() {
  const box = $("#gap-table"); if (!box) return; box.replaceChildren();
  const rows = R.gaps.filter((g) => g.target === state.country && g.metric === "pr_auc");
  if (!rows.length) { box.textContent = "Bootstrap results not exported yet."; return; }
  const label = { "A-B": "A − B (add local data to foreign)", "C-B": "C − B (local only vs foreign only)", "A-C": "A − C (does pooling help local?)" };
  rows.sort((a, b) => MODEL_ORDER.indexOf(a.model) - MODEL_ORDER.indexOf(b.model) || a.comparison.localeCompare(b.comparison));
  table(box, [
    { label: "Model", get: (r) => MODEL_NAME[r.model] },
    { label: "Comparison", get: (r) => label[r.comparison] || r.comparison },
    { label: "Difference in PR-AUC", num: true, get: (r) => signed(r.diff), cls: (r) => (r.diff_lo > 0 ? "pos" : r.diff_hi < 0 ? "negv" : "") },
    { label: "95% CI", num: true, get: (r) => `${signed(r.diff_lo)} to ${signed(r.diff_hi)}` },
    { label: "Districts", num: true, get: (r) => r.n_districts },
  ], rows);
}

/* pooled models that know the country (added 15 Nov): A vs A+country vs C, same test districts */
const COUNTRY_ROWS = [["logreg", "country", "Logistic regression, country intercept only", "LogReg, intercept only"],
  ["logreg", "country_x", "Logistic regression, country × features", "LogReg, country × features"],
  ["tree", "country", "Decision tree", "Tree"], ["rf", "country", "Random forest", "Random forest"], ["xgb", "country", "XGBoost", "XGBoost"]];
function renderCountryModels() {
  const box = $("#country-chart"); if (!box || !R.country?.length) return;
  const ctr = $("#country-controls"); ctr.replaceChildren();
  segmented(ctr, COUNTRIES.map((c) => [c, c]), state.country, (c) => { state.country = c; renderCountryModels(); });
  const cmp = (m, v, other) => R.country.find((r) => r.target === state.country && r.model === m && r.variant === v && r.comparison === `A+country - ${other}`);
  const narrow = (box.clientWidth || 600) < 560;
  const rows = COUNTRY_ROWS.map(([m, v, long, short]) => {
    const vsC = cmp(m, v, "C"), vsA = cmp(m, v, "A"); if (!vsC || !vsA) return null;
    return { label: narrow ? short : long, long, vsC, vsA, A: vsA.b, A_lo: vsA.b_lo, A_hi: vsA.b_hi, X: vsC.a, X_lo: vsC.a_lo, X_hi: vsC.a_hi, C: vsC.b, C_lo: vsC.b_lo, C_hi: vsC.b_hi };
  }).filter(Boolean);
  box.replaceChildren();
  const prev = rows[0]?.vsC.prevalence;
  const series = [{ key: "A", name: "A · pooled, country unknown", color: css("--s1") }, { key: "X", name: "A + country · pooled, country known", color: css("--s4") },
    { key: "C", name: REGIME.C, color: css("--s3") }];
  legend(box, [...series.map((s) => ({ label: s.name, color: s.color })), { label: `random = no-water rate (${pct(prev, 1)})`, color: css("--neutral"), kind: "dash" }]);
  dotPlot(el("div", { class: "chart" }, box), { rows, series, refs: [{ x: prev, label: "random" }], xlabel: narrow ? "Test PR-AUC · 95% CI" : "Test PR-AUC · whiskers: 95% CI", ml: narrow ? 150 : 230 });
  const tb = $("#country-table"); tb.replaceChildren();
  const d = (r) => `${signed(r.diff)} (${signed(r.diff_lo)} to ${signed(r.diff_hi)})`;
  const sig = (r) => (r.diff_lo > 0 ? "pos" : r.diff_hi < 0 ? "negv" : "");
  table(tb, [
    { label: "Model", get: (r) => r.long },
    { label: "Knowing the country: A+country − A", num: true, get: (r) => d(r.vsA), cls: (r) => sig(r.vsA) },
    { label: "Pooled vs local: A+country − C", num: true, get: (r) => d(r.vsC), cls: (r) => sig(r.vsC) },
  ], rows);
}

function renderHeadline() {
  const box = $("#headline-chart"); if (!box) return; box.replaceChildren();
  const rows = COUNTRIES.filter((c) => R.final_choice?.[c]).map((c) => {
    const ch = R.final_choice[c], t = R.final_test.find((r) => r.target === c && r.chosen), p = t.prevalence;
    const vb = ch.vs_best_B, vs = ch.vs_local_stump;
    const k = (v) => skill(v, p);
    return { label: c, F: k(vb.a), F_lo: k(vb.a_lo), F_hi: k(vb.a_hi), B: k(vb.b), B_lo: k(vb.b_lo), B_hi: k(vb.b_hi), S: k(vs.b), S_lo: k(vs.b_lo), S_hi: k(vs.b_hi), prev: p,
      chosen: candName(ch.chosen), bname: MODEL_NAME[ch.best_B_model] };
  });
  if (!rows.length) { box.textContent = "Final-model results not exported yet."; return; }
  const series = [{ key: "F", name: "Final model (chosen by CV, uses local data)", color: css("--s1") }, { key: "B", name: "Best foreign-only model (B)", color: css("--s2") },
    { key: "S", name: "One-rule stump (local)", color: css("--s3") }];
  legend(box, [...series.map((s) => ({ label: s.name, color: s.color })), { label: "random ranking = 0", color: css("--neutral"), kind: "dash" }]);
  dotPlot(el("div", { class: "chart" }, box), { rows, series, refs: [{ x: 0, label: "random" }], xmin: Math.min(-0.1, ...rows.flatMap((r) => [r.B_lo, r.S_lo, r.F_lo])) ,
    xlabel: "PR-AUC skill (0 = random, 1 = perfect) · 95% CI",
    tipfmt: f3, ml: 90, rowH: 40, extraTip: (r, s) => [{ value: s.key === "F" ? r.chosen : s.key === "B" ? r.bname : "One-rule stump", label: "" },
      { value: f2((s.key === "F" ? r.F : s.key === "B" ? r.B : r.S) * (1 - r.prev) / r.prev + 1) + "×", label: "lift over random" }] });
}

function renderRandom() {
  const box = $("#random-table"); if (!box || !R.random?.length) return; box.replaceChildren();
  const ci = (a, b, f) => `${f(a)}–${f(b)}`, x = (v) => f2(v) + "×";
  table(box, [
    { label: "Country", get: (r) => r.target },
    { label: "Lift, riskiest 10%", num: true, get: (r) => x(r.lift10) },
    { label: "95% CI", num: true, get: (r) => ci(r.lift10_lo, r.lift10_hi, x) },
    { label: "Ceiling", num: true, get: (r) => f2(r.lift_ceiling) + "×" },
    { label: "Skill", num: true, get: (r) => f3(r.skill) },
    { label: "95% CI", num: true, get: (r) => ci(r.skill_lo, r.skill_hi, f3) },
    { label: "Random, 99.9th pct", num: true, get: (r) => x(r.null_lift10_p999) },
  ], COUNTRIES.map((c) => R.random.find((r) => r.target === c)).filter(Boolean));
}

function renderOperating() {
  const box = $("#op-table"); if (!box || !R.operating?.length) return;
  const ctr = $("#op-controls"); ctr.replaceChildren();
  segmented(ctr, COUNTRIES.map((c) => [c, c]), state.country, (c) => { state.country = c; renderOperating(); });
  box.replaceChildren();
  const rows = R.operating.filter((r) => r.target === state.country);
  table(box, [
    { label: "Inspect", get: (r) => `top ${pct(r.inspect)} (${fmtInt(r.k)})` },
    { label: "Dry points found", num: true, get: (r) => `${fmtInt(r.found)} of ${fmtInt(r.broken)}` },
    { label: "Random visits", num: true, get: (r) => fmtInt(r.found_random) },
    { label: "Foreign-only model", num: true, get: (r) => fmtInt(r.found_foreign) },
    { label: "Trips that find a dry point", num: true, get: (r) => pct(r.precision, 1) },
    { label: "Share of dry points found", num: true, get: (r) => pct(r.recall, 1) },
    { label: "Lift", num: true, get: (r) => f2(r.lift) + "×" },
  ], rows);
}

function renderAccuracy() {
  const box = $("#acc-table"); if (!box || !R.accuracy?.length) return; box.replaceChildren();
  table(box, [
    { label: "Country", get: (r) => r.target },
    { label: "\"Every point works\"", num: true, get: (r) => pct(r.acc_all_working, 1) },
    { label: "Model, 50% threshold", num: true, get: (r) => pct(r.acc_at_half, 1), cls: (r) => (r.acc_at_half > r.acc_all_working + 0.02 ? "pos" : "") },
    { label: "Points flagged at 50%", num: true, get: (r) => pct(r.flagged_at_half, 1) },
    { label: "Best possible threshold", num: true, get: (r) => pct(r.best_threshold_acc, 1) },
  ], COUNTRIES.map((c) => R.accuracy.find((r) => r.target === c)).filter(Boolean));
}

function renderProvider() {
  const box = $("#prov-table"); if (!box || !R.provider?.length) return;
  const ctr = $("#prov-controls"); ctr.replaceChildren();
  segmented(ctr, COUNTRIES.map((c) => [c, c]), state.country, (c) => { state.country = c; renderProvider(); });
  box.replaceChildren();
  const rows = R.provider.filter((r) => r.target === state.country).sort((a, b) => b.n - a.n);
  if (!rows.length) return;
  $("#prov-summary").textContent = `${state.country}: final model ROC-AUC ${f2(rows[0].model_roc)} on all test points; provider only ${f2(rows[0].provider_only_roc)}.`;
  table(box, [
    { label: "Survey provider", get: (r) => r.source },
    { label: "Test points", num: true, get: (r) => fmtInt(r.n) },
    { label: "No water", num: true, get: (r) => pct(r.no_water, 1) },
    { label: "Share of model's top 20%", num: true, get: (r) => pct(r.share_of_flagged, 1) },
    { label: "Ranking within provider (ROC-AUC)", num: true, get: (r) => f2(r.within_roc), cls: (r) => (r.within_roc != null && r.within_roc < 0.6 ? "negv" : "") },
  ], rows);
}

function renderFinal() {
  const box = $("#final-table"); if (!box) return;
  const ctr = $("#final-controls"); ctr.replaceChildren();
  segmented(ctr, COUNTRIES.map((c) => [c, c]), state.country, (c) => { state.country = c; renderFinal(); });
  box.replaceChildren();
  const cv = R.final_cv.filter((r) => r.target === state.country), te = R.final_test.filter((r) => r.target === state.country);
  if (!cv.length) { box.textContent = "Final-model results not exported yet."; return; }
  const rows = cv.map((r) => ({ ...r, test: te.find((t) => t.candidate === r.candidate) })).sort((a, b) => b.cv_pr_auc - a.cv_pr_auc);
  table(box, [
    { label: "Candidate", get: (r) => candName(r.candidate) },
    { label: "CV PR-AUC (local districts)", num: true, get: (r) => `${f3(r.cv_pr_auc)} ± ${f3(r.cv_sd)}` },
    { label: "Test PR-AUC", num: true, get: (r) => f3(r.test?.pr_auc) },
    { label: "Test ROC-AUC", num: true, get: (r) => f3(r.test?.roc_auc) },
    { label: "Brier", num: true, get: (r) => f3(r.test?.brier) },
    { label: "Precision @ top 10%", num: true, get: (r) => pct(r.test?.["prec@10%"], 1) },
  ], rows, (r) => (r.test?.chosen ? "chosen" : ""));
  const st = cv.find((r) => r.candidate === "stack");
  if (st) {
    const w = JSON.parse(st.params).weights;
    $("#stack-weights").textContent = "Stack weights on each model's log-odds: " + Object.entries(w).map(([k, v]) => `${candName(k)} ${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`).join(" · ");
  }
  renderCalibration(); renderImportance();
}

function renderCalibration() {
  const box = $("#calib-chart"); if (!box) return; box.replaceChildren();
  const r = R.reliability.filter((x) => x.target === state.country);
  if (!r.length) return;
  const ch = R.final_choice?.[state.country] || {};
  const vers = [["raw", "Raw model output", css("--s2"), ch.brier_raw], ["platt", "After Platt scaling", css("--s1"), ch.brier_platt]].filter(([v]) => r.some((x) => x.version === v));
  legend(box, vers.map(([, name, color, b]) => ({ label: `${name}${b != null ? ` (Brier ${f3(b)})` : ""}`, color, kind: "line" })).concat([{ label: "perfect calibration", color: css("--neutral"), kind: "dash" }]));
  const mx = Math.max(...r.map((x) => Math.max(x.pred, x.obs))) * 1.05;
  const series = vers.map(([v, name, color]) => ({ name, color, markers: true, pts: r.filter((x) => x.version === v).map((x) => ({ x: x.pred, y: x.obs, n: x.n })) }));
  const api = lineChart(el("div", { class: "chart" }, box), { h: 260, series, notip: true, xmin: 0, xmax: mx, ymin: 0, ymax: mx, refs: [{ diag: true }],
    xlabel: "Mean predicted probability (10 bins)", ylabel: "Observed rate", xfmt: (v) => pct(v), yfmt: (v) => pct(v) });
  for (const s of series) for (const p of s.pts) {
    const hit = el("circle", { cx: api.xs(p.x), cy: api.ys(p.y), r: 12, fill: "transparent", tabindex: 0 }, api.f.svg);
    const show = (e) => showTip(e, s.name, [{ color: s.color, value: pct(p.y, 1), label: `observed, where the model said ${pct(p.x, 1)} (${fmtInt(p.n)} points)` }]);
    hit.addEventListener("pointermove", show); hit.addEventListener("focus", show);
    hit.addEventListener("pointerleave", hideTip); hit.addEventListener("blur", hideTip);
  }
}

function renderImportance() {
  const box = $("#imp-chart"); if (!box) return; box.replaceChildren();
  const r = R.importance.filter((x) => x.target === state.country).sort((a, b) => b.pr_auc_drop - a.pr_auc_drop);
  if (!r.length) return;
  hbars(el("div", { class: "chart" }, box), { rows: r.map((x) => ({ label: FEATURE_NAME[x.feature] || x.feature, value: x.pr_auc_drop, err: x.sd, tip: `± ${f3(x.sd)} over 5 shuffles` })),
    xlabel: "Drop in test PR-AUC when shuffled", vfmt: f3 });
}

/* ---------- interactive: map + crew slider + transfer toggle ---------- */
let map, layer;
const mapState = { model: "score", k: 0.1 };
async function renderMap() {
  const ctr = $("#map-controls"); if (!ctr) return; ctr.replaceChildren();
  segmented(ctr, COUNTRIES.map((c) => [c, c]), state.country, (c) => { state.country = c; renderMap(); });
  segmented(ctr, [["score", "Final model (uses local data)"], ["score_b", "Foreign-only model (never saw this country)"]], mapState.model, (m) => { mapState.model = m; drawMap(true); });
  const lab = el("label", {}, ctr); lab.appendChild(document.createTextNode("Crew capacity: inspect top "));
  const sl = el("input", { type: "range", min: 1, max: 50, value: Math.round(mapState.k * 100), "aria-label": "Share of water points the crew can inspect" }, lab);
  const out = el("b", { text: ` ${Math.round(mapState.k * 100)}%` }, lab);
  sl.addEventListener("input", () => { mapState.k = sl.value / 100; out.textContent = ` ${sl.value}%`; drawMap(false); });
  await loadPoints(state.country);
  drawMap(true);
}
function ranked(P, key) {
  const idx = P.y.map((_, i) => i);
  // break ties at random but reproducibly so tied scores are not ordered by file position
  let s = 7; const jitter = idx.map(() => ((s = (s * 16807) % 2147483647) / 2147483647) * 1e-9);
  return idx.sort((a, b) => P[key][b] + jitter[b] - (P[key][a] + jitter[a]));
}
function drawMap(fit) {
  const P = points[state.country]; if (!P) { $("#map").textContent = "Map data not exported yet."; return; }
  if (!map) {
    map = L.map("map", { preferCanvas: true, scrollWheelZoom: false });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "&copy; OpenStreetMap contributors", maxZoom: 18 }).addTo(map);
  }
  const order = ranked(P, mapState.model), n = P.y.length, k = Math.max(1, Math.round(mapState.k * n));
  const flagged = new Uint8Array(n); for (let i = 0; i < k; i++) flagged[order[i]] = 1;
  const totalBroken = P.y.reduce((a, b) => a + b, 0);
  let found = 0; for (let i = 0; i < k; i++) found += P.y[order[i]];
  if (layer) layer.remove();
  layer = L.layerGroup();
  const col = { found: css("--s1"), missed: css("--s2"), alarm: css("--muted"), ok: css("--axis") };
  const draw = [[0, 0], [1, 0], [0, 1], [1, 1]];
  for (const [f, y] of draw) for (let i = 0; i < n; i++) {
    if (flagged[i] !== f || P.y[i] !== y) continue;
    const kind = f && y ? "found" : !f && y ? "missed" : f ? "alarm" : "ok";
    const m = L.circleMarker([P.lat[i], P.lon[i]], { radius: kind === "ok" ? 2 : 3.5, stroke: false, fillColor: col[kind], fillOpacity: kind === "ok" ? 0.35 : 0.85 });
    m.bindTooltip(() => {
      const d = document.createElement("div");
      const lines = [`${P.districts[P.district[i]]}`, `Predicted risk ${pct(P[mapState.model][i], 1)} (rank ${order.indexOf(i) + 1} of ${n})`,
        `Survey: ${P.y[i] ? "no water" : "water"}`, `${P.cats.water_tech_clean.levels[P.cats.water_tech_clean.codes[i]]} · ${P.cats.water_source_clean.levels[P.cats.water_source_clean.codes[i]]}`,
        `Age: ${P.age[i] == null ? "unknown" : P.age[i] + " years"}`];
      lines.forEach((t, j) => { const e = document.createElement(j ? "div" : "b"); e.textContent = t; d.appendChild(e); });
      return d;
    });
    layer.addLayer(m);
  }
  layer.addTo(map);
  if (fit) map.fitBounds(L.latLngBounds(P.lat.map((a, i) => [a, P.lon[i]])), { padding: [10, 10] });
  const lg = $("#map-legend"); lg.replaceChildren();
  legend(lg, [{ label: "Flagged and broken (found)", color: col.found }, { label: "Broken but not flagged (missed)", color: col.missed },
    { label: "Flagged but working (wasted trip)", color: col.alarm }, { label: "Not flagged, working", color: col.ok }]);
  const ro = $("#map-readout"); ro.replaceChildren();
  const randomFound = (k * totalBroken) / n;
  for (const [v, l] of [[fmtInt(k), `points inspected of ${fmtInt(n)}`], [fmtInt(found), `broken points found (${pct(found / totalBroken)} of all ${fmtInt(totalBroken)})`],
    [pct(found / k, 1), "of trips find a broken point"], [`${f2(found / randomFound)}×`, `vs random inspection (${fmtInt(randomFound)} found)`]]) {
    const d = el("div", {}, ro); el("b", { text: v }, d); d.appendChild(document.createTextNode(l));
  }
  renderCrewCurve(P); renderErrors(P, order, flagged);
}
function renderCrewCurve(P) {
  const box = $("#crew-chart"); if (!box) return;
  const n = P.y.length, tot = P.y.reduce((a, b) => a + b, 0);
  const curve = (key) => { const o = ranked(P, key); const pts = []; let c = 0; for (let i = 0; i < n; i++) { c += P.y[o[i]]; const f = (i + 1) / n; if (i === n - 1 || (i + 1) % Math.max(1, Math.floor(n / 100)) === 0) pts.push({ x: f, y: c / tot }); } return pts; };
  const fin = curve("score");
  lineChart(box, { h: 240, series: [
    { name: "Final model", color: css("--s1"), pts: fin }, { name: "Foreign-only model", color: css("--s2"), pts: curve("score_b") },
    { name: "Random inspection (expected)", color: css("--neutral"), dash: true, pts: fin.map((q) => ({ x: q.x, y: q.x })) }],
    xmin: 0, xmax: 1, ymin: 0, ymax: 1, xlabel: "Share of points inspected, riskiest first", ylabel: "Broken points found",
    xfmt: (v) => pct(v), yfmt: (v) => pct(v), tipfmt: (v) => pct(v, 1), xtipfmt: (v) => `inspect top ${pct(v)}`, refs: [] });
  const leg = $("#crew-legend"); leg.replaceChildren();
  legend(leg, [{ label: "Final model", color: css("--s1"), kind: "line" }, { label: "Foreign-only model", color: css("--s2"), kind: "line" }, { label: "Random inspection", color: css("--neutral"), kind: "dash" }]);
}

/* ---------- error explorer ---------- */
const errState = { by: "water_tech_clean" };
function renderErrors(P, order, flagged) {
  const box = $("#err-table"); if (!box) return;
  const ctr = $("#err-controls"); ctr.replaceChildren();
  const lab = el("label", { text: "Break errors down by " }, ctr);
  const sel = el("select", {}, lab);
  for (const [v, l] of [["water_tech_clean", "Technology"], ["water_source_clean", "Water source"], ["management_clean", "Management"], ["age", "Age band"], ["district", "District"]])
    el("option", { value: v, text: l, selected: v === errState.by ? "" : null }, sel);
  sel.addEventListener("change", () => { errState.by = sel.value; renderErrors(P, order, flagged); });
  const key = (i) => {
    if (errState.by === "district") return P.districts[P.district[i]];
    if (errState.by === "age") { const a = P.age[i]; return a == null ? "unknown" : a < 5 ? "0–4 years" : a < 10 ? "5–9" : a < 20 ? "10–19" : "20+"; }
    const c = P.cats[errState.by]; return c.levels[c.codes[i]];
  };
  const g = new Map();
  for (let i = 0; i < P.y.length; i++) {
    const k = key(i); if (!g.has(k)) g.set(k, { k, n: 0, broken: 0, pred: 0, found: 0, missed: 0, alarm: 0 });
    const r = g.get(k); r.n++; r.broken += P.y[i]; r.pred += P[mapState.model][i];
    if (flagged[i] && P.y[i]) r.found++; else if (!flagged[i] && P.y[i]) r.missed++; else if (flagged[i]) r.alarm++;
  }
  const rows = [...g.values()].filter((r) => r.n >= 20).sort((a, b) => b.n - a.n).slice(0, 25);
  box.replaceChildren();
  table(box, [
    { label: "Group", get: (r) => r.k }, { label: "Points", num: true, get: (r) => fmtInt(r.n) },
    { label: "Observed no-water", num: true, get: (r) => pct(r.broken / r.n, 1) }, { label: "Mean predicted", num: true, get: (r) => pct(r.pred / r.n, 1),
      cls: (r) => (Math.abs(r.pred / r.n - r.broken / r.n) > 0.1 ? "negv" : "") },
    { label: "Found", num: true, get: (r) => fmtInt(r.found) }, { label: "Missed", num: true, get: (r) => fmtInt(r.missed) },
    { label: "Wasted trips", num: true, get: (r) => fmtInt(r.alarm) },
  ], rows);
  el("p", { class: "note", text: "Groups with fewer than 20 test points are hidden. Red: mean predicted risk is more than 10 points away from the observed rate (the model is miscalibrated for that group). Found / missed / wasted use the crew capacity set above." }, box);
}

/* ---------- concepts ---------- */
function renderConcepts() {
  const lr = C.lr_scratch;
  if (lr && $("#lr-chart")) {
    lineChart($("#lr-chart"), { h: 240, xlog: true, series: [
      { name: "Gradient descent", color: css("--s2"), pts: lr.gd.map((r) => ({ x: r.iteration, y: r.gd_loss })) },
      { name: "Newton / IRLS", color: css("--s1"), markers: true, pts: lr.newton.map((r) => ({ x: r.iteration, y: r.newton_loss })) }],
      xlabel: "Iteration (log scale)", ylabel: "Training loss", yfmt: (v) => v.toFixed(3), tipfmt: (v) => v.toFixed(5), xtipfmt: (v) => `iteration ${v}` });
    const ck = lr.check;
    $("#lr-table").replaceChildren();
    table($("#lr-table"), [{ label: "Implementation", get: (r) => r[0] }, { label: "Iterations", num: true, get: (r) => r[1].iterations ?? "–" },
      { label: "Test PR-AUC", num: true, get: (r) => r[1].test_pr_auc.toFixed(5) }, { label: "Test ROC-AUC", num: true, get: (r) => r[1].test_roc_auc.toFixed(5) }],
      [["From scratch, Newton", ck.scratch_newton], ["From scratch, gradient descent", ck.scratch_gd], ["scikit-learn (L-BFGS)", ck.sklearn_lbfgs]]);
    $("#lr-diff").textContent = `Largest coefficient difference vs scikit-learn: Newton ${ck.max_abs_coef_diff_newton_vs_sklearn.toExponential(1)}, gradient descent ${ck.max_abs_coef_diff_gd_vs_sklearn.toFixed(3)} after ${ck.scratch_gd.iterations} steps. Largest predicted-probability difference (Newton): ${ck.max_abs_prob_diff_newton_vs_sklearn.toExponential(1)}. Fitted on ${fmtInt(ck.n_train)} Uganda training points, ${ck.n_features} features, λ = ${ck.lambda}.`;
  }
  if (C.reg_path && $("#reg-l1")) {
    for (const pen of ["l1", "l2"]) {
      const rows = C.reg_path.filter((r) => r.penalty === pen), feats = [...new Set(rows.map((r) => r.feature))];
      const maxC = Math.max(...rows.map((r) => r.C));
      const top = feats.map((f) => [f, Math.abs(rows.find((r) => r.feature === f && r.C === maxC).coef)]).sort((a, b) => b[1] - a[1]).slice(0, 3).map((x) => x[0]);
      const colors = [css("--s1"), css("--s2"), css("--s3")];
      const series = feats.map((f) => ({ name: prettyFeat(f), color: top.includes(f) ? colors[top.indexOf(f)] : css("--axis"), width: top.includes(f) ? 2 : 1,
        notip: !top.includes(f), pts: rows.filter((r) => r.feature === f).map((r) => ({ x: r.C, y: r.coef })) }));
      series.sort((a, b) => (a.notip ? 0 : 1) - (b.notip ? 0 : 1));
      const box = $(`#reg-${pen}`); box.replaceChildren();
      legend(box, top.map((f, i) => ({ label: prettyFeat(f), color: colors[i], kind: "line" })).concat([{ label: "other features", color: css("--axis"), kind: "line" }]));
      const nz = [...new Set(rows.map((r) => r.C))].sort((a, b) => a - b).map((c) => rows.filter((r) => r.C === c && Math.abs(r.coef) > 1e-6).length);
      lineChart(el("div", { class: "chart" }, box), { h: 240, xlog: true, series, xlabel: "C = 1/λ (log; right = weaker penalty)", ylabel: "Coefficient", xfmt: (v) => v.toExponential(0), tipfmt: (v) => v.toFixed(3), xtipfmt: (v) => `C = ${v.toExponential(1)}` });
      el("p", { class: "note", text: `${pen.toUpperCase()}: non-zero coefficients go from ${nz[0]} to ${nz[nz.length - 1]} of ${feats.length} as the penalty weakens.` }, box);
    }
  }
  if (C.pruning && $("#prune-chart")) {
    const p = C.pruning, xs = p.map((_, i) => i);
    lineChart($("#prune-chart"), { h: 250, series: [{ name: "Training districts", color: css("--s2"), markers: true, pts: p.map((r, i) => ({ x: i, y: r.train_pr_auc })) },
      { name: "Held-out districts", color: css("--s1"), markers: true, pts: p.map((r, i) => ({ x: i, y: r.val_pr_auc })) }],
      xticks: xs, xfmt: (i) => `${p[i].ccp_alpha}`, xtipfmt: (i) => `α = ${p[i].ccp_alpha} · ${Math.round(p[i].leaves)} leaves`, ymin: 0, ymax: 1,
      xlabel: "Cost-complexity α (more pruning →)", ylabel: "PR-AUC", tipfmt: f3 });
  }
  if (C.cv_optimism && $("#cvopt-chart")) {
    const r = C.cv_optimism;
    hbars($("#cvopt-chart"), { rows: r.map((x) => ({ label: x.estimate, value: x.pr_auc, err: x.sd, color: x.estimate.startsWith("district") ? css("--s1") : css("--s2") })),
      refs: [{ x: r[0].prevalence, label: "random" }], xlabel: "Estimated PR-AUC, same forest", ml: 140 });
  }
  if (C.boosting && $("#boost-chart")) {
    const b = C.boosting, colors = { 2: css("--s3"), 6: css("--s1"), 12: css("--s2") };
    const series = [];
    for (const d of [2, 6, 12]) {
      series.push({ name: `depth ${d}, held-out`, color: colors[d], pts: b.map((r) => ({ x: r.round, y: r[`val_d${d}`] })) });
      series.push({ name: `depth ${d}, training`, color: colors[d], dash: true, width: 1.5, pts: b.map((r) => ({ x: r.round, y: r[`train_d${d}`] })) });
    }
    const box = $("#boost-chart"); box.replaceChildren();
    legend(box, [2, 6, 12].map((d) => ({ label: `max depth ${d}`, color: colors[d], kind: "line" })).concat([{ label: "dashed = training", color: css("--muted"), kind: "dash" }]));
    lineChart(el("div", { class: "chart" }, box), { h: 260, series, xlabel: "Boosting round", ylabel: "PR-AUC", tipfmt: f3, xtipfmt: (v) => `round ${v}` });
  }
  if (C.svm_grid && $("#svm-chart")) {
    const g = C.svm_grid, Cs = [...new Set(g.map((r) => r.C))].sort((a, b) => a - b), gs = [...new Set(g.map((r) => r.gamma))].sort((a, b) => a - b);
    heatmap($("#svm-chart"), { rows: Cs, cols: gs, values: Cs.map((c) => gs.map((x) => g.find((r) => r.C === c && r.gamma === x)?.val_pr_auc ?? NaN)),
      xlabel: "γ", ylabel: "C", vlabel: "held-out-district PR-AUC" });
  }
  renderTuning();
}
const tuneState = { model: "tree", regime: "C" };
function renderTuning() {
  const box = $("#tune-chart"); if (!box || !C.tuning?.length) return;
  const ctr = $("#tune-controls"); ctr.replaceChildren();
  const models = ["logreg", "tree", "rf"].filter((m) => C.tuning.some((t) => t.model === m));
  segmented(ctr, models.map((m) => [m, MODEL_NAME[m]]), tuneState.model, (m) => { tuneState.model = m; renderTuning(); });
  segmented(ctr, ["A", "B", "C"].filter((r) => C.tuning.some((t) => t.regime === r)).map((r) => [r, `Regime ${r}`]), tuneState.regime, (r) => { tuneState.regime = r; renderTuning(); });
  const param = { logreg: "C", tree: "ccp_alpha", rf: "min_samples_leaf" }[tuneState.model];
  const series = COUNTRIES.map((c, i) => {
    const rows = C.tuning.filter((t) => t.model === tuneState.model && t.regime === tuneState.regime && t.target === c).sort((a, b) => a.params[param] - b.params[param]);
    return { name: c, color: [css("--s1"), css("--s2"), css("--s3"), css("--s4")][i], markers: true, pts: rows.map((r) => ({ x: Math.max(r.params[param], 1e-5), y: r.val })) };
  }).filter((s) => s.pts.length);
  box.replaceChildren();
  if (!series.length) { box.textContent = "No tuning results for this combination yet."; return; }
  legend(box, series.map((s) => ({ label: s.name, color: s.color, kind: "line" })));
  lineChart(el("div", { class: "chart" }, box), { h: 240, xlog: param !== "min_samples_leaf", series, xlabel: `${param} (grid searched)`, ylabel: "CV score (normalised PR-AUC)",
    xfmt: (v) => (v < 0.01 ? v.toExponential(0) : v), tipfmt: f3, xtipfmt: (v) => `${param} = ${v}` });
}

/* ---------- ablations ---------- */
const ablState = { model: "xgb" };
function renderAblations() {
  const box = $("#abl-chart"); if (!box) return;
  const ctr = $("#abl-controls"); ctr.replaceChildren();
  segmented(ctr, COUNTRIES.map((c) => [c, c]), state.country, (c) => { state.country = c; renderAblations(); });
  segmented(ctr, ["logreg", "tree", "xgb"].map((m) => [m, MODEL_NAME[m]]), ablState.model, (m) => { ablState.model = m; renderAblations(); });
  const variants = Object.keys(VARIANT_NAME).filter((v) => R.metrics.some((m) => m.variant === v && m.target === state.country));
  const rows = variants.map((v) => {
    const row = { label: VARIANT_NAME[v] };
    for (const r of ["A", "B", "C"]) { const x = R.metrics.find((m) => m.variant === v && m.target === state.country && m.model === ablState.model && m.regime === r); if (x) row[r] = skill(x.pr_auc, x.prevalence); }
    return row;
  });
  box.replaceChildren();
  const series = ["A", "B", "C"].map((r) => ({ key: r, name: REGIME[r], color: css(REGIME_COLOR[r]) }));
  legend(box, [...series.map((s) => ({ label: s.name, color: s.color })), { label: "random ranking = 0", color: css("--neutral"), kind: "dash" }]);
  dotPlot(el("div", { class: "chart" }, box), { rows, series, refs: [{ x: 0, label: "random" }], xmin: Math.min(-0.05, ...rows.flatMap((r) => [r.A, r.B, r.C].filter((v) => v != null))),
    xlabel: "Test PR-AUC skill (0 = random, 1 = perfect)", tipfmt: f3, ml: 190 });
}

/* ---------- boot ---------- */
function renderAll() {
  renderHero(); renderDataTable(); renderHeadline(); renderRandom(); renderOperating(); renderAccuracy(); renderRegimes(); renderCountryModels(); renderFinal();
  renderProvider(); renderConcepts(); renderAblations();
  if (window.L) renderMap();
}
async function boot() {
  [R, C, O] = await Promise.all(["results", "concepts", "overview"].map((n) => fetch(`data/${n}.json`).then((r) => r.json())));
  renderAll();
  let t; window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { renderHero(); renderHeadline(); renderRegimes(); renderCountryModels(); renderFinal(); renderConcepts(); renderAblations(); if (points[state.country]) renderCrewCurve(points[state.country]); }, 200); });
  const btn = $("#theme");
  btn?.addEventListener("click", () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark" || (!document.documentElement.hasAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
    try { localStorage.setItem("theme", dark ? "light" : "dark"); } catch (e) { /* storage unavailable */ }
    renderAll();
  });
}
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.setAttribute("data-theme", t); } catch (e) { /* storage unavailable */ }
boot();
