"use strict";
/* NIFTY News to Volatility Predictor - web app.
   Pages: Overview (/), System 1 News & Sentiment (/system-1), System 2 Predictions (/system-2),
   System 3 Backtesting (/system-3), Settings (/settings). Client-side routing with real URLs. */
const API = "/api";
const state = { config: null, charts: {}, companies: [], tz: "Asia/Kolkata", page: null };

// ============================================================ helpers
const $ = (s, root = document) => root.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '<span class="missing">n/a</span>' : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "n/a" : (v * 100).toFixed(1) + "%");
const signed = (v, d = 2) => (v === null || v === undefined ? '<span class="missing">n/a</span>' : (v >= 0 ? "+" : "") + Number(v).toFixed(d));
const fmtTime = (iso) => (iso ? new Intl.DateTimeFormat("en-IN", { timeZone: state.tz, dateStyle: "medium", timeStyle: "short" }).format(new Date(iso)) : "n/a");
const fmtDate = (d) => (d ? new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeZone: "UTC" }).format(new Date(d + "T00:00:00Z")) : "n/a");
const ago = (iso) => {
  if (!iso) return "never";
  const m = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (m < 1) return "just now"; if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60); if (h < 48) return `${h} h ago`;
  return `${Math.round(h / 24)} days ago`;
};
// Admin token (public deployments): actions need it, viewing does not. Kept only in this browser.
const TOKEN_KEY = "nvp-admin-token";
const getToken = () => { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; } };
const setToken = (t) => { try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch {} };
async function api(path, opts = {}, retried = false) {
  const headers = { "Content-Type": "application/json", ...(getToken() ? { "X-Admin-Token": getToken() } : {}) };
  const res = await fetch(API + path, { ...opts, headers });
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (res.status === 401 && !retried) {
    const t = prompt("This action needs the admin token (ADMIN_TOKEN on the server). It is stored only in this browser.");
    if (t) { setToken(t.trim()); return api(path, opts, true); }
  }
  if (res.status === 401) setToken("");
  if (!res.ok) throw new Error(body?.detail || body || res.statusText);
  return body;
}
function status(msg, isErr = false) { const el = $("#status"); el.textContent = msg || ""; el.classList.toggle("err", !!isErr); }
const tile = (label, value, sub = "", color = "") => `<div class="tile"><div class="label">${label}</div><div class="value"${color ? ` style="color:${color}"` : ""}>${value}</div><div class="sub">${sub}</div></div>`;
const pill = (text, color) => `<span class="pill" style="--c:${color}">${esc(text)}</span>`;
/** cls: "stack" = list table that becomes one card per row on phones; "wide" = numeric table that scrolls. */
function tableHtml(headers, rows, id = "", cls = "") {
  return `<table${id ? ` id="${id}"` : ""}${cls ? ` class="${cls}"` : ""}><thead><tr>${headers.map((h) => `<th class="${h.num ? "num" : ""}">${esc(h.t ?? h)}</th>`).join("")}</tr></thead>
    <tbody>${rows.length ? rows.join("") : `<tr><td colspan="${headers.length}" class="missing">No data</td></tr>`}</tbody></table>`;
}
function table(el, headers, rows, cls = "") { el.outerHTML = tableHtml(headers, rows, el.id, cls); }
/** Copies each column header into its cells (data-label) so stacked phone cards stay labelled. */
function labelStacks(root = document) {
  root.querySelectorAll("table.stack").forEach((t) => {
    const heads = [...t.querySelectorAll("thead th")].map((th) => th.textContent.trim());
    t.querySelectorAll("tbody tr").forEach((tr) => [...tr.children].forEach((td, i) => {
      if (td.dataset.label !== undefined) return;
      td.dataset.label = heads[i] ?? "";
      if (i > 0 && td.childNodes.length) {   // one value block per cell, so the phone card shows "label | value"
        const w = document.createElement("div"); w.className = "cv";
        while (td.firstChild) w.appendChild(td.firstChild);
        td.appendChild(w);
      }
    }));
  });
}
new MutationObserver(() => labelStacks($("#view"))).observe(document.getElementById("view"), { childList: true, subtree: true });
function chart(id, cfg) {
  state.charts[id]?.destroy();
  const el = document.getElementById(id);
  if (!el || !window.Chart) return;
  Chart.defaults.color = css("--text-2");
  Chart.defaults.borderColor = css("--grid");
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  state.charts[id] = new Chart(el, cfg);
}
function sentimentBadge(score) {
  if (score === null || score === undefined) return '<span class="missing">n/a</span>';
  const c = score > 15 ? "var(--pos)" : score < -15 ? "var(--neg)" : "var(--muted)";
  const label = score > 15 ? "Positive" : score < -15 ? "Negative" : "Neutral";
  return `<span class="dot" style="background:${c}"></span>${Number(score).toFixed(1)} <span class="muted">${label}</span>`;
}
function relativeCell(flow) {
  if (!flow) return '<span class="missing">n/a (raw model)</span>';
  if (flow.relative_sentiment === null || flow.relative_sentiment === undefined)
    return `<span class="missing">n/a</span><div class="muted small">${esc(flow.status)}</div>`;
  const r = Number(flow.relative_sentiment);
  const c = r > 0 ? "var(--pos)" : r < 0 ? "var(--neg)" : "var(--muted)";
  return `<span class="dot" style="background:${c}"></span>${r >= 0 ? "+" : ""}${r.toFixed(1)}
    <div class="muted small">${flow.news_count} new · z ${flow.sentiment_zscore === null || flow.sentiment_zscore === undefined ? "n/a" : Number(flow.sentiment_zscore).toFixed(2)}</div>`;
}
const AVAIL = {
  verified_pre_market: ["VERIFIED", "var(--good)", "collected by this system before the 09:15 IST cut-off"],
  published_before_cutoff: ["UNCERTAIN", "var(--serious)", "precise timestamp, but collected after the cut-off"],
  collected_after_event: ["POST-EVENT", "var(--critical)", "collected after the session it could predict had closed"],
  unknown: ["UNCERTAIN", "var(--serious)", "date-only / unknown timestamp, not verified"],
};
function availChip(st, precision) {
  const [label, c, why] = AVAIL[st] || AVAIL.unknown;
  return `<span class="badge" style="color:${c}" title="${esc(why)}">${label}</span>${precision ? ` <span class="chip">${esc(precision)}</span>` : ""}`;
}
function qualityBadge(flag) {
  const m = { VERIFIED: ["VERIFIED DATA", "var(--good)"], UNCERTAIN: ["UNCERTAIN DATA", "var(--serious)"], NO_NEW_NEWS: ["NO NEW NEWS", "var(--muted)"] };
  const [t, c] = m[flag] || ["QUALITY N/A (older run)", "var(--muted)"];
  return `<span class="badge" style="color:${c}">${t}</span>`;
}
function directionBadge(d) {
  const sym = { UP: "▲ UP", DOWN: "▼ DOWN", NEUTRAL: "● NEUTRAL", MISSING: "— MISSING" }[d] || esc(d || "");
  const c = { UP: "var(--pos)", DOWN: "var(--neg)" }[d] || "var(--muted)";
  return `<span class="badge" style="color:${c}">${sym}</span>`;
}
const PROVIDER_LABEL = { openai: "OpenAI", anthropic: "Claude (Anthropic)", ollama: "Ollama (local model)", "fake-llm": "test stand-in", none: "disabled" };
const REVIEW = { supported: ["✓ SUPPORTED", "var(--good)"], weakly_supported: ["~ WEAKLY SUPPORTED", "var(--serious)"],
  contradicted: ["✕ CONTRADICTED", "var(--critical)"], indeterminate: ["? INDETERMINATE", "var(--muted)"],
  validation_error: ["⚠ VALIDATION ERROR", "var(--muted)"], api_error: ["⚠ API ERROR", "var(--muted)"],
  not_performed: ["NOT PERFORMED", "var(--muted)"], CONSISTENT: ["✓ CONSISTENT", "var(--good)"],
  QUESTIONABLE: ["! QUESTIONABLE", "var(--serious)"], INCONSISTENT: ["✕ INCONSISTENT", "var(--critical)"],
  INSUFFICIENT_DATA: ["? INSUFFICIENT DATA", "var(--muted)"], VALIDATION_FAILED: ["⚠ FAILED", "var(--muted)"] };
const reviewBadge = (s) => { const [t, c] = REVIEW[s] || [s, "var(--muted)"]; return `<span class="badge" style="color:${c}">${esc(t)}</span>`; };
function setHead(title, sub, actionsHtml = "") {
  $("#page-title").textContent = title;
  $("#page-sub").innerHTML = sub;
  $("#page-actions").innerHTML = actionsHtml;
  document.title = `${title} · NIFTY News Volatility`;
}
/** Run a long action from a button: disables it, shows progress, reports the result, reloads the page. */
async function runAction(btn, label, fn) {
  const all = [...document.querySelectorAll("#page-actions button")];
  all.forEach((b) => (b.disabled = true));
  status(label);
  try { status(await fn()); await reloadPage(); }
  catch (e) { status("Error: " + e.message, true); }
  finally { all.forEach((b) => (b.disabled = false)); }
}

// ============================================================ router
const PAGES = {
  "/": { title: "Overview", load: pageOverview },
  "/overview": { title: "Overview", load: pageOverview },
  "/system-1": { title: "News & Sentiment", load: pageSystem1 },
  "/system-2": { title: "Predictions", load: pageSystem2 },
  "/system-3": { title: "Backtesting", load: pageSystem3 },
  "/settings": { title: "Settings", load: pageSettings },
};
function navigate(path) {
  if (path !== location.pathname) history.pushState({}, "", path);
  render();
}
async function render() {
  const path = PAGES[location.pathname] ? location.pathname : "/";
  state.page = path;
  Object.values(state.charts).forEach((c) => c.destroy());
  state.charts = {};
  document.querySelectorAll(".nav a").forEach((a) => {
    const r = a.dataset.route;
    a.toggleAttribute("aria-current", r === path || (r === "/" && path === "/overview"));
    if (a.hasAttribute("aria-current")) a.setAttribute("aria-current", "page");
  });
  status("");
  $("#view").innerHTML = '<div class="skeleton">Loading…</div>';
  try { await PAGES[path].load(); }
  catch (e) { $("#view").innerHTML = `<div class="card"><b>Could not load this page.</b><p class="missing">${esc(e.message)}</p></div>`; }
  window.scrollTo(0, 0);
}
async function reloadPage() { const y = window.scrollY; await PAGES[state.page].load(); window.scrollTo(0, y); }
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-link]");
  if (!a || e.ctrlKey || e.metaKey || e.shiftKey || e.button !== 0) return;
  e.preventDefault();
  navigate(a.getAttribute("href"));
});
window.addEventListener("popstate", render);

// ============================================================ OVERVIEW
async function pageOverview() {
  setHead("Overview", "Status of the three systems, scheduled jobs and configuration.",
    `<button id="ov-refresh">Refresh</button>`);
  $("#ov-refresh").onclick = () => reloadPage();
  const s = await api("/status");
  const s1 = s.system1, s2 = s.system2, s3 = s.system3, run = s2.latest_run, bt = s3.latest_backtest;
  const dirs = run?.next_session_directions || {};
  const vals = run?.llm_validations || {};
  const valText = Object.keys(vals).length ? Object.entries(vals).map(([k, n]) => `${reviewBadge(k)} ×${n}`).join(" ") : '<span class="muted">none yet</span>';
  const llmState = s2.llm_configured === false ? pill(s2.llm_status || "not configured", "var(--serious)")
    : pill(`${PROVIDER_LABEL[s2.llm_provider] || s2.llm_provider}${s2.llm_model ? " · " + s2.llm_model : ""}`, "var(--good)");
  const btVerdict = !bt ? '<span class="muted">no backtest yet</span>'
    : bt.beats_all_baselines ? pill("beats every baseline on MAE (check significance)", "var(--good)")
    : pill("does not beat the no-change baseline", "var(--serious)");
  const sysCard = (n, title, href, body, open) => `<div class="card system-card"><div class="head"><span class="badge-num">${n}</span><div><h2>${title}</h2></div></div>${body}
    <div class="go"><a class="btn-link" href="${href}" data-link>${open} →</a></div></div>`;
  const jobs = (s.schedule || []).map((j) => `<tr><td><b>${esc(j.task)}</b><div class="muted small">${esc(j.task === "NiftyNews-PreMarket" ? "08:30 IST · collect → predict → LLM review" : "16:30 IST · prices → actual outcomes")}</div></td>
      <td>${j.registered ? pill(j.status || "registered", "var(--good)") : pill("not registered", "var(--serious)")}</td>
      <td>${esc(j.next_run || "—")}</td><td>${esc(j.last_run || "—")}</td><td>${esc(j.last_result ?? "—")}</td>
      <td>${j.log_updated ? `${ago(j.log_updated)}<details><summary class="small">log</summary><pre class="log">${esc(j.log_tail || "")}</pre></details>` : '<span class="muted">no log yet</span>'}</td></tr>`);
  $("#view").innerHTML = `
    <div class="grid3">
      ${sysCard(1, "News collector & sentiment", "/system-1", `<dl>
        <dt>Articles stored</dt><dd>${s1.articles.toLocaleString("en-IN")}</dd>
        <dt>Verified pre-market</dt><dd>${s1.verified_pre_market.toLocaleString("en-IN")}</dd>
        <dt>Last collection</dt><dd>${fmtTime(s1.last_fetched_at)} <span class="muted">(${ago(s1.last_fetched_at)})</span></dd>
        <dt>News providers</dt><dd>${s1.providers_active.map((p) => `<span class="chip">${esc(p)}</span>`).join("")}</dd>
        <dt>Sentiment engine</dt><dd>${esc(s1.sentiment_engine)}</dd></dl>`, "Open System 1")}
      ${sysCard(2, "Deviation predictor", "/system-2", run ? `<dl>
        <dt>Latest run</dt><dd>${fmtTime(run.created_at)} <span class="muted">(${ago(run.created_at)})</span></dd>
        <dt>Target session</dt><dd>${run.target_session ? fmtDate(run.target_session) : "n/a"}</dd>
        <dt>Cut-off</dt><dd>${run.prediction_cutoff ? fmtTime(run.prediction_cutoff) + " IST" : "n/a"}</dd>
        <dt>Predictions</dt><dd>${run.ok}/${run.predictions} · ${directionBadge("UP")} ${dirs.UP ?? 0} ${directionBadge("DOWN")} ${dirs.DOWN ?? 0} ${directionBadge("NEUTRAL")} ${dirs.NEUTRAL ?? 0}</dd>
        <dt>Model</dt><dd>${esc(run.model_version)}</dd>
        <dt>LLM validation</dt><dd>${llmState}<div class="small">${valText}</div></dd>
        <dt>Live outcomes</dt><dd>${s2.live_outcomes_scored} scored · ${s2.live_outcomes_pending} pending</dd></dl>` : '<p class="missing">No predictions yet.</p>', "Open System 2")}
      ${sysCard(3, "Backtesting & fine-tuning", "/system-3", bt ? `<dl>
        <dt>Latest backtest</dt><dd>${esc(bt.run_id)} <span class="muted">(${ago(bt.finished_at)})</span></dd>
        <dt>Period</dt><dd>${esc(bt.period)} · ${bt.n?.toLocaleString("en-IN") ?? "n/a"} predictions</dd>
        <dt>Next-session MAE</dt><dd>${num(bt.next_session_mae, 3)} <span class="muted">vs no-change ${num(bt.baseline_mae?.no_change, 3)}</span></dd>
        <dt>Sign accuracy</dt><dd>${pct(bt.next_session_sign_accuracy)}</dd>
        <dt>Verdict</dt><dd>${btVerdict}</dd>
        <dt>Tuning</dt><dd class="small">${esc(bt.tuning_reason || "not tuned")}</dd></dl>` : '<p class="missing">No backtests yet.</p>', "Open System 3")}
    </div>
    <div class="section-title"><h2>Scheduled jobs</h2><span class="muted small">Windows Task Scheduler · run only while this user is logged on</span></div>
    <div class="card"><div class="scroll">${tableHtml(["Job", "Status", "Next run", "Last run", "Last result", "Log"], jobs, "", "stack")}</div></div>
    <div class="section-title"><h2>Configuration</h2><a href="/settings" data-link class="small">Settings →</a></div>
    <div class="tiles">
      ${tile("Active parameters", esc(s.config.active_parameters), "System 2 model settings")}
      ${tile("Universe", `<span style="font-size:15px">${esc(s.config.universe_version)}</span>`, s.config.universe_members.join(", "))}
      ${tile("LLM validation", `<span style="font-size:15px">${esc(PROVIDER_LABEL[s2.llm_provider] || s2.llm_provider)}</span>`, esc(s2.llm_model || s2.llm_status || ""))}
      ${tile("Display timezone", `<span style="font-size:15px">${esc(s.config.display_timezone)}</span>`, "all times on this site")}
    </div>
    <div class="note">The model is <b>experimental</b>: on May–Sep 2026 history it does not beat a no-change forecast. Accuracy can only be judged on news collected before each 09:15 IST cut-off, which the scheduled jobs are now building up.</div>`;
}

// ============================================================ SYSTEM 1
async function pageSystem1() {
  setHead("System 1 · News & Sentiment", "Collects news for the Top-10 NIFTY companies and macro topics, scores sentiment from −100 to +100 per company and for NIFTY, for today, the next 3 month-ends and the next 6 quarter-ends.",
    `<button id="a-collect" class="primary">Collect news now</button><button id="a-sent">Recompute sentiment</button>`);
  $("#a-collect").onclick = () => runAction($("#a-collect"), "Collecting news from all providers (about a minute)…", async () => {
    const r = await api("/news/collect", { method: "POST", body: "{}" });
    return `Collected ${r.new} new of ${r.fetched} fetched (${r.exact_duplicates} duplicates, ${r.near_duplicates} near-duplicates, ${r.rejected_future || 0} future-dated rejected), analysed ${r.analyzed}. Errors: ${r.errors.length}.`;
  });
  $("#a-sent").onclick = () => runAction($("#a-sent"), "Recomputing sentiment for all horizons…", async () => {
    const r = await api("/sentiment/aggregate", { method: "POST" });
    return `Sentiment recomputed (${r.horizons.length} horizons) at ${fmtTime(r.timestamp)}.`;
  });
  $("#view").innerHTML = `
    <div class="filters"><label>Horizon <select id="s-horizon"></select></label><span id="s-asof" class="muted small"></span></div>
    <div class="tiles" id="s-tiles"></div>
    <div class="grid2">
      <div class="card"><h3>Company sentiment (−100 … +100)</h3><div class="chart"><canvas id="c-company"></canvas></div></div>
      <div class="card"><h3>Sentiment over time (next-session horizon)</h3><div class="chart"><canvas id="c-history"></canvas></div></div>
    </div>
    <div class="card"><h3>Scores by horizon</h3><div class="scroll"><table id="t-horizons"></table></div></div>
    <div class="section-title"><h2>News data quality</h2><span class="muted small">only VERIFIED news is used for live predictions</span></div>
    <div id="dq-body" class="skeleton">Loading…</div>
    <div class="section-title"><h2>Latest news</h2></div>
    <div class="card">
      <div class="filters">
        <label>Company <select id="n-company"><option value="">All</option>${state.companies.map((x) => `<option value="${esc(x.ticker)}">${esc(x.name)}</option>`).join("")}<option value="MACRO">Macro / market</option></select></label>
        <label>Availability <select id="n-avail"><option value="">All</option><option value="verified_pre_market">Verified pre-market</option><option value="uncertain">Uncertain</option><option value="collected_after_event">Post-event</option></select></label>
        <span id="n-count" class="muted small"></span>
      </div>
      <div class="scroll"><table id="t-news"></table></div>
    </div>`;
  const s = await api("/sentiment");
  const sel = $("#s-horizon");
  if (s.status === "MISSING") {
    $("#s-tiles").innerHTML = tile("Sentiment", '<span class="missing">none yet</span>', esc(s.message));
  } else {
    sel.innerHTML = s.horizons.map((h, i) => `<option value="${i}">${esc(h.label)} (${h.target_date})</option>`).join("");
    $("#s-asof").textContent = `Computed ${fmtTime(s.timestamp)} · parameters ${s.param_version}`;
    sel.onchange = () => renderHorizon(s);
    renderHorizon(s);
    renderHorizonTable(s);
  }
  $("#n-company").onchange = loadNewsTable;
  $("#n-avail").onchange = loadNewsTable;
  await Promise.all([loadHistory(), loadNewsTable(), loadDataQuality()]);
}
function renderHorizon(s) {
  const h = s.horizons[$("#s-horizon").value || 0];
  const t = (label, v, sub) => tile(label, v === null || v === undefined ? '<span class="missing">n/a</span>' : Number(v).toFixed(1), sub);
  $("#s-tiles").innerHTML = t("NIFTY overall", h.nifty_score, "company-weighted + macro blend") +
    t("NIFTY company-weighted", h.nifty_company_weighted_score, `weight coverage ${pct(h.nifty_weight_coverage)}`) +
    t("Macro", h.macro_score, `${h.macro_news_count ?? 0} macro stories`) +
    tile("Companies with news", h.company_scores.filter((c) => c.news_count > 0).length, `of ${h.company_scores.length}`);
  const cs = [...h.company_scores].sort((a, b) => b.sentiment_score - a.sentiment_score);
  chart("c-company", {
    type: "bar",
    data: { labels: cs.map((c) => c.ticker), datasets: [{ data: cs.map((c) => c.sentiment_score),
      backgroundColor: cs.map((c) => (c.sentiment_score >= 0 ? css("--pos") : css("--neg"))), borderRadius: 4, maxBarThickness: 26 }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => `score ${ctx.raw.toFixed(1)} · ${cs[ctx.dataIndex].news_count} stories` } } },
      scales: { x: { min: -100, max: 100, grid: { color: css("--grid") } }, y: { grid: { display: false } } } },
  });
}
function renderHorizonTable(s) {
  const tickers = s.horizons[0].company_scores.map((c) => c.ticker);
  const rows = s.horizons.map((h) => {
    const by = Object.fromEntries(h.company_scores.map((c) => [c.ticker, c.sentiment_score]));
    return `<tr><td>${esc(h.label)}</td><td>${h.target_date}</td><td class="num">${num(h.nifty_score, 1)}</td><td class="num">${num(h.nifty_company_weighted_score, 1)}</td><td class="num">${num(h.macro_score, 1)}</td>${tickers.map((t) => `<td class="num">${num(by[t], 1)}</td>`).join("")}</tr>`;
  });
  table($("#t-horizons"), ["Horizon", "Target", { t: "NIFTY", num: 1 }, { t: "NIFTY (co.)", num: 1 }, { t: "Macro", num: 1 }, ...tickers.map((t) => ({ t, num: 1 }))], rows, "wide");
}
async function loadHistory() {
  const series = [["NIFTY", "NIFTY overall", "--s1"], ["NIFTY_COMPANY", "NIFTY company-weighted", "--s2"], ["MACRO", "Macro", "--s3"]];
  const data = await Promise.all(series.map(([e]) => api(`/sentiment/history?entity=${e}&horizon_type=today&days=180`)));
  const labels = [...new Set(data.flatMap((d) => d.points.map((p) => p.date)))].sort();
  chart("c-history", {
    type: "line",
    data: { labels, datasets: series.map(([, label, color], i) => {
      const by = Object.fromEntries(data[i].points.map((p) => [p.date, p.score]));
      return { label, data: labels.map((l) => by[l] ?? null), borderColor: css(color), backgroundColor: css(color), borderWidth: 2, pointRadius: labels.length < 30 ? 4 : 0, spanGaps: true, tension: 0.2 };
    }) },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "top", labels: { boxWidth: 10 } } },
      scales: { y: { min: -100, max: 100, grid: { color: css("--grid") } }, x: { grid: { display: false } } } },
  });
}
async function loadDataQuality() {
  const a = await api("/news/audit");
  const o = a.overall;
  const uncertain = o.status.unknown + o.status.published_before_cutoff;
  const provRows = Object.entries(a.by_provider).map(([p, v]) => `<tr><td><b>${esc(p)}</b></td><td class="num">${v.articles.toLocaleString("en-IN")}</td>
    <td class="num">${pct(v.precision_share.exact + v.precision_share.minute)}</td><td class="num">${pct(v.precision_share.hour)}</td>
    <td class="num">${pct(v.precision_share.date_only)}</td><td class="num">${pct(v.verified_pre_market_share)}</td>
    <td class="small">${esc(v.timestamp_quality)}</td></tr>`);
  $("#dq-body").outerHTML = `<div id="dq-body"><div class="tiles">
      ${tile("VERIFIED pre-market", o.status.verified_pre_market.toLocaleString("en-IN"), `${pct(o.verified_pre_market_share)} of ${o.articles.toLocaleString("en-IN")} — held before the 09:15 IST cut-off`, "var(--good)")}
      ${tile("UNCERTAIN", uncertain.toLocaleString("en-IN"), "date-only / unverified timestamps", "var(--serious)")}
      ${tile("POST-EVENT", o.status.collected_after_event.toLocaleString("en-IN"), "collected after the session closed")}
      ${tile("Duplicates", pct(o.duplicate_rate), `rejected: future ${o.rejected_future} · missing time ${o.rejected_missing_timestamp}`)}
    </div>
    <div class="card"><h3>Providers</h3><div class="scroll">${tableHtml(["Provider", { t: "Articles", num: 1 }, { t: "Exact/minute", num: 1 }, { t: "Hour", num: 1 }, { t: "Date-only", num: 1 }, { t: "Verified", num: 1 }, "Timestamp quality"], provRows, "", "stack")}</div></div></div>`;
}
async function loadNewsTable() {
  const t = $("#n-company").value, av = $("#n-avail").value;
  const r = await api(`/news?limit=300${t ? "&ticker=" + encodeURIComponent(t) : ""}`);
  const items = r.items.filter((a) => !av || (av === "uncertain" ? ["unknown", "published_before_cutoff"].includes(a.availability_status) : a.availability_status === av));
  $("#n-count").textContent = `${items.length} shown (latest ${r.count} articles)`;
  const rows = items.slice(0, 150).map((a) => {
    const an = a.analysis.filter((x) => x.entity !== "NONE");
    return `<tr><td class="small" style="min-width:110px">${fmtTime(a.published_at)}<div>${availChip(a.availability_status, a.timestamp_precision)}</div></td>
      <td class="small">${esc(a.source)}<div class="muted">${esc(a.provider)}</div>${a.duplicate_of ? '<span class="chip">duplicate</span>' : ""}</td>
      <td><a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.title)}</a></td>
      <td>${an.map((x) => `<span class="chip">${esc(x.entity)} · ${esc(x.category)}${x.evidence?.price_report ? " · price move" : ""}</span>`).join("")}</td>
      <td>${(an[0]?.keywords || []).slice(0, 5).map((k) => `<span class="chip">${esc(k)}</span>`).join("")}</td>
      <td class="num">${sentimentBadge(an[0]?.sentiment_score)}</td></tr>`;
  });
  table($("#t-news"), ["Published / availability", "Source", "Headline", "Company · event", "Keywords", { t: "Sentiment", num: 1 }], rows, "stack");
}

// ============================================================ SYSTEM 2
const s2 = { data: null, runs: [] };
async function pageSystem2() {
  const llm = state.config;
  const llmReady = llm.llm_configured !== false;
  setHead("System 2 · Predictions", "Predicted % movement per company for the next session, the next 3 month-ends and the next 6 quarter-ends, from verified pre-market news and the previous close. LLM validation reviews the predictions but never changes them.",
    `<button id="a-prices">Refresh prices</button><button id="a-predict" class="primary">Run predictions</button>
     <select id="a-val-ticker" aria-label="Company to validate"><option value="">All companies</option>${state.companies.map((x) => `<option value="${esc(x.ticker)}">${esc(x.ticker)}</option>`).join("")}</select>
     <button id="a-validate" ${llmReady ? "" : "disabled title=\"OpenAI validation not configured\""}>Validate with ${esc(PROVIDER_LABEL[llm.llm_provider] || llm.llm_provider)}</button>
     <button id="a-score">Score outcomes</button>`);
  $("#a-prices").onclick = () => runAction($("#a-prices"), "Downloading latest prices…", async () => {
    const r = await api("/market-data/refresh", { method: "POST" });
    const bad = Object.entries(r).filter(([, v]) => v.status !== "ok");
    return `Prices refreshed for ${Object.keys(r).length - bad.length} symbols${bad.length ? "; failed: " + bad.map(([k]) => k).join(", ") : ""}.`;
  });
  $("#a-predict").onclick = () => runAction($("#a-predict"), "Running System 1 + System 2…", async () => {
    const r = await api("/predictions", { method: "POST", body: "{}" });
    s2.runId = r.run_id;
    return `Prediction run ${r.run_id} created for ${r.target_session} (cut-off ${r.prediction_cutoff}): ${r.predictions.length} predictions.`;
  });
  $("#a-validate").onclick = () => {
    const t = $("#a-val-ticker").value;
    const n = t ? 1 : 10;
    runAction($("#a-validate"), `Validating ${t || "all companies"} with ${PROVIDER_LABEL[llm.llm_provider] || llm.llm_provider} (about ${n * 25} s)…`, async () => {
      const body = { run_id: s2.runId || undefined, tickers: t ? [t] : undefined };
      const r = await api("/predictions/validate", { method: "POST", body: JSON.stringify(body) });
      const res = Object.values(r.results || {});
      const ok = res.filter((x) => x.validation_performed).length;
      return `LLM validation for ${res.length} compan${res.length === 1 ? "y" : "ies"}: ${ok} completed${ok < res.length ? `, ${res.length - ok} not performed` : ""}. Predictions unchanged.`;
    });
  };
  $("#a-score").onclick = () => runAction($("#a-score"), "Attaching actual outcomes to matured predictions…", async () => {
    const r = await api("/backtest/score-live", { method: "POST" });
    return `Outcomes: ${r.scored} newly scored, ${r.not_matured ?? 0} not yet matured, ${r.awaiting_price ?? 0} awaiting prices.`;
  });
  s2.runs = await api("/predictions/runs?limit=40");
  if (!s2.runs.length) { $("#view").innerHTML = '<div class="card missing">No predictions yet — click “Run predictions”.</div>'; return; }
  if (!s2.runId || !s2.runs.some((r) => r.run_id === s2.runId)) s2.runId = s2.runs[0].run_id;
  $("#view").innerHTML = `
    <div class="warning" role="note"><b>These are model estimates, not guaranteed outcomes.</b> The model is <b>experimental</b>: it has not beaten a no-change forecast out-of-sample (see Backtesting).</div>
    <div class="filters">
      <label>Run <select id="p-runsel">${s2.runs.map((r) => `<option value="${r.run_id}">${fmtTime(r.as_of)} · target ${r.target_session || "n/a"} · ${esc(r.model_version)}</option>`).join("")}</select></label>
    </div>
    <div id="p-meta"></div>
    <div class="section-title"><h2>Model predictions</h2><span class="muted small">authoritative numbers from System 2</span></div>
    <div class="card">
      <div class="filters">
        <label>Company <select id="p-company"><option value="">All</option>${state.companies.map((x) => `<option value="${esc(x.ticker)}">${esc(x.name)}</option>`).join("")}</select></label>
        <label>Horizon <select id="p-horizon"><option value="today" selected>Next session</option><option value="month_end">Month-ends</option><option value="quarter_end">Quarter-ends</option><option value="">All horizons</option></select></label>
        <label>Target date <select id="p-date"><option value="">All</option></select></label>
      </div>
      <div class="scroll"><table id="t-pred"></table></div>
      <p class="muted small">Estimated price = previous close × (1 + predicted % / 100). ±1σ = historical-volatility band for the horizon. “Relative” sentiment = news since the previous 09:15 IST cut-off minus the company’s usual tone.</p>
    </div>
    <div class="section-title"><h2>LLM validation</h2><span class="muted small">review only — never changes the predictions above</span></div>
    <div id="p-reviews"></div>
    <div class="section-title"><h2>Live track record</h2><span class="muted small">predictions whose target session has closed</span></div>
    <div id="p-live"></div>`;
  $("#p-runsel").value = s2.runId;
  $("#p-runsel").onchange = async (e) => { s2.runId = e.target.value; await loadRun(); };
  ["#p-company", "#p-horizon", "#p-date"].forEach((id) => ($(id).onchange = renderPredictions));
  await Promise.all([loadRun(), loadLive()]);
}
async function loadRun() {
  s2.data = await api(`/predictions?run_id=${encodeURIComponent(s2.runId)}&limit=2000`);
  const items = s2.data.items || [], p0 = items[0];
  const today = items.filter((p) => p.horizon_type === "today");
  const flags = {}; today.forEach((p) => { flags[p.data_quality_flag] = (flags[p.data_quality_flag] || 0) + 1; });
  const dirs = { UP: 0, DOWN: 0, NEUTRAL: 0 }; today.forEach((p) => { if (dirs[p.predicted_direction] !== undefined) dirs[p.predicted_direction]++; });
  $("#p-meta").innerHTML = p0 ? `<div class="tiles">
      ${tile("Target session", p0.horizon_type === "today" || today.length ? esc(today[0]?.target_date || p0.target_date) : "n/a", today[0] ? esc(today[0].horizon_label) : "")}
      ${tile("Prediction cut-off", p0.prediction_cutoff ? `<span style="font-size:16px">${fmtTime(p0.prediction_cutoff)}</span>` : '<span class="missing">not recorded</span>', "news after this time is never used")}
      ${tile("Next-session calls", `<span style="font-size:16px">▲ ${dirs.UP} · ▼ ${dirs.DOWN} · ● ${dirs.NEUTRAL}</span>`, `${today.length} companies`)}
      ${tile("Data quality", Object.entries(flags).map(([f, n]) => `<div style="font-size:13px">${qualityBadge(f === "null" ? null : f)} ×${n}</div>`).join(""), `news policy: ${esc(p0.availability_policy || "n/a")}`)}
      ${tile("Model", `<span style="font-size:15px">${esc(p0.model_version)}</span>`, `universe ${esc(p0.universe_version || "n/a")}`)}
    </div>` : "";
  const dates = [...new Set(items.map((p) => p.target_date))].sort();
  const dsel = $("#p-date"), prev = dsel.value;
  dsel.innerHTML = '<option value="">All</option>' + dates.map((d) => `<option>${d}</option>`).join("");
  dsel.value = dates.includes(prev) ? prev : "";
  renderPredictions();
  renderReviews(today);
}
function renderPredictions() {
  const items = s2.data?.items || [];
  if (!items.length) { table($("#t-pred"), ["Predictions"], [`<tr><td class="missing">${esc(s2.data?.message || "No predictions")}</td></tr>`]); return; }
  const c = $("#p-company").value, h = $("#p-horizon").value, d = $("#p-date").value;
  const rows = items.filter((p) => (!c || p.ticker === c) && (!h || p.horizon_type === h) && (!d || p.target_date === d)).map((p) => {
    const missing = p.status !== "OK" ? `<div class="missing small">${esc(p.missing_reason)}</div>` : "";
    const contrib = Object.entries(p.contributions_pct || {}).map(([k, x]) => `${k.replace("_sentiment", "")} ${x >= 0 ? "+" : ""}${Number(x).toFixed(3)}`).join(" · ");
    const actual = p.actual ? `${signed(p.actual.actual_movement_percent)}% ${directionBadge(p.actual.actual_direction || "")}<div class="muted small">|error| ${num(p.actual.abs_error ?? Math.abs(p.actual.error))}</div>` : '<span class="muted small">pending (after 15:30 IST close)</span>';
    return `<tr><td><b>${esc(p.company)}</b><div class="muted small">${esc(p.ticker)}</div></td>
      <td>${esc(p.horizon_label)}<div class="muted small">${p.target_date} · ${p.trading_days_ahead} session${p.trading_days_ahead === 1 ? "" : "s"} ahead</div></td>
      <td class="num">${num(p.previous_close)}<div class="muted small">${p.previous_close_date ?? ""}</div></td>
      <td class="num">${relativeCell(p.features?.news_flow)}<div class="muted small">raw ${num(p.sentiment_score, 1)}</div></td>
      <td class="num"><b>${p.predicted_movement_percent === null ? '<span class="missing">n/a</span>' : signed(p.predicted_movement_percent) + "%"}</b><div class="muted small">±${num(p.uncertainty_1sigma_pct)}% 1σ</div></td>
      <td class="num">${num(p.estimated_price)}<div class="muted small">estimate</div></td>
      <td>${directionBadge(p.predicted_direction)}${missing}<div>${qualityBadge(p.data_quality_flag)}</div><div class="muted small">${esc(contrib)}</div></td>
      <td>${actual}</td></tr>`;
  });
  table($("#t-pred"), ["Company", "Horizon", { t: "Previous close", num: 1 }, { t: "Sentiment (relative)", num: 1 }, { t: "Predicted move", num: 1 }, { t: "Est. price", num: 1 }, "Direction · data quality · drivers", "Actual"], rows, "stack");
}
function renderReviews(today) {
  const cfg = state.config;
  const li = (title, xs) => (xs && xs.length ? `<div class="small"><b>${title}</b><ul>${xs.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>` : "");
  const cards = today.map((p) => {
    const v = p.validations[p.validations.length - 1];
    if (!v) {
      const msg = cfg.llm_provider === "openai" && cfg.llm_configured === false ? "OpenAI validation not configured." : "Not validated yet — use “Validate”.";
      return `<div class="card review"><h3>${esc(p.company)} <span class="muted small">${esc(p.ticker)}</span></h3><p class="missing small">${msg}</p></div>`;
    }
    const who = `${esc(PROVIDER_LABEL[v.provider] || v.provider)} · ${esc(v.model)} · ${fmtTime(v.created_at)}`;
    if (v.request_status === "not_configured" || v.validation_status === "not_performed")
      return `<div class="card review"><h3>${esc(p.company)} ${reviewBadge("not_performed")}</h3><p class="small">${esc(v.provider === "openai" ? "OpenAI validation not configured." : v.reasoning)}</p><div class="muted small">${who}</div></div>`;
    return `<div class="card review"><h3>${esc(p.company)} ${reviewBadge(v.validation_status)}${v.cutoff_check ? `<span class="chip">cut-off check: ${esc(v.cutoff_check)}</span>` : ""}</h3>
      <div class="small">Reviewing: ${directionBadge(p.predicted_direction)} ${signed(p.predicted_movement_percent)}% for ${esc(p.target_date)}</div>
      <p class="small">${esc(v.reasoning)}</p>
      <details><summary class="small">Details (${(v.supporting_factors || []).length} supporting · ${(v.contradicting_factors || v.inconsistencies || []).length} contradicting · ${(v.data_quality_issues || []).length} data-quality issues)</summary>
        ${li("Supporting", v.supporting_factors)}${li("Contradicting", v.contradicting_factors || v.inconsistencies)}
        ${li("Data-quality issues", v.data_quality_issues)}${li("Missing information", v.missing_information)}
        ${v.error_message ? `<div class="muted small">error: ${esc(v.error_message)}</div>` : ""}</details>
      <div class="muted small">${who}</div></div>`;
  });
  $("#p-reviews").innerHTML = `<div class="grid2">${cards.join("")}</div>`;
}
async function loadLive() {
  const m = await api("/predictions/live-metrics");
  if (!m.n) { $("#p-live").innerHTML = '<div class="card small missing">No live prediction has matured yet. Outcomes are attached after each session’s 15:30 IST close (scheduled job, or “Score outcomes”).</div>'; return; }
  const bc = m.baseline_zero_change || {};
  $("#p-live").innerHTML = `<div class="tiles">
    ${tile("Scored predictions", m.n, "all horizons")}
    ${tile("MAE (pct-pts)", num(m.mae?.value, 3), `no-change ${num(bc.mae, 3)}`)}
    ${tile("Sign accuracy", pct(m.sign_accuracy?.value), `n=${m.sign_accuracy?.n ?? 0}`)}
    ${tile("Directional accuracy", pct(m.directional_accuracy?.value), `n=${m.directional_accuracy?.n ?? 0}`)}
    ${tile("Bias", num(m.bias, 3), "mean predicted − actual")}</div>`;
}

// ============================================================ SYSTEM 3
const s3 = { runId: null };
async function pageSystem3() {
  setHead("System 3 · Backtesting & fine-tuning", "Replays Systems 1 and 2 on historical news and prices with no look-ahead, measures accuracy against four baselines, tunes parameters on train / validation / untouched test and writes human- and AI-readable reports.");
  $("#view").innerHTML = `
    <div class="warning" role="note"><b>Historical performance does not guarantee future performance.</b> Most historical news was collected after the fact; see Data quality for how much was demonstrably available before each cut-off.</div>
    <div class="card"><h3>Run a backtest</h3>
      <form id="bt-form" class="filters">
        <label>Start <input type="date" name="start" value="2026-05-01" required></label>
        <label>End <input type="date" name="end" value="2026-09-30" required></label>
        <label>News <select name="availability_mode"><option value="provider_timestamp">all (provider timestamps)</option><option value="exact_timestamp">exact timestamps only</option><option value="verified">verified pre-market only</option></select></label>
        <label><input type="checkbox" name="tune"> Tune parameters</label>
        <label>Trials <input type="number" name="n_trials" value="30" min="1" max="500" style="width:80px"></label>
        <button type="submit" class="primary">Start backtest</button>
      </form>
      <p class="muted small">Tuning splits the period chronologically 60 / 20 / 20 (train / validation / untouched test). A tuned set is only proposed if it beats the baselines significantly on unseen data, and it is never activated automatically.</p></div>
    <div class="filters"><label>Backtest run <select id="bt-run"></select></label><span id="bt-links" class="small"></span></div>
    <div id="bt-body"></div>
    <div class="section-title"><h2>Out-of-sample experiments</h2><span class="muted small">before/after and fitted models on train May–Jul · validation Aug · test Sep</span></div>
    <div id="exp-body" class="skeleton">Loading…</div>
    <div class="section-title"><h2>Reports library</h2><span class="muted small">Markdown for people · JSON for AI</span></div>
    <div class="card"><div class="scroll" id="rep-body"></div></div>`;
  $("#bt-form").onsubmit = async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    const body = { start: f.get("start"), end: f.get("end"), tune: f.get("tune") === "on", n_trials: +f.get("n_trials"), availability_mode: f.get("availability_mode") };
    try { const r = await api("/backtest", { method: "POST", body: JSON.stringify(body) }); status(`Backtest ${r.run_id} started — this can take several minutes…`); pollRun(r.run_id); }
    catch (err) { status("Error: " + err.message, true); }
  };
  $("#bt-run").onchange = (e) => { s3.runId = e.target.value; renderBacktest(s3.runId); };
  await Promise.all([loadBacktests(), loadExperiments(), loadReports()]);
}
async function loadBacktests() {
  const runs = await api("/backtest/results");
  const sel = $("#bt-run");
  sel.innerHTML = runs.map((r) => `<option value="${r.run_id}">${r.run_id} · ${r.kind} · ${r.status} · ${r.start}→${r.end}${r.n ? " · n=" + r.n : ""} · ${esc(r.param_version)}</option>`).join("");
  if (!runs.length) { $("#bt-body").innerHTML = '<div class="card missing">No backtests yet.</div>'; return; }
  s3.runId = s3.runId && runs.some((r) => r.run_id === s3.runId) ? s3.runId : (runs.find((r) => r.status === "completed") || runs[0]).run_id;
  sel.value = s3.runId;
  await renderBacktest(s3.runId);
}
async function pollRun(id) {
  const r = await api(`/backtest/results/${id}`);
  if (state.page !== "/system-3") return;
  if (r.status === "pending" || r.status === "running") { status(`Backtest ${id}: ${r.status}…`); setTimeout(() => pollRun(id), 4000); return; }
  status(`Backtest ${id}: ${r.status}${r.error ? " — " + r.error : ""}`, r.status !== "completed");
  s3.runId = id;
  await loadBacktests(); await loadReports();
}
const mrow = (name, m) => `<tr><td>${esc(name)}</td><td class="num">${m.n}</td><td class="num">${pct(m.directional_accuracy?.value)}</td>
  <td class="num">${pct(m.sign_accuracy?.value)} <span class="muted">(${m.sign_accuracy?.n ?? 0})</span></td>
  <td class="num">${num(m.mae?.value, 3)}</td><td class="num">${num(m.rmse?.value, 3)}</td><td class="num">${num(m.correlation?.value, 3)}</td>
  <td class="num">${num(m.baseline_zero_change?.mae, 3)}</td><td class="num">${pct(m.mae_skill_vs_zero)}</td></tr>`;
const mhead = ["Group", { t: "N", num: 1 }, { t: "Directional", num: 1 }, { t: "Sign acc. (n)", num: 1 }, { t: "MAE", num: 1 }, { t: "RMSE", num: 1 }, { t: "Corr.", num: 1 }, { t: "No-change MAE", num: 1 }, { t: "MAE skill", num: 1 }];
const BASELINE_LABELS = { no_change: "No change", historical_mean: "Historical mean (all stocks)", company_mean: "Company historical mean", rolling_mean_20d: "Rolling mean (20 sessions)" };
function dataQualityCard(dq, mode) {
  if (!dq) return `<div class="card"><h3>Data quality</h3><p class="missing small">Not recorded for this (older) run.</p></div>`;
  const st = dq.availability_status, tot = dq.total_news_articles || 0;
  const share = (n) => (tot ? ` (${((n / tot) * 100).toFixed(1)}%)` : "");
  return `<div class="card"><h3>Data quality · news mode: ${esc(mode || "provider_timestamp")}</h3><div class="tiles">
    ${tile("News articles", tot.toLocaleString("en-IN"), `duplicate rate ${pct(dq.duplicate_rate)}`)}
    ${tile("Verified pre-market", st.verified_pre_market, "collected before the cut-off" + share(st.verified_pre_market))}
    ${tile("Post-event", st.collected_after_event, "collected after the predicted session" + share(st.collected_after_event))}
    ${tile("Unknown availability", st.unknown, "date-only timestamps" + share(st.unknown))}
    ${tile("Price coverage", pct(dq.price_coverage), `${dq.price_sessions} sessions`)}</div></div>`;
}
function baselineCard(m) {
  const bc = m?.baseline_comparison;
  if (!bc?.baselines) return "";
  const row = (name, s, skill) => `<tr><td>${esc(name)}</td><td class="num">${num(s.mae, 3)}</td><td class="num">${num(s.rmse, 3)}</td>
    <td class="num">${pct(s.directional_accuracy)}</td><td class="num">${pct(s.sign_accuracy?.value)} <span class="muted">(${s.sign_accuracy?.n ?? 0})</span></td>
    <td class="num">${num(s.bias, 3)}</td><td class="num">${skill}</td></tr>`;
  const rows = [row("Model", bc.model, "—"), ...Object.entries(bc.baselines).map(([k, s]) => row(BASELINE_LABELS[k] || k, s, pct(s.model_mae_skill)))];
  return `<div class="card"><h3>Model vs baselines (next session, same ${bc.n} samples)</h3><div class="scroll">${tableHtml(["Forecast", { t: "MAE", num: 1 }, { t: "RMSE", num: 1 }, { t: "Directional", num: 1 }, { t: "Sign acc. (n)", num: 1 }, { t: "Bias", num: 1 }, { t: "Model MAE skill vs it", num: 1 }], rows, "", "wide")}</div>
    <p class="muted small">Skill &gt; 0 means the model's MAE is lower than that baseline's. Baselines use closes before the prediction date only.</p></div>`;
}
function distributionCard(m) {
  const cm = m?.confusion_matrix, pd = m?.prediction_distribution;
  if (!cm || !pd) return "";
  const D = ["UP", "DOWN", "NEUTRAL"];
  const rows = D.map((p) => `<tr><td>${p}</td>${D.map((a) => `<td class="num ${p === a ? "hit" : ""}">${cm[p][a]}</td>`).join("")}<td class="num">${D.reduce((s, a) => s + cm[p][a], 0)}</td></tr>`);
  const dist = (k) => D.map((d) => `${d} ${pd[k][d].count} (${pct(pd[k][d].share)})`).join(" · ");
  return `<div class="card"><h3>Prediction distribution (next session)</h3><p class="small">Predicted: ${dist("predicted")}<br>Actual: ${dist("actual")}</p>
    <div class="scroll"><table class="cm wide"><thead><tr><th>Predicted \\ Actual</th>${D.map((d) => `<th class="num">${d}</th>`).join("")}<th class="num">Total</th></tr></thead><tbody>${rows.join("")}</tbody></table></div></div>`;
}
async function renderBacktest(id) {
  const r = await api(`/backtest/results/${id}`);
  const body = $("#bt-body");
  $("#bt-links").innerHTML = r.report ? `<a href="${API}/backtest/results/${id}/report.md" target="_blank">Human report (Markdown)</a> · <a href="${API}/backtest/results/${id}" target="_blank">AI report (JSON)</a>` : "";
  if (!r.report) { body.innerHTML = `<div class="card">Status: ${esc(r.status)} ${r.error ? "— " + esc(r.error) : ""}</div>`; return; }
  const rep = r.report, m = rep.metrics;
  if (!m.n) { body.innerHTML = `<div class="card missing">No evaluable predictions: ${esc(m.note)}</div>`; return; }
  const companyRows = Object.entries(rep.company_results).flatMap(([t, byH]) => Object.entries(byH).map(([h, mm]) => mrow(`${t} · ${h}`, mm)));
  const changes = rep.parameter_changes.map((c) => `<tr><td>${esc(c.parameter_name)}</td><td class="num">${esc(c.old_value)}</td><td class="num">${esc(c.new_value)}</td><td class="small">${esc(c.reason)}</td></tr>`);
  const cand = rep.parameter_changes[0]?.to_version;
  const t = rep.tuning;
  body.innerHTML = `
    <div class="tiles">
      ${tile("Backtest period", `<span style="font-size:15px">${rep.backtest_period.start} → ${rep.backtest_period.end}</span>`, `${rep.stocks_tested.length} stocks · ${rep.number_of_news_events.toLocaleString("en-IN")} news`)}
      ${tile("Predictions", rep.total_predictions.toLocaleString("en-IN"), rep.horizon_types.join(", "))}
      ${tile("Directional accuracy", pct(m.directional_accuracy.value), `n=${m.directional_accuracy.n}`)}
      ${tile("MAE (pct-pts)", num(m.mae.value, 3), `no-change ${num(m.baseline_zero_change.mae, 3)}`)}
      ${tile("RMSE", num(m.rmse.value, 3), `n=${m.rmse.n}`)}
      ${tile("Model", `<span style="font-size:14px">${esc(rep.model_version)}</span>`, `news: ${esc(rep.availability_mode)}`)}
    </div>
    ${dataQualityCard(rep.data_quality, rep.availability_mode)}
    ${baselineCard(rep.metrics_by_horizon?.today)}
    ${distributionCard(rep.metrics_by_horizon?.today)}
    <div class="grid2">
      <div class="card"><h3>Predicted vs actual (next session, %)</h3><div class="chart"><canvas id="c-scatter"></canvas></div></div>
      <div class="card"><h3>MAE by company (next session)</h3><div class="chart"><canvas id="c-mae"></canvas></div></div>
    </div>
    <div class="card"><h3>By horizon</h3><div class="scroll">${tableHtml(mhead, Object.entries(rep.metrics_by_horizon).map(([h, mm]) => mrow(h, mm)), "", "wide")}</div></div>
    <div class="card"><h3>Company-wise results</h3><div class="scroll">${tableHtml(mhead, companyRows, "", "wide")}</div></div>
    <div class="card"><h3>By month (next session)</h3><div class="scroll">${tableHtml(mhead, Object.entries(rep.monthly_results?.today || {}).map(([mo, mm]) => mrow(mo, mm)), "", "wide")}</div></div>
    <div class="card"><h3>Fine-tuning ${cand ? `<button id="activate" data-v="${esc(cand)}">Activate ${esc(cand)}</button>` : ""}</h3>
      ${t ? `<p class="small">${esc(t.reason)}<br><span class="muted">Train ${t.train_dates[0]} → ${t.train_dates[1]} · validation ${t.validation_dates[0]} → ${t.validation_dates[1]}${t.test_dates && t.test_dates[2] ? ` · untouched test ${t.test_dates[0]} → ${t.test_dates[1]}` : ""}</span></p>` : '<p class="muted small">This run did not tune parameters.</p>'}
      <div class="scroll">${tableHtml(["Parameter", { t: "Old", num: 1 }, { t: "New", num: 1 }, "Reason"], changes, "", "stack")}</div></div>
    <div class="card"><h3>Largest misses</h3><div class="scroll">${tableHtml(["Date", "Ticker", "Horizon", { t: "Pred %", num: 1 }, { t: "Actual %", num: 1 }, { t: "Error", num: 1 }, "Evidence"],
      rep.failure_cases.map((f) => `<tr><td>${f.as_of_date}</td><td>${esc(f.ticker)}</td><td>${esc(f.horizon)}</td><td class="num">${f.predicted_pct.toFixed(2)}</td><td class="num">${f.actual_pct.toFixed(2)}</td><td class="num">${f.error_pct_points.toFixed(2)}</td><td class="small">${f.evidence.map(esc).join("<br>") || "—"}</td></tr>`), "", "stack")}</div></div>
    <div class="card"><h3>Observations</h3><ul>${rep.observations.map((o) => `<li class="small">${esc(o)}</li>`).join("")}</ul></div>`;
  $("#activate")?.addEventListener("click", async (e) => {
    if (!confirm(`Activate parameter set ${e.target.dataset.v} for live predictions?`)) return;
    try { await api(`/parameters/${e.target.dataset.v}/activate`, { method: "POST" }); status(`Activated ${e.target.dataset.v}`); } catch (err) { status("Error: " + err.message, true); }
  });
  const pts = rep.predicted_vs_actual.filter((p) => p.horizon_type === "today").slice(0, 3000);
  const lim = Math.max(1, ...pts.map((p) => Math.max(Math.abs(p.predicted), Math.abs(p.actual))));
  chart("c-scatter", { type: "scatter",
    data: { datasets: [{ data: pts.map((p) => ({ x: p.predicted, y: p.actual, t: p.ticker, d: p.as_of_date })), backgroundColor: css("--s1") + "99", pointRadius: 3, pointHoverRadius: 6 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.raw.t} ${c.raw.d}: predicted ${c.raw.x.toFixed(2)}%, actual ${c.raw.y.toFixed(2)}%` } } },
      scales: { x: { title: { display: true, text: "Predicted %" }, grid: { color: css("--grid") } }, y: { title: { display: true, text: "Actual %" }, min: -lim, max: lim, grid: { color: css("--grid") } } } } });
  const maes = Object.entries(rep.company_results).map(([tk, byH]) => [tk, byH.today?.mae?.value, byH.today?.baseline_zero_change?.mae]).filter((x) => x[1] !== undefined);
  chart("c-mae", { type: "bar",
    data: { labels: maes.map((x) => x[0]), datasets: [
      { label: "Model MAE", data: maes.map((x) => x[1]), backgroundColor: css("--s1"), borderRadius: 4, maxBarThickness: 22 },
      { label: "No-change MAE", data: maes.map((x) => x[2]), backgroundColor: css("--s2"), borderRadius: 4, maxBarThickness: 22 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "top", labels: { boxWidth: 10 } } }, scales: { y: { beginAtZero: true, grid: { color: css("--grid") } }, x: { grid: { display: false } } } } });
}
async function loadExperiments() {
  const reps = (await api("/reports")).filter((r) => r.kind === "experiment" && r.format === "json");
  if (!reps.length) { $("#exp-body").outerHTML = '<div id="exp-body" class="card missing">No experiments yet (python -m backend.cli experiment …).</div>'; return; }
  $("#exp-body").outerHTML = `<div id="exp-body"><div class="filters"><label>Experiment <select id="exp-sel">${reps.map((r) => `<option value="${r.name}">${r.name.replace(".json", "")} · ${fmtTime(r.modified)}</option>`).join("")}</select></label><span id="exp-links" class="small"></span></div><div id="exp-view"></div></div>`;
  $("#exp-sel").onchange = (e) => renderExperiment(e.target.value);
  await renderExperiment(reps[0].name);
}
async function renderExperiment(name) {
  const x = await api(`/reports/${name}`);
  $("#exp-links").innerHTML = `<a href="${API}/reports/${name.replace(".json", ".md")}" target="_blank">Markdown</a> · <a href="${API}/reports/${name}" target="_blank">JSON</a>`;
  const P = x.periods, part = P.test?.rows ? "test" : "validation";
  const row = (name2, s) => `<tr><td>${esc(name2)}</td><td class="num">${s?.n ?? 0}</td><td class="num">${num(s?.mae, 3)}</td><td class="num">${num(s?.rmse, 3)}</td><td class="num">${pct(s?.sign_accuracy?.value)} <span class="muted">(${s?.sign_accuracy?.n ?? 0})</span></td><td class="num">${num(s?.bias, 3)}</td></tr>`;
  const rows = [...Object.entries(x.reference_models || {}).map(([n, r]) => row(n, r[part])),
    row(`selected ridge model (${x.selected_feature_set})`, x.models[x.selected_feature_set]?.[part]),
    ...Object.entries(x.baselines[part] || {}).map(([b, s]) => row("baseline: " + (BASELINE_LABELS[b] || b), s))];
  const sig = Object.entries(x.selected_beats_baselines?.[part] || {}).map(([b, v]) => `<tr><td>${esc(BASELINE_LABELS[b] || b)}</td><td>${v.mae_lower ? "yes" : "no"}</td><td class="num">${num(v.p_value_improvement, 3)}</td><td>${v.significant ? pill("significant", "var(--good)") : pill("not significant", "var(--muted)")}</td></tr>`);
  const per = (k) => `${P[k].start || "—"} → ${P[k].end || "—"} (${P[k].sessions || 0} sessions)`;
  $("#exp-view").innerHTML = `<div class="tiles">
      ${tile("Train", `<span style="font-size:14px">${per("train")}</span>`, "parameters fitted here")}
      ${tile("Validation", `<span style="font-size:14px">${per("validation")}</span>`, "model choice made here")}
      ${tile("Untouched test", `<span style="font-size:14px">${per("test")}</span>`, "scored once, can only veto")}
      ${tile("Result", x.candidate_created ? pill("candidate stored", "var(--good)") : pill("no candidate", "var(--serious)"), esc(x.recommendation))}
    </div>
    <div class="grid2">
      <div class="card"><h3>${part === "test" ? "Untouched test period" : "Validation period"} — all forecasts on the same rows</h3><div class="scroll">${tableHtml(["Forecast", { t: "N", num: 1 }, { t: "MAE", num: 1 }, { t: "RMSE", num: 1 }, { t: "Sign acc. (n)", num: 1 }, { t: "Bias", num: 1 }], rows, "", "wide")}</div></div>
      <div class="card"><h3>Is the selected model's improvement real? (${part})</h3><div class="scroll">${tableHtml(["vs baseline", "MAE lower?", { t: "p-value", num: 1 }, "Verdict"], sig, "", "stack")}</div>
        <p class="muted small">Paired, date-clustered one-sided t-test on absolute errors. Only p &lt; 0.05 on both validation and test counts as skill.</p></div>
    </div>`;
}
async function loadReports() {
  const reps = await api("/reports");
  const by = {};
  reps.forEach((r) => { const k = r.name.replace(/\.(md|json)$/, ""); (by[k] = by[k] || { kind: r.kind, modified: r.modified, files: {} }).files[r.format] = r.url; });
  const rows = Object.entries(by).map(([k, v]) => `<tr><td><b>${esc(k)}</b></td><td>${esc(v.kind)}</td><td>${fmtTime(v.modified)}</td>
    <td>${v.files.md ? `<a href="${v.files.md}" target="_blank">Markdown</a>` : ""}${v.files.md && v.files.json ? " · " : ""}${v.files.json ? `<a href="${v.files.json}" target="_blank">JSON</a>` : ""}</td></tr>`);
  $("#rep-body").innerHTML = tableHtml(["Report", "Type", "Written", "Open"], rows, "", "stack");
}

// ============================================================ SETTINGS
async function pageSettings() {
  setHead("Settings", "Index universe, model parameter sets, news providers, LLM validation and scheduled jobs.");
  const [uni, params, sched, audit] = await Promise.all([api("/universe"), api("/parameters"), api("/schedule"), api("/news/audit")]);
  const c = state.config;
  const memberRows = (ms) => ms.map((m) => `<tr><td><b>${esc(m.ticker)}</b></td><td>${esc(m.company)}</td><td class="num">${num(m.weight, 2)}</td><td>${esc(m.effective_from)}${m.effective_to ? " → " + esc(m.effective_to) : " → current"}</td><td>${esc(m.source_date || "")}</td></tr>`);
  const hist = uni.history.map((v) => `<details class="card"><summary><b>${esc(v.version)}</b> · ${esc(v.effective_from)}${v.effective_to ? " → " + esc(v.effective_to) : ""} · ${v.status === "confirmed" ? pill("confirmed", "var(--good)") : pill(v.status, "var(--serious)")}</summary>
      <p class="small muted">${esc(v.source)}</p><div class="scroll">${tableHtml(["Ticker", "Company", { t: "Weight %", num: 1 }, "Effective", "Source date"], memberRows(v.members), "", "stack")}</div></details>`).join("");
  const paramRows = params.map((p) => `<tr><td><b>${esc(p.version)}</b></td><td>${p.status === "active" ? pill("active", "var(--good)") : pill(p.status, "var(--muted)")}</td>
      <td>${esc(p.params?.system2?.model || "factor")} · ${esc(p.params?.system2?.sentiment_input || "raw")} sentiment</td><td>${fmtTime(p.created_at)}</td>
      <td class="small">${esc(p.notes || "")}</td><td>${p.status === "active" ? "" : `<button data-activate="${esc(p.version)}">Activate</button>`}</td></tr>`);
  const prov = Object.entries(audit.by_provider).map(([p, v]) => `<tr><td><b>${esc(p)}</b></td><td>${pill("active", "var(--good)")}</td><td class="num">${v.articles.toLocaleString("en-IN")}</td><td class="small">${esc(v.timestamp_quality)}</td></tr>`)
    .concat(c.news_providers_skipped.map((p) => `<tr><td><b>${esc(p.split(" ")[0])}</b></td><td>${pill(p.includes("(") ? p.slice(p.indexOf("(") + 1, -1) : "skipped", "var(--muted)")}</td><td class="num">0</td><td class="small muted">Add its API key to .env to enable.</td></tr>`));
  const jobs = sched.jobs.map((j) => `<tr><td><b>${esc(j.task)}</b></td><td>${j.registered ? pill(j.status || "registered", "var(--good)") : pill("not registered", "var(--serious)")}</td><td>${esc(j.next_run || "—")}</td><td>${esc(j.last_run || "—")}</td><td>${esc(j.last_result ?? "—")}</td></tr>`);
  const auth = await api("/auth");
  $("#view").innerHTML = `
    ${auth.actions_require_token ? `<div class="section-title"><h2>Admin access</h2><span class="muted small">needed for actions (collect, predict, validate, backtests, activation); viewing is public</span></div>
    <div class="card"><div class="filters">
      <label>Admin token <input id="tok" type="password" autocomplete="off" placeholder="${getToken() ? "stored in this browser" : "not set"}"></label>
      <button id="tok-save" class="primary">Save</button><button id="tok-clear">Forget</button>
      <span class="small">${getToken() ? pill("token stored in this browser", "var(--good)") : pill("no token stored", "var(--muted)")}</span></div></div>` : ""}
    <div class="section-title"><h2>Top-10 NIFTY universe</h2><span class="muted small">active on ${esc(uni.as_of)} · weights are reference data, not live values</span></div>
    <div class="card"><h3>${esc(uni.active.version)} ${pill("active", "var(--good)")}</h3><p class="small muted">${esc(uni.active.source || "")}</p>
      <div class="scroll">${tableHtml(["Ticker", "Company", { t: "Weight %", num: 1 }, "Effective", "Source date"], memberRows(uni.active.members), "", "stack")}</div></div>
    ${uni.requires_confirmation.length ? `<div class="warning">Pending confirmation: ${uni.requires_confirmation.map((p) => esc(p.version)).join(", ")} (CLI: <code>python -m backend.cli universe --confirm VERSION</code>)</div>` : ""}
    <h3>Version history</h3>${hist}
    <div class="section-title"><h2>Model parameter sets</h2><span class="muted small">activation is always explicit</span></div>
    <div class="card"><div class="scroll">${tableHtml(["Version", "Status", "Model", "Created", "Notes", ""], paramRows, "", "stack")}</div></div>
    <div class="section-title"><h2>News providers</h2></div>
    <div class="card"><div class="scroll">${tableHtml(["Provider", "Status", { t: "Articles", num: 1 }, "Timestamp quality"], prov, "", "stack")}</div></div>
    <div class="section-title"><h2>LLM validation</h2></div>
    <div class="tiles">
      ${tile("Provider", `<span style="font-size:16px">${esc(PROVIDER_LABEL[c.llm_provider] || c.llm_provider)}</span>`, "LLM_PROVIDER in .env")}
      ${tile("Model", `<span style="font-size:16px">${esc(c.llm_model || "not set")}</span>`, "OPENAI_MODEL in .env")}
      ${tile("Status", c.llm_configured === false ? pill("not configured", "var(--serious)") : pill("configured", "var(--good)"), esc(c.llm_status || "API key stored in .env (never shown)"))}
    </div>
    <div class="section-title"><h2>Scheduled jobs</h2><span class="muted small">create / remove: <code>scripts\\schedule_windows.ps1 [-Remove]</code></span></div>
    <div class="card"><div class="scroll">${tableHtml(["Job", "Status", "Next run", "Last run", "Last result"], jobs, "", "stack")}</div></div>`;
  if ($("#tok-save")) {
    $("#tok-save").onclick = () => { const v = $("#tok").value.trim(); if (v) { setToken(v); status("Admin token saved in this browser."); reloadPage(); } };
    $("#tok-clear").onclick = () => { setToken(""); status("Admin token removed from this browser."); reloadPage(); };
  }
  document.querySelectorAll("[data-activate]").forEach((b) => (b.onclick = async () => {
    const v = b.dataset.activate;
    if (!confirm(`Activate parameter set ${v}? Live predictions will use it from the next run.`)) return;
    try { await api(`/parameters/${encodeURIComponent(v)}/activate`, { method: "POST" }); status(`Activated ${v}.`); await reloadPage(); }
    catch (e) { status("Error: " + e.message, true); }
  }));
}

// ============================================================ boot
async function sideFoot() {
  const c = state.config;
  $("#side-foot").innerHTML = `${pill(`Universe ${c.universe_version || "n/a"}`, "var(--s3)")}
    ${c.llm_configured === false ? pill("LLM validation not configured", "var(--serious)") : pill(`${PROVIDER_LABEL[c.llm_provider] || c.llm_provider} · ${c.llm_model || ""}`, "var(--good)")}
    ${pill(`News: ${c.news_providers_active.join(", ")}`, "var(--s1)")}
    <span>Times in ${esc(c.display_timezone)}</span>`;
}
(async function init() {
  try {
    state.config = await api("/config");
    state.tz = state.config.display_timezone;
    state.companies = await api("/companies");
    sideFoot();
  } catch (e) { status("Error loading configuration: " + e.message, true); }
  render();
})();
