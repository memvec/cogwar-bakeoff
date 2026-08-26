// Point this at the deployed backend -- everything else in this file reads
// from API_BASE, nothing else hardcodes a host.
const API_BASE = "http://127.0.0.1:8000";

// ============================================================ fetch helpers

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}${await errorDetail(res)}`);
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}${await errorDetail(res)}`);
  return res.json();
}

async function errorDetail(res) {
  try {
    const body = await res.json();
    return body.detail ? `: ${body.detail}` : "";
  } catch (e) {
    return "";
  }
}

// ============================================================ formatting / color utils

const SOURCE_COLORS = { telegram: "#5dade2", youtube_video: "#ec7063", channel: "#b48ee0" };
const fallbackColor = d3.scaleOrdinal(d3.schemeSet2);
function colorForSource(sourceType) {
  return SOURCE_COLORS[sourceType] || fallbackColor(sourceType);
}
function sourceTypeOf(authorId) {
  return (authorId || "").split(":")[0];
}
function stanceColor(v) {
  if (v > 0.03) return "var(--positive)";
  if (v < -0.03) return "var(--negative)";
  return "var(--neutral)";
}
function fmtSigned(v) {
  return (v >= 0 ? "+" : "") + v.toFixed(2);
}
function fmtNum(v) {
  return d3.format(",")(v);
}
function truncateLabel(name, max = 26) {
  if (!name) return "";
  return name.length > max ? name.slice(0, max - 1) + "…" : name;
}
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}
function formatDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}
function formatDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso).slice(0, 10);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}
function formatSpanDays(span) {
  if (!span || !span.first_seen || !span.last_seen) return "—";
  return Math.max(0, Math.round((new Date(span.last_seen) - new Date(span.first_seen)) / 86400000));
}

// ============================================================ view-stack navigation
//
// The whole app is one linear drill-down stack: ask a question (or open the
// network), then each click on an actor/stance/cluster pushes a new view on
// top. Breadcrumbs show the trail and can pop back to any point. This keeps
// "click to verify" a single consistent gesture everywhere instead of a
// different modal/panel per feature.

let viewStack = [];
let renderToken = 0;
let activeGraphRefresh = null;

function pushView(view) {
  viewStack.push(view);
  scheduleRender();
}

function scheduleRender() {
  renderToken += 1;
  const token = renderToken;
  const isCurrent = () => token === renderToken;

  // A graph node/edge tooltip has no chance to fire mouseleave when its
  // element is torn down by a navigation (vs. the mouse actually leaving
  // it), so it can otherwise stay stuck on-screen after leaving the graph.
  hideTooltip();

  renderBreadcrumbs();
  const view = viewStack[viewStack.length - 1];
  if (!view || view.type !== "graph") activeGraphRefresh = null;

  const container = d3.select("#resultsArea");
  if (!view) return renderHome(container);
  if (view.type === "ask") return renderAskView(container, view);
  if (view.type === "profile") return renderProfileView(container, view, isCurrent);
  if (view.type === "sources") return renderSourcesView(container, view, isCurrent);
  if (view.type === "cluster") return renderClusterView(container, view, isCurrent);
  if (view.type === "graph") return renderGraphView(container, view, isCurrent);
}

function renderBreadcrumbs() {
  const bc = d3.select("#breadcrumbs");
  if (!viewStack.length) {
    bc.classed("hidden", true).html("");
    return;
  }
  bc.classed("hidden", false).html("");
  bc.append("button")
    .attr("class", "crumb")
    .text("Home")
    .on("click", () => {
      viewStack = [];
      scheduleRender();
    });
  viewStack.forEach((view, i) => {
    bc.append("span").attr("class", "crumb-sep").text("›");
    const isLast = i === viewStack.length - 1;
    bc.append("button")
      .attr("class", `crumb ${isLast ? "current" : ""}`)
      .text(view.label)
      .on("click", () => {
        if (!isLast) {
          viewStack = viewStack.slice(0, i + 1);
          scheduleRender();
        }
      });
  });
}

// ============================================================ shared state-panel helpers

function renderHome(container) {
  container.html("");
  container.append("div").attr("class", "state-panel home").html(
    `<p class="state-title">Ask a question above, or try one of the examples.</p>
     <p>Every answer traces back to real posts -- click through any actor, stance, or coordination cluster to see (and open) the original content.</p>`
  );
}

function renderSkeleton(container) {
  const wrap = container.append("div");
  [55, 85, 85, 40].forEach((w) => wrap.append("div").attr("class", "skeleton-line").style("width", `${w}%`));
}

function renderErrorState(container, message, err) {
  container.append("div").attr("class", "error-panel").text(`${message}${err ? " (" + err.message + ")" : ""}`);
}

function addMeta(container, value, label) {
  const item = container.append("div").attr("class", "profile-meta-item");
  item.append("div").attr("class", "profile-meta-value").text(value);
  item.append("div").attr("class", "profile-meta-label").text(label);
}

// ============================================================ ask view (the hero result)

function renderAskView(container, view) {
  container.html("");
  const { response } = view.params;
  const { summary, intent, result, error } = response;

  const banner = container.append("div").attr("class", `answer-summary ${error ? "unsupported" : ""}`);
  banner.append("div").attr("class", "dot");
  banner.append("p").text(summary);

  if (error || !result) return;

  const qType = intent && intent.query_type;
  if (qType === "consistent_actors") renderConsistentActors(container, result);
  else if (qType === "author_stance_on_entity") renderAuthorStanceCard(container, result);
  else if (qType === "entity_timeline") renderTimeline(container, result);
  else if (qType === "author_profile") renderProfileContent(container, result);
  else if (qType === "topic_coordination") renderTopicCoordination(container, result);
}

// ---------- consistent_actors -> ranked actor list ----------

function renderConsistentActors(container, result) {
  container
    .append("div")
    .attr("class", "section-title")
    .text(`Consistent actors -- ${result.canonical_name} (min. ${result.min_volume} stance-bearing items)`);
  const cols = container.append("div").attr("class", "actor-columns");

  const posCol = cols.append("div");
  posCol
    .append("div")
    .attr("class", "actor-column-title")
    .html('<span class="swatch-dot" style="background:var(--positive)"></span> Positive');
  renderActorList(posCol, result.consistently_positive);

  const negCol = cols.append("div");
  negCol
    .append("div")
    .attr("class", "actor-column-title")
    .html('<span class="swatch-dot" style="background:var(--negative)"></span> Negative');
  renderActorList(negCol, result.consistently_negative);
}

function renderActorList(container, actors) {
  if (!actors.length) {
    container.append("div").attr("class", "empty-column").text("No author clears the ranking bar.");
    return;
  }
  const list = container.append("div").attr("class", "actor-list");
  const rows = list
    .selectAll(".actor-row")
    .data(actors)
    .join("div")
    .attr("class", "actor-row")
    .on("click", (event, d) => pushView({ type: "profile", label: d.author_id, params: { authorId: d.author_id } }));

  rows.append("span").attr("class", "source-dot").style("background", (d) => colorForSource(sourceTypeOf(d.author_id)));
  rows.append("span").attr("class", "actor-id").text((d) => d.author_id);
  rows.append("span").attr("class", "actor-metric emphasis").text((d) => `net ${fmtSigned(d.net_stance)}`);
  rows.append("span").attr("class", "actor-metric").text((d) => `consistency ${d.stance_consistency.toFixed(2)}`);
  rows.append("span").attr("class", "actor-metric").text((d) => `vol ${fmtNum(d.volume)}`);
  rows.append("span").attr("class", "row-affordance").html("&rsaquo;");
}

// ---------- author_stance_on_entity -> single stance card ----------

function renderAuthorStanceCard(container, result) {
  const { author, entity, stance } = result;
  const card = container.append("div").attr("class", "single-stance-card").on("click", () => {
    if (!stance) return;
    pushView({
      type: "sources",
      label: `${author.display_name} → ${entity.canonical_name}`,
      params: { authorId: author.author_id, entityId: entity.entity_id, authorLabel: author.display_name, entityLabel: entity.canonical_name },
    });
  });

  card
    .append("div")
    .attr("class", "single-stance-title")
    .html(
      `<span class="source-dot" style="background:${colorForSource(author.source_type)}"></span> ${escapeHtml(author.display_name)} <span class="arrow">→</span> ${escapeHtml(entity.canonical_name)}`
    );

  if (!stance) {
    card.append("p").style("color", "var(--text-faint)").style("font-size", "13px").text("No recorded stance edges toward this entity.");
    return;
  }

  const track = card.append("div").attr("class", "stance-bar-track");
  track.append("div").attr("class", "stance-bar-zero");
  track
    .append("div")
    .attr("class", `stance-bar ${stance.net_stance >= 0 ? "positive" : "negative"}`)
    .style("width", `${Math.min(Math.abs(stance.net_stance), 1) * 50}%`)
    .style("height", "10px");

  const pills = card.append("div").attr("class", "stat-pills");
  addPill(pills, fmtSigned(stance.net_stance), "net stance");
  addPill(pills, stance.stance_consistency.toFixed(2), "consistency");
  addPill(pills, fmtNum(stance.volume), "items");
  addPill(pills, `${stance.positive_count} / ${stance.negative_count} / ${stance.neutral_count}`, "pos / neg / neu");

  card
    .append("div")
    .attr("class", "secondary-link")
    .text("View full profile →")
    .on("click", (event) => {
      event.stopPropagation();
      pushView({ type: "profile", label: author.display_name, params: { authorId: author.author_id } });
    });

  card.append("p").style("margin", "12px 0 0").style("font-size", "12px").style("color", "var(--text-faint)").text("Click card to see the real posts behind this stance →");
}

function addPill(container, value, label) {
  const p = container.append("div").attr("class", "stat-pill");
  p.append("div").attr("class", "stat-pill-value").text(value);
  p.append("div").attr("class", "stat-pill-label").text(label);
}

// ---------- entity_timeline -> small diverging bar chart ----------

function renderTimeline(container, result) {
  container.append("div").attr("class", "section-title").text(`${result.canonical_name} -- stance over time (${result.bucket_size})`);
  const wrap = container.append("div").attr("class", "timeline-chart");

  const legend = wrap.append("div").attr("class", "timeline-legend");
  legend.append("span").html('<span class="swatch-dot" style="background:var(--positive)"></span> positive');
  legend.append("span").html('<span class="swatch-dot" style="background:var(--negative)"></span> negative');

  const data = result.timeline;
  if (!data.length) {
    wrap.append("div").style("color", "var(--text-faint)").style("font-size", "13px").style("padding", "16px 0").text("No timeline data.");
    return;
  }

  const width = 800;
  const height = 200;
  const margin = { top: 8, right: 12, bottom: 26, left: 12 };
  const svg = wrap.append("svg").attr("viewBox", `0 0 ${width} ${height}`).style("width", "100%").style("height", "auto");

  const x = d3
    .scaleBand()
    .domain(data.map((d) => d.bucket_start))
    .range([margin.left, width - margin.right])
    .padding(0.25);
  const maxVal = d3.max(data, (d) => Math.max(d.positive, d.negative)) || 1;
  const mid = (height - margin.bottom) / 2;
  const plotHalf = mid - 8;
  const barHeight = (v) => (v / maxVal) * plotHalf;

  svg.append("line").attr("x1", margin.left).attr("x2", width - margin.right).attr("y1", mid).attr("y2", mid).attr("stroke", "var(--border)");

  svg
    .selectAll(".bar-pos")
    .data(data)
    .join("rect")
    .attr("x", (d) => x(d.bucket_start))
    .attr("width", x.bandwidth())
    .attr("y", (d) => mid - barHeight(d.positive))
    .attr("height", (d) => barHeight(d.positive))
    .attr("fill", "var(--positive)")
    .attr("rx", 2);

  svg
    .selectAll(".bar-neg")
    .data(data)
    .join("rect")
    .attr("x", (d) => x(d.bucket_start))
    .attr("width", x.bandwidth())
    .attr("y", mid)
    .attr("height", (d) => barHeight(d.negative))
    .attr("fill", "var(--negative)")
    .attr("rx", 2);

  const tickStep = Math.max(1, Math.ceil(data.length / 8));
  const xAxis = d3
    .axisBottom(x)
    .tickValues(x.domain().filter((_, i) => i % tickStep === 0))
    .tickFormat((d) => d.slice(0, 10));
  const axisG = svg.append("g").attr("transform", `translate(0,${height - margin.bottom + 4})`).call(xAxis);
  axisG.select(".domain").attr("stroke", "var(--border)");
  axisG.selectAll("line").attr("stroke", "var(--border)");
  axisG.selectAll("text").attr("fill", "var(--text-faint)").attr("font-size", "9.5px");
}

// ---------- author_profile (used both as a pushed view and inline from /api/ask) ----------

function renderProfileContent(container, profile) {
  const header = container.append("div").attr("class", "profile-header");
  const left = header.append("div");
  left
    .append("span")
    .attr("class", "profile-source-badge")
    .style("background", colorForSource(profile.source_type))
    .text(profile.source_type);
  left.append("h2").attr("class", "profile-name").text(profile.display_name || profile.author_id);
  left.append("div").attr("class", "profile-id").text(profile.author_id);

  const meta = container.append("div").attr("class", "profile-meta");
  addMeta(meta, fmtNum(profile.item_count), "items");
  addMeta(meta, formatSpanDays(profile.time_span), "day span");
  addMeta(meta, profile.stance_vector.length, "entities shown");

  container.append("div").attr("class", "section-title").text("Stance fingerprint -- click any entity to see the real posts");

  if (!profile.stance_vector.length) {
    container.append("div").attr("class", "empty-column").text("No stance edges recorded for this author.");
    return;
  }

  const rows = container
    .selectAll(".stance-row")
    .data(profile.stance_vector)
    .join("div")
    .attr("class", "stance-row")
    .on("click", (event, d) =>
      pushView({
        type: "sources",
        label: `${profile.display_name || profile.author_id} → ${d.canonical_name}`,
        params: { authorId: profile.author_id, entityId: d.entity_id, authorLabel: profile.display_name, entityLabel: d.canonical_name },
      })
    );

  const labelRow = rows.append("div").attr("class", "stance-label-row");
  labelRow
    .append("div")
    .html((d) => `<span class="entity-name">${escapeHtml(d.canonical_name)}</span><span class="entity-type">${escapeHtml(d.entity_type || "")}</span>`);
  labelRow
    .append("div")
    .attr("class", "stance-value")
    .style("color", (d) => stanceColor(d.net_stance))
    .text((d) => fmtSigned(d.net_stance));

  const track = rows.append("div").attr("class", "stance-bar-track");
  track.append("div").attr("class", "stance-bar-zero");
  track
    .append("div")
    .attr("class", (d) => `stance-bar ${d.net_stance >= 0 ? "positive" : "negative"}`)
    .style("width", (d) => `${Math.min(Math.abs(d.net_stance), 1) * 50}%`)
    .style("height", "9px");

  const metaRow = rows.append("div").attr("class", "stance-meta");
  metaRow
    .append("span")
    .text((d) => `vol ${fmtNum(d.volume)} · ${d.positive_count} pos / ${d.negative_count} neg / ${d.neutral_count} neu · consistency ${d.stance_consistency.toFixed(2)}`);
  metaRow.append("span").attr("class", "stance-affordance").html("view posts &rsaquo;");
}

async function renderProfileView(container, view, isCurrent) {
  container.html("");
  if (!view.data) {
    renderSkeleton(container);
    try {
      view.data = await fetchJSON(`/api/author/${encodeURIComponent(view.params.authorId)}?limit=20`);
    } catch (err) {
      if (!isCurrent()) return;
      container.html("");
      renderErrorState(container, "Couldn't load this author's profile.", err);
      return;
    }
  }
  if (!isCurrent()) return;
  container.html("");
  renderProfileContent(container, view.data);
}

// ---------- sources view -- the "verify it yourself" moment ----------

function renderSourceCards(container, items, options = {}) {
  const cards = container.selectAll(".source-card").data(items).join("div").attr("class", "source-card");

  const head = cards.append("div").attr("class", "source-card-head");
  if (options.showPolarity) {
    head.append("span").attr("class", (d) => `polarity-badge ${d.polarity}`).text((d) => d.polarity);
  }
  head
    .append("span")
    .attr("class", "source-author")
    .style("color", (d) => colorForSource(d.source_type))
    .text((d) => options.authorLabelFor ? options.authorLabelFor(d) : d.source_type);
  head.append("span").attr("class", "source-timestamp").text((d) => formatDate(d.published_at));

  cards.append("p").attr("class", "source-text").text((d) => d.text || d.transcript_snippet || d.text_snippet || "(no text captured)");

  const footer = cards.append("div").attr("class", "source-footer");
  footer
    .append("span")
    .attr("class", "source-strength")
    .text((d) => (options.showPolarity ? `strength ${d.strength != null ? d.strength.toFixed(2) : "—"} · confidence ${d.confidence != null ? d.confidence.toFixed(2) : "—"}` : d.source_native_id || ""));
  footer
    .append("a")
    .attr("class", (d) => `view-original ${d.source_url ? "" : "disabled"}`)
    .attr("href", (d) => d.source_url || null)
    .attr("target", (d) => (d.source_url ? "_blank" : null))
    .attr("rel", "noopener noreferrer")
    .html((d) => (d.source_url ? 'View original <span aria-hidden="true">→</span>' : `no link available (id: ${escapeHtml(d.source_native_id || d.item_id || "")})`));
}

async function renderSourcesView(container, view, isCurrent) {
  container.html("");
  renderSkeleton(container);
  if (!view.data) {
    try {
      view.data = await fetchJSON(
        `/api/author/${encodeURIComponent(view.params.authorId)}/entity/${encodeURIComponent(view.params.entityId)}/sources?limit=20`
      );
    } catch (err) {
      if (!isCurrent()) return;
      container.html("");
      renderErrorState(container, "Couldn't load source items for this author/entity pair.", err);
      return;
    }
  }
  if (!isCurrent()) return;
  container.html("");
  const data = view.data;
  container
    .append("div")
    .attr("class", "section-title")
    .text(`${data.n_items} source item${data.n_items === 1 ? "" : "s"} -- ${view.params.authorLabel || data.author_id} → ${data.canonical_name}`);

  if (!data.items.length) {
    container.append("div").attr("class", "empty-column").text("No source items found for this pairing.");
    return;
  }
  renderSourceCards(container, data.items, { showPolarity: true });
}

// ---------- coordination cluster: topic_coordination cards + full cluster drill-down ----------

function renderTopicCoordination(container, result) {
  container.append("div").attr("class", "section-title").text(`Coordinated clusters matching "${result.topic_query}"`);
  if (!result.clusters.length) {
    container.append("div").attr("class", "empty-column").text("No coordinated clusters matched this topic.");
    return;
  }
  const cards = container
    .selectAll(".cluster-card")
    .data(result.clusters)
    .join("div")
    .attr("class", "cluster-card")
    .on("click", (event, d) =>
      pushView({ type: "cluster", label: `Cluster (${d.distinct_authors} channels)`, params: { narrativeId: d.narrative_id } })
    );

  const head = cards.append("div").attr("class", "cluster-card-head");
  head.append("span").attr("class", "cluster-authors-badge").text((d) => `${d.distinct_authors} channels`);
  head.append("span").style("font-size", "12px").style("color", "var(--text-faint)").text((d) => `${d.size} items`);
  head.append("span").attr("class", "cluster-time-range").text((d) => `${formatDate(d.time_range_start)} – ${formatDate(d.time_range_end)}`);

  cards.append("p").attr("class", "cluster-sample").text((d) => d.sample_text);
}

async function renderClusterView(container, view, isCurrent) {
  container.html("");
  renderSkeleton(container);
  if (!view.data) {
    try {
      view.data = await fetchJSON(`/api/narrative/${encodeURIComponent(view.params.narrativeId)}`);
    } catch (err) {
      if (!isCurrent()) return;
      container.html("");
      renderErrorState(container, "Couldn't load this coordination cluster.", err);
      return;
    }
  }
  if (!isCurrent()) return;
  container.html("");
  const data = view.data;
  container
    .append("div")
    .attr("class", "section-title")
    .text(`${data.size} items across ${data.distinct_authors} channels -- identical or near-identical content, grouped by channel`);

  const groups = d3.groups(data.members, (d) => d.author_id).sort((a, b) => d3.ascending(a[1][0].published_at, b[1][0].published_at));

  for (const [authorId, members] of groups) {
    const group = container.append("div").attr("class", "cluster-items-group");
    group.append("div").attr("class", "cluster-group-label").text(`${members[0].author_display_name || authorId}  ·  ${authorId}`);
    const asItems = members.map((m) => ({
      item_id: m.item_id,
      source_type: sourceTypeOf(authorId),
      published_at: m.published_at,
      text_snippet: m.text_snippet,
      source_url: m.source_url,
      source_native_id: m.source_native_id,
    }));
    renderSourceCards(group, asItems, { showPolarity: false });
  }
}

// ============================================================ coordination graph

function isTemporal(edge) {
  return Array.isArray(edge.edge_types) && edge.edge_types.includes("temporal_cocluster");
}

function dragBehavior(sim) {
  function started(event, d) {
    if (!event.active) sim.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }
  function ended(event, d) {
    if (!event.active) sim.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
  return d3.drag().on("start", started).on("drag", dragged).on("end", ended);
}

const tooltip = d3.select("#tooltip");

function showNodeTooltip(event, d) {
  tooltip
    .classed("hidden", false)
    .html(
      `<div class="tt-title">${escapeHtml(d.display_name)}</div>
       <div class="tt-row">${d.source_type} · ${fmtNum(d.total_items)} items</div>
       <div class="tt-row">${fmtNum(d.distinct_entities)} distinct entities</div>
       <div class="tt-row" style="margin-top:4px;color:var(--accent)">click to view profile</div>`
    );
  positionTooltip(event);
}

function showEdgeTooltip(event, d) {
  const gapText = d.min_time_gap_seconds == null ? "n/a" : formatDuration(d.min_time_gap_seconds);
  const typesText = (d.edge_types || []).join(", ");
  tooltip
    .classed("hidden", false)
    .html(
      `<div class="tt-title">${d.edge_count} shared/synchrony edges</div>
       <div class="tt-row ${d.min_time_gap_seconds < 300 ? "tt-highlight" : ""}">min gap: ${gapText}</div>
       <div class="tt-row">${escapeHtml(typesText)}</div>`
    );
  positionTooltip(event);
}

function positionTooltip(event) {
  tooltip.style("left", `${event.clientX + 16}px`).style("top", `${event.clientY + 12}px`);
}

function hideTooltip() {
  tooltip.classed("hidden", true);
}

async function loadAndRenderGraph(svg, status, minEdges, isCurrent) {
  status.text("loading…");
  let data;
  try {
    data = await fetchJSON(`/api/graph/coordination?min_edges=${minEdges}&limit=150`);
  } catch (err) {
    if (!isCurrent()) return;
    status.text(`Couldn't reach the backend at ${API_BASE}.`);
    console.error("failed to load coordination graph", err);
    return;
  }
  if (!isCurrent()) return;
  renderGraphSvg(svg, status, data);
}

function renderGraphSvg(svgSel, statusSel, data) {
  const nodes = data.nodes.map((d) => ({ ...d }));
  const links = data.edges.map((d) => ({ ...d }));

  svgSel.selectAll("*").remove();
  const bounds = svgSel.node().getBoundingClientRect();
  const width = bounds.width || 900;
  const height = bounds.height || 500;
  svgSel.attr("viewBox", [0, 0, width, height]);

  if (!nodes.length) {
    statusSel.text("No author pairs meet this threshold -- lower the slider.");
    return;
  }
  statusSel.text(`${nodes.length} authors · ${links.length} coordination links`);

  const maxItems = d3.max(nodes, (d) => d.total_items) || 1;
  const radiusScale = d3.scaleSqrt().domain([0, maxItems]).range([6, 32]);
  const edgeCounts = links.map((d) => d.edge_count);
  const widthScale = d3
    .scaleSqrt()
    .domain([d3.min(edgeCounts) || 1, d3.max(edgeCounts) || 1])
    .range([1.5, 10]);

  const g = svgSel.append("g");
  svgSel.call(d3.zoom().scaleExtent([0.15, 6]).on("zoom", (event) => g.attr("transform", event.transform)));

  const simulation = d3
    .forceSimulation(nodes)
    .alphaDecay(0.05) // settle noticeably faster than d3's default (~5s) -- this graph can have 1000+ mostly-unconnected nodes
    .force("link", d3.forceLink(links).id((d) => d.id).distance(85).strength(0.35))
    .force("charge", d3.forceManyBody().strength(-240))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius((d) => radiusScale(d.total_items) + 9));

  const edgeSel = g
    .append("g")
    .attr("class", "edges")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("class", "edge")
    .attr("stroke", (d) => (isTemporal(d) ? "var(--edge-temporal)" : "var(--edge-plain)"))
    .attr("stroke-width", (d) => widthScale(d.edge_count))
    .attr("stroke-opacity", (d) => (isTemporal(d) ? 0.85 : 0.5))
    .on("mouseenter", showEdgeTooltip)
    .on("mousemove", positionTooltip)
    .on("mouseleave", hideTooltip);

  const nodeSel = g
    .append("g")
    .attr("class", "nodes")
    .selectAll("g.node")
    .data(nodes, (d) => d.id)
    .join("g")
    .attr("class", "node")
    .call(dragBehavior(simulation))
    .on("click", (event, d) => {
      event.stopPropagation();
      pushView({ type: "profile", label: d.display_name || d.id, params: { authorId: d.id } });
    })
    .on("mouseenter", showNodeTooltip)
    .on("mousemove", positionTooltip)
    .on("mouseleave", hideTooltip);

  nodeSel.append("circle").attr("r", (d) => radiusScale(d.total_items)).attr("fill", (d) => colorForSource(d.source_type));
  nodeSel
    .append("text")
    .attr("dy", (d) => radiusScale(d.total_items) + 12)
    .attr("text-anchor", "middle")
    .text((d) => truncateLabel(d.display_name));

  svgSel.on("click", () => {
    nodeSel.classed("selected dimmed", false);
    edgeSel.classed("dimmed", false);
  });

  simulation.on("tick", () => {
    edgeSel.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
    nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });
}

function renderGraphView(container, view, isCurrent) {
  container.html("");
  const toolbar = container.append("div").attr("class", "graph-toolbar");

  const controlGroup = toolbar.append("div").attr("class", "control-group");
  controlGroup.append("label").attr("for", "minEdges").text("Min shared edges per author pair");
  const sliderRow = controlGroup.append("div").attr("class", "slider-row");
  const minEdges = view.params.minEdges || 5;
  const slider = sliderRow.append("input").attr("type", "range").attr("id", "minEdges").attr("min", 1).attr("max", 30).attr("value", minEdges).attr("step", 1);
  const sliderValue = sliderRow.append("span").attr("class", "slider-value").text(minEdges);

  const legend = toolbar.append("div").attr("class", "legend");
  legend.append("div").attr("class", "legend-item").html('<span class="swatch swatch-telegram"></span>Telegram');
  legend.append("div").attr("class", "legend-item").html('<span class="swatch swatch-youtube"></span>YouTube');
  legend.append("div").attr("class", "legend-item").html('<span class="swatch swatch-channel"></span>Channel record');
  legend.append("div").attr("class", "legend-item").html('<span class="swatch swatch-edge"></span>Shared content');
  legend.append("div").attr("class", "legend-item").html('<span class="swatch swatch-edge-temporal"></span>+ Temporal cocluster');

  const canvas = container.append("div").attr("class", "graph-canvas");
  const svg = canvas.append("svg").attr("id", "graph");
  const status = canvas.append("div").attr("class", "graph-status");

  // Record the canvas's just-laid-out size so a same-size resize event
  // some browsers fire on first paint doesn't trigger a redundant refetch.
  requestAnimationFrame(() => {
    const rect = canvas.node().getBoundingClientRect();
    lastGraphCanvasSize = `${Math.round(rect.width)}x${Math.round(rect.height)}`;
  });

  const refresh = () => loadAndRenderGraph(svg, status, view.params.minEdges || 5, isCurrent);
  activeGraphRefresh = refresh;

  let debounce = null;
  slider.on("input", function () {
    sliderValue.text(this.value);
    view.params.minEdges = Number(this.value);
    clearTimeout(debounce);
    debounce = setTimeout(refresh, 150);
  });

  refresh();
}

// Some browsers fire a synthetic resize on first layout even when the
// canvas's own size hasn't changed -- guard against that so opening the
// graph doesn't immediately re-fetch and re-render a second time.
let lastGraphCanvasSize = null;
window.addEventListener("resize", () => {
  if (!activeGraphRefresh) return;
  const canvas = document.querySelector(".graph-canvas");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const key = `${Math.round(rect.width)}x${Math.round(rect.height)}`;
  if (key === lastGraphCanvasSize) return;
  lastGraphCanvasSize = key;
  activeGraphRefresh();
});

// ============================================================ header stats

async function loadStats() {
  try {
    const stats = await fetchJSON("/api/stats");
    const values = [stats.n_items, stats.n_authors, stats.n_entities, stats.n_stance_edges, stats.n_coordination_clusters];
    d3.selectAll("#stats .stat-value").each(function (_, i) {
      d3.select(this).text(fmtNum(values[i]));
    });
  } catch (err) {
    console.error("failed to load stats", err);
    d3.select(".stats-strip").html('<div class="stat"><span class="stat-value">—</span><span class="stat-label">backend unreachable</span></div>');
  }
}

// ============================================================ wiring

const askForm = document.getElementById("askForm");
const askInput = document.getElementById("askInput");
const askSubmit = document.getElementById("askSubmit");

async function submitQuestion(question) {
  question = (question || "").trim();
  if (!question) return;
  askInput.value = question;
  askSubmit.classList.add("loading");
  askSubmit.disabled = true;

  viewStack = [];
  renderBreadcrumbs();
  const container = d3.select("#resultsArea");
  container.html("");
  container
    .append("div")
    .attr("class", "state-panel")
    .html('<p class="state-title">Thinking…</p><p>Routing your question through the local model -- this can take several seconds.</p>');

  try {
    const response = await postJSON("/api/ask", { question });
    viewStack = [{ type: "ask", label: truncateLabel(question, 44), params: { question, response } }];
    scheduleRender();
  } catch (err) {
    console.error("ask failed", err);
    container.html("");
    renderErrorState(container, `Couldn't reach the backend at ${API_BASE}.`, err);
  } finally {
    askSubmit.classList.remove("loading");
    askSubmit.disabled = false;
  }
}

askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(askInput.value);
});

document.getElementById("exampleChips").addEventListener("click", (event) => {
  const btn = event.target.closest(".chip");
  if (!btn) return;
  submitQuestion(btn.dataset.q);
});

document.getElementById("networkToggle").addEventListener("click", () => {
  pushView({ type: "graph", label: "Coordination network", params: { minEdges: 5 } });
});

loadStats();
scheduleRender();
