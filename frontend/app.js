// Point this at the deployed backend when there is one -- everything else
// in this file reads from API_BASE, nothing else hardcodes a host.
const API_BASE = "http://127.0.0.1:8899";

const SOURCE_COLORS = {
  telegram: "#5dade2",
  youtube_video: "#ec7063",
};
const fallbackColor = d3.scaleOrdinal(d3.schemeSet2);
function colorForSource(sourceType) {
  return SOURCE_COLORS[sourceType] || fallbackColor(sourceType);
}

const svg = d3.select("#graph");
const tooltip = d3.select("#tooltip");
const graphStatus = d3.select("#graphStatus");
const sidePanel = d3.select("#sidePanel");

let selectedNodeId = null;

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------- stats

async function loadStats() {
  try {
    const stats = await fetchJSON("/api/stats");
    const values = [
      stats.n_items,
      stats.n_authors,
      stats.n_entities,
      stats.n_stance_edges,
      stats.n_coordination_clusters,
    ];
    d3.selectAll("#stats .stat-value").each(function (_, i) {
      d3.select(this).text(d3.format(",")(values[i]));
    });
  } catch (err) {
    console.error("failed to load stats", err);
    d3.select(".stats-strip").append("div").attr("class", "stat").text("backend unreachable");
  }
}

// ---------------------------------------------------------------- graph

let simulation = null;

async function loadGraph(minEdges) {
  graphStatus.text("loading…");
  let data;
  try {
    data = await fetchJSON(`/api/graph/coordination?min_edges=${minEdges}&limit=150`);
  } catch (err) {
    console.error("failed to load coordination graph", err);
    graphStatus.text("failed to reach backend -- is uvicorn running on " + API_BASE + "?");
    return;
  }
  renderGraph(data);
}

function renderGraph(data) {
  const nodes = data.nodes.map((d) => ({ ...d }));
  // d3-force mutates link endpoints in place, replacing string ids with
  // node object refs -- copy so re-renders (slider changes) always start
  // from plain {source, target} id pairs again.
  const links = data.edges.map((d) => ({ ...d }));

  svg.selectAll("*").remove();

  const bounds = svg.node().getBoundingClientRect();
  const width = bounds.width || 900;
  const height = bounds.height || 600;
  svg.attr("viewBox", [0, 0, width, height]);

  if (nodes.length === 0) {
    graphStatus.text("no author pairs meet this threshold -- lower min_edges");
    return;
  }
  graphStatus.text(`${nodes.length} authors · ${links.length} coordination links`);

  const maxItems = d3.max(nodes, (d) => d.total_items) || 1;
  const radiusScale = d3.scaleSqrt().domain([0, maxItems]).range([6, 34]);

  const edgeCounts = links.map((d) => d.edge_count);
  const widthScale = d3
    .scaleSqrt()
    .domain([d3.min(edgeCounts) || 1, d3.max(edgeCounts) || 1])
    .range([1.5, 11]);

  const g = svg.append("g");

  svg.call(
    d3
      .zoom()
      .scaleExtent([0.15, 6])
      .on("zoom", (event) => g.attr("transform", event.transform))
  );

  simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .distance(90)
        .strength(0.35)
    )
    .force("charge", d3.forceManyBody().strength(-260))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force(
      "collide",
      d3.forceCollide().radius((d) => radiusScale(d.total_items) + 10)
    );

  const edgeGroup = g.append("g").attr("class", "edges");
  const edgeSel = edgeGroup
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("class", "edge")
    .attr("stroke", (d) => (isTemporal(d) ? "var(--edge-temporal)" : "var(--edge-plain)"))
    .attr("stroke-width", (d) => widthScale(d.edge_count))
    .attr("stroke-opacity", (d) => (isTemporal(d) ? 0.85 : 0.55))
    .on("mouseenter", (event, d) => showEdgeTooltip(event, d))
    .on("mousemove", (event) => positionTooltip(event))
    .on("mouseleave", hideTooltip);

  const nodeGroup = g.append("g").attr("class", "nodes");
  const nodeSel = nodeGroup
    .selectAll("g.node")
    .data(nodes, (d) => d.id)
    .join("g")
    .attr("class", "node")
    .call(drag(simulation))
    .on("click", (event, d) => {
      event.stopPropagation();
      selectNode(d.id, nodeSel, edgeSel);
      showAuthorPanel(d.id);
    })
    .on("mouseenter", (event, d) => showNodeTooltip(event, d))
    .on("mousemove", (event) => positionTooltip(event))
    .on("mouseleave", hideTooltip);

  nodeSel
    .append("circle")
    .attr("r", (d) => radiusScale(d.total_items))
    .attr("fill", (d) => colorForSource(d.source_type));

  nodeSel
    .append("text")
    .attr("dy", (d) => radiusScale(d.total_items) + 13)
    .attr("text-anchor", "middle")
    .text((d) => truncateLabel(d.display_name));

  svg.on("click", () => {
    selectedNodeId = null;
    nodeSel.classed("selected dimmed", false);
    edgeSel.classed("dimmed", false);
  });

  simulation.on("tick", () => {
    edgeSel
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
    nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });

  // Preserve selection across a threshold change if the node is still present.
  if (selectedNodeId && nodes.some((n) => n.id === selectedNodeId)) {
    selectNode(selectedNodeId, nodeSel, edgeSel);
  }
}

function isTemporal(edge) {
  return Array.isArray(edge.edge_types) && edge.edge_types.includes("temporal_cocluster");
}

function truncateLabel(name, max = 22) {
  if (!name) return "";
  return name.length > max ? name.slice(0, max - 1) + "…" : name;
}

function selectNode(id, nodeSel, edgeSel) {
  selectedNodeId = id;
  nodeSel.classed("selected", (d) => d.id === id);
  nodeSel.classed("dimmed", (d) => d.id !== id && !isNeighbor(id, d.id, edgeSel));
  edgeSel.classed("dimmed", (d) => d.source.id !== id && d.target.id !== id);
}

function isNeighbor(selectedId, otherId, edgeSel) {
  let found = false;
  edgeSel.each((d) => {
    if (
      (d.source.id === selectedId && d.target.id === otherId) ||
      (d.target.id === selectedId && d.source.id === otherId)
    ) {
      found = true;
    }
  });
  return found;
}

function drag(sim) {
  function dragstarted(event, d) {
    if (!event.active) sim.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }
  function dragended(event, d) {
    if (!event.active) sim.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
  return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
}

// ---------------------------------------------------------------- tooltips

function showNodeTooltip(event, d) {
  tooltip
    .classed("hidden", false)
    .html(
      `<div class="tt-title">${escapeHtml(d.display_name)}</div>
       <div class="tt-row">${d.source_type} · ${d3.format(",")(d.total_items)} items</div>
       <div class="tt-row">${d3.format(",")(d.distinct_entities)} distinct entities</div>`
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
  const panelRect = document.querySelector(".graph-panel").getBoundingClientRect();
  tooltip.style("left", `${event.clientX - panelRect.left + 16}px`).style("top", `${event.clientY - panelRect.top + 12}px`);
}

function hideTooltip() {
  tooltip.classed("hidden", true);
}

function formatDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------- side panel

async function showAuthorPanel(authorId) {
  sidePanel.html('<div class="empty-state"><p>Loading…</p></div>');
  let profile;
  try {
    profile = await fetchJSON(`/api/author/${encodeURIComponent(authorId)}?limit=15`);
  } catch (err) {
    console.error("failed to load author profile", err);
    sidePanel.html('<div class="empty-state"><p>Failed to load this author.</p></div>');
    return;
  }
  renderAuthorPanel(profile);
}

function renderAuthorPanel(profile) {
  sidePanel.html("");

  const header = sidePanel.append("div").attr("class", "panel-header");
  header
    .append("span")
    .attr("class", "panel-source")
    .style("background", colorForSource(profile.source_type))
    .text(profile.source_type);
  header.append("h2").attr("class", "panel-name").text(profile.display_name);
  header.append("div").attr("class", "panel-id").text(profile.author_id);

  const meta = sidePanel.append("div").attr("class", "panel-meta");
  const span = formatSpan(profile.time_span);
  addMeta(meta, d3.format(",")(profile.item_count), "items");
  addMeta(meta, span.days, "day span");
  addMeta(meta, profile.stance_vector.length, "entities shown");

  sidePanel.append("div").attr("class", "panel-section-title").text("Stance fingerprint");

  if (profile.stance_vector.length === 0) {
    sidePanel.append("div").attr("class", "no-vector").text("No stance edges recorded for this author.");
    return;
  }

  const rows = sidePanel.selectAll(".stance-row").data(profile.stance_vector).join("div").attr("class", "stance-row");

  const maxVolume = d3.max(profile.stance_vector, (d) => d.volume) || 1;
  const barHeightScale = d3.scaleSqrt().domain([0, maxVolume]).range([5, 14]);

  const labelRow = rows.append("div").attr("class", "stance-label-row");
  labelRow.append("div").html((d) => `<span class="entity-name">${escapeHtml(d.canonical_name)}</span><span class="entity-type">${escapeHtml(d.entity_type)}</span>`);
  labelRow
    .append("div")
    .attr("class", "stance-value")
    .style("color", (d) => stanceColor(d.net_stance))
    .text((d) => (d.net_stance >= 0 ? "+" : "") + d.net_stance.toFixed(2));

  const track = rows.append("div").attr("class", "stance-bar-track");
  track.append("div").attr("class", "stance-bar-zero");
  track
    .append("div")
    .attr("class", (d) => `stance-bar ${d.net_stance >= 0 ? "positive" : "negative"}`)
    .style("width", (d) => `${Math.min(Math.abs(d.net_stance), 1) * 50}%`)
    .style("height", (d) => `${barHeightScale(d.volume)}px`);

  rows
    .append("div")
    .attr("class", "stance-meta")
    .text(
      (d) =>
        `vol ${d3.format(",")(d.volume)} · ${d.positive_count} pos / ${d.negative_count} neg / ${d.neutral_count} neu · consistency ${d.stance_consistency.toFixed(2)}`
    );

  if (profile.generate_vs_amplify == null) {
    sidePanel.append("div").attr("class", "panel-section-title").style("margin-top", "18px").text("Generate vs. amplify");
    sidePanel.append("div").attr("class", "no-vector").text("Not computed yet.");
  }
}

function addMeta(container, value, label) {
  const item = container.append("div").attr("class", "panel-meta-item");
  item.append("div").attr("class", "panel-meta-value").text(value);
  item.append("div").attr("class", "panel-meta-label").text(label);
}

function formatSpan(span) {
  if (!span || !span.first_seen || !span.last_seen) return { days: "—" };
  const days = Math.max(0, Math.round((new Date(span.last_seen) - new Date(span.first_seen)) / 86400000));
  return { days };
}

function stanceColor(value) {
  if (value > 0.03) return "var(--positive)";
  if (value < -0.03) return "var(--negative)";
  return "var(--neutral)";
}

// ---------------------------------------------------------------- wiring

const minEdgesInput = document.getElementById("minEdges");
const minEdgesValue = document.getElementById("minEdgesValue");
let debounceTimer = null;
minEdgesInput.addEventListener("input", () => {
  minEdgesValue.textContent = minEdgesInput.value;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => loadGraph(Number(minEdgesInput.value)), 150);
});

window.addEventListener("resize", () => {
  if (simulation) loadGraph(Number(minEdgesInput.value));
});

loadStats();
loadGraph(Number(minEdgesInput.value));
