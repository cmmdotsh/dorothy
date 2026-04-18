/**
 * Claim Graph Visualization
 *
 * Renders a bipartite force-directed graph: corroborated claims in the center,
 * source nodes orbiting around them, edges showing which outlet confirmed which fact.
 * Unique details listed below the graph as isolated attributions.
 */
(function () {
  "use strict";

  const container = document.getElementById("claim-graph");
  if (!container) return;

  const dataEl = document.getElementById("claim-graph-data");
  if (!dataEl) return;

  let data;
  try {
    data = JSON.parse(dataEl.textContent);
  } catch (e) {
    return;
  }

  if (!data || !data.corroborated || data.corroborated.length === 0) {
    container.innerHTML =
      '<p class="cg-empty">No cross-source corroboration data available for this story.</p>';
    return;
  }

  /* ── Theme detection ── */
  function isDark() {
    return document.documentElement.getAttribute("data-theme") === "dark";
  }

  const BIAS_COLORS = {
    left: "#3b82f6",
    "lean-left": "#60a5fa",
    center: "#a855f7",
    "lean-right": "#f97316",
    right: "#ef4444",
  };

  function biasColor(bias) {
    return BIAS_COLORS[bias] || "#888";
  }

  /* ── Build graph structure ── */
  const nodes = [];
  const links = [];
  const sourceMap = new Map(); // slug → node index

  // Claim nodes (one per corroborated cluster)
  data.corroborated.forEach((cluster, ci) => {
    nodes.push({
      id: "claim-" + ci,
      type: "claim",
      label: cluster.representative_text.slice(0, 120),
      sourceCount: cluster.source_count,
      similarity: cluster.avg_similarity,
      sourceNames: cluster.source_names,
    });
  });

  // Source nodes (deduplicated across all clusters)
  data.corroborated.forEach((cluster, ci) => {
    cluster.sources.forEach((src) => {
      const key = src.source_slug || src.source_name;
      if (!sourceMap.has(key)) {
        const idx = nodes.length;
        sourceMap.set(key, idx);
        nodes.push({
          id: "src-" + key,
          type: "source",
          label: src.source_name,
          slug: key,
          bias: src.source_bias,
          claimCount: 0,
        });
      }
      // Count how many claims this source appears in
      nodes[sourceMap.get(key)].claimCount += 1;

      links.push({
        source: ci, // claim node index
        target: sourceMap.get(key),
      });
    });
  });

  const claimCount = data.corroborated.length;
  const sourceCount = sourceMap.size;

  /* ── Responsive sizing ── */
  const isMobile = window.innerWidth < 600;
  const width = container.clientWidth || 700;
  const height = isMobile
    ? Math.max(320, claimCount * 80 + sourceCount * 30)
    : Math.max(400, Math.min(600, claimCount * 100 + 100));

  /* ── SVG setup ── */
  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [0, 0, width, height]);

  // Defs for patterns/markers
  const defs = svg.append("defs");

  // Crosshatch pattern for claim nodes (editorial engraving feel)
  const pat = defs
    .append("pattern")
    .attr("id", "cg-hatch")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 4)
    .attr("height", 4);
  pat
    .append("path")
    .attr("d", "M-1,1 l2,-2 M0,4 l4,-4 M3,5 l2,-2")
    .attr("stroke", isDark() ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)")
    .attr("stroke-width", 1);

  const g = svg.append("g");

  /* ── Zoom ── */
  const zoom = d3
    .zoom()
    .scaleExtent([0.5, 3])
    .on("zoom", (event) => g.attr("transform", event.transform));
  svg.call(zoom);

  /* ── Force simulation ── */
  const sim = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d, i) => i)
        .distance(isMobile ? 60 : 90)
        .strength(0.7)
    )
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force(
      "collide",
      d3.forceCollide().radius((d) => (d.type === "claim" ? 50 : 20))
    )
    .force(
      "x",
      d3
        .forceX()
        .x((d) => {
          if (d.type === "claim") return width / 2;
          // Push sources outward based on bias
          const biasOrder = {
            left: 0,
            "lean-left": 0.25,
            center: 0.5,
            "lean-right": 0.75,
            right: 1,
          };
          const t = biasOrder[d.bias] ?? 0.5;
          return width * 0.15 + t * width * 0.7;
        })
        .strength(0.15)
    )
    .force(
      "y",
      d3
        .forceY()
        .y((d) => (d.type === "claim" ? height * 0.45 : height * 0.55))
        .strength(0.05)
    );

  /* ── Links ── */
  const link = g
    .append("g")
    .attr("class", "cg-links")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", isDark() ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)")
    .attr("stroke-width", 1.5);

  /* ── Claim nodes ── */
  const claimNodes = g
    .append("g")
    .attr("class", "cg-claims")
    .selectAll("g")
    .data(nodes.filter((d) => d.type === "claim"))
    .join("g")
    .attr("class", "cg-claim-node")
    .call(drag(sim));

  // Claim background rect
  claimNodes
    .append("rect")
    .attr("rx", 3)
    .attr("ry", 3)
    .attr("width", isMobile ? 100 : 140)
    .attr("height", isMobile ? 36 : 44)
    .attr("x", isMobile ? -50 : -70)
    .attr("y", isMobile ? -18 : -22)
    .attr("fill", isDark() ? "#2a2a2a" : "#fff")
    .attr("stroke", isDark() ? "#444" : "#1a1a1a")
    .attr("stroke-width", 1.5);

  // Hatch overlay
  claimNodes
    .append("rect")
    .attr("rx", 3)
    .attr("ry", 3)
    .attr("width", isMobile ? 100 : 140)
    .attr("height", isMobile ? 36 : 44)
    .attr("x", isMobile ? -50 : -70)
    .attr("y", isMobile ? -18 : -22)
    .attr("fill", "url(#cg-hatch)")
    .attr("pointer-events", "none");

  // Claim text
  claimNodes
    .append("text")
    .attr("class", "cg-claim-text")
    .attr("text-anchor", "middle")
    .attr("dy", "-0.15em")
    .text((d) => {
      const max = isMobile ? 28 : 40;
      return d.label.length > max ? d.label.slice(0, max - 1) + "…" : d.label;
    });

  // Source count badge
  claimNodes
    .append("text")
    .attr("class", "cg-claim-badge")
    .attr("text-anchor", "middle")
    .attr("dy", "1.2em")
    .text((d) => d.sourceCount + " sources · " + Math.round(d.similarity * 100) + "% match");

  /* ── Source nodes ── */
  const sourceNodes = g
    .append("g")
    .attr("class", "cg-sources")
    .selectAll("g")
    .data(nodes.filter((d) => d.type === "source"))
    .join("g")
    .attr("class", "cg-source-node")
    .call(drag(sim));

  // Source circle
  sourceNodes
    .append("circle")
    .attr("r", (d) => 6 + d.claimCount * 2.5)
    .attr("fill", (d) => biasColor(d.bias))
    .attr("stroke", isDark() ? "#242424" : "#fff")
    .attr("stroke-width", 2);

  // Source label
  sourceNodes
    .append("text")
    .attr("class", "cg-source-label")
    .attr("dx", (d) => 10 + d.claimCount * 2.5)
    .attr("dy", "0.35em")
    .text((d) => d.label);

  /* ── Interaction: highlight on hover ── */
  claimNodes
    .on("mouseenter", function (event, d) {
      const connectedSources = new Set();
      links.forEach((l) => {
        const srcIdx = typeof l.source === "object" ? l.source.index : l.source;
        const tgtIdx = typeof l.target === "object" ? l.target.index : l.target;
        if (srcIdx === d.index || tgtIdx === d.index) {
          connectedSources.add(srcIdx);
          connectedSources.add(tgtIdx);
        }
      });

      sourceNodes.attr("opacity", (s) =>
        connectedSources.has(s.index) ? 1 : 0.15
      );
      link.attr("opacity", (l) => {
        const srcIdx = typeof l.source === "object" ? l.source.index : l.source;
        const tgtIdx = typeof l.target === "object" ? l.target.index : l.target;
        return connectedSources.has(srcIdx) && connectedSources.has(tgtIdx) ? 1 : 0.05;
      });
      link
        .filter((l) => {
          const srcIdx = typeof l.source === "object" ? l.source.index : l.source;
          const tgtIdx = typeof l.target === "object" ? l.target.index : l.target;
          return connectedSources.has(srcIdx) && connectedSources.has(tgtIdx);
        })
        .attr("stroke", isDark() ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.35)")
        .attr("stroke-width", 2.5);
    })
    .on("mouseleave", function () {
      sourceNodes.attr("opacity", 1);
      link
        .attr("opacity", 1)
        .attr("stroke", isDark() ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)")
        .attr("stroke-width", 1.5);
    });

  sourceNodes
    .on("mouseenter", function (event, d) {
      const connectedClaims = new Set();
      links.forEach((l) => {
        const srcIdx = typeof l.source === "object" ? l.source.index : l.source;
        const tgtIdx = typeof l.target === "object" ? l.target.index : l.target;
        if (srcIdx === d.index || tgtIdx === d.index) {
          connectedClaims.add(srcIdx);
          connectedClaims.add(tgtIdx);
        }
      });

      claimNodes.attr("opacity", (c) =>
        connectedClaims.has(c.index) ? 1 : 0.15
      );
      link.attr("opacity", (l) => {
        const srcIdx = typeof l.source === "object" ? l.source.index : l.source;
        const tgtIdx = typeof l.target === "object" ? l.target.index : l.target;
        return connectedClaims.has(srcIdx) && connectedClaims.has(tgtIdx) ? 1 : 0.05;
      });
      link
        .filter((l) => {
          const srcIdx = typeof l.source === "object" ? l.source.index : l.source;
          const tgtIdx = typeof l.target === "object" ? l.target.index : l.target;
          return connectedClaims.has(srcIdx) && connectedClaims.has(tgtIdx);
        })
        .attr("stroke", biasColor(d.bias))
        .attr("stroke-width", 2.5)
        .attr("stroke-opacity", 0.6);
    })
    .on("mouseleave", function () {
      claimNodes.attr("opacity", 1);
      link
        .attr("opacity", 1)
        .attr("stroke", isDark() ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)")
        .attr("stroke-width", 1.5);
    });

  /* ── Tick ── */
  sim.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    claimNodes.attr("transform", (d) => `translate(${d.x},${d.y})`);
    sourceNodes.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });

  /* ── Drag ── */
  function drag(simulation) {
    return d3
      .drag()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
  }

  /* ── Unique details list ── */
  if (data.unique_details && data.unique_details.length > 0) {
    const listDiv = document.createElement("div");
    listDiv.className = "cg-unique-details";

    const heading = document.createElement("h4");
    heading.className = "cg-unique-heading";
    heading.textContent = "Unique Details";
    listDiv.appendChild(heading);

    const subhead = document.createElement("p");
    subhead.className = "cg-unique-subhead";
    subhead.textContent =
      "Facts reported by only one source — not independently corroborated.";
    listDiv.appendChild(subhead);

    const ul = document.createElement("ul");
    ul.className = "cg-unique-list";

    // Group by source
    const bySource = {};
    data.unique_details.forEach((d) => {
      const key = d.source_slug || d.source_name;
      if (!bySource[key]) bySource[key] = { name: d.source_name, bias: d.source_bias, items: [] };
      bySource[key].items.push(d.text);
    });

    Object.values(bySource).forEach((src) => {
      const li = document.createElement("li");
      const badge = document.createElement("span");
      badge.className = "cg-unique-badge";
      badge.style.background = biasColor(src.bias);
      badge.textContent = src.name;
      li.appendChild(badge);

      const txt = document.createElement("span");
      txt.className = "cg-unique-text";
      // Show first item, indicate if more
      txt.textContent =
        src.items[0].slice(0, 140) +
        (src.items[0].length > 140 ? "…" : "") +
        (src.items.length > 1 ? " (+" + (src.items.length - 1) + " more)" : "");
      li.appendChild(txt);
      ul.appendChild(li);
    });

    listDiv.appendChild(ul);
    container.appendChild(listDiv);
  }

  /* ── Stats footer ── */
  const stats = document.createElement("div");
  stats.className = "cg-stats";
  stats.innerHTML =
    '<span class="cg-stat">' + data.chunk_count + " passages analyzed</span>" +
    '<span class="cg-stat-sep">·</span>' +
    '<span class="cg-stat">' + data.corroborated.length + " corroborated claims</span>" +
    '<span class="cg-stat-sep">·</span>' +
    '<span class="cg-stat">' + sourceCount + " sources</span>";
  container.appendChild(stats);

  /* ── Theme change listener ── */
  const observer = new MutationObserver(() => {
    const dark = isDark();
    claimNodes
      .selectAll("rect:first-child")
      .attr("fill", dark ? "#2a2a2a" : "#fff")
      .attr("stroke", dark ? "#444" : "#1a1a1a");
    link.attr("stroke", dark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)");
    sourceNodes.selectAll("circle").attr("stroke", dark ? "#242424" : "#fff");
  });
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
})();
