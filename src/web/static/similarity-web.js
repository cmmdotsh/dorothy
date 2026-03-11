/* Dorothy — Source Similarity Web (D3 v7 force-directed graph + mobile matrix) */
(function () {
  var el = document.getElementById("similarity-data");
  if (!el) return;

  var data;
  try {
    data = JSON.parse(el.textContent);
  } catch (e) {
    return;
  }

  if (!data.nodes || !data.edges || data.nodes.length < 2) return;

  var tabRadio = document.getElementById("tab-sources");
  if (!tabRadio) return;

  var initialized = false;

  function init() {
    if (initialized) return;
    initialized = true;

    // ════════════════════════════════════════
    //  SHARED: constants, aggregation, helpers
    // ════════════════════════════════════════

    var BIAS_COLORS = {
      left: "#3b82f6",
      "lean-left": "#60a5fa",
      center: "#a855f7",
      "lean-right": "#f97316",
      right: "#ef4444",
    };

    var REGION_COLORS = {
      us: "#3b82f6",
      canada: "#ef4444",
      mexico: "#22c55e",
      uk: "#6366f1",
      australia: "#eab308",
      india: "#f97316",
      japan: "#ec4899",
      korea: "#14b8a6",
      international: "#8b5cf6",
    };

    var isSports = data.column === "sports";
    var isTech = data.column === "tech";

    var PERSPECTIVE_COLORS = {
      consumer: "#3b82f6",
      enterprise: "#f97316",
      academic: "#a855f7",
      culture: "#22c55e",
    };

    // ── Aggregate nodes by source ──
    var sourceMap = {};
    data.nodes.forEach(function (n) {
      var key = n.source_name;
      if (!sourceMap[key]) {
        sourceMap[key] = {
          source_name: n.source_name,
          source_bias: n.source_bias,
          source_region: n.source_region,
          source_perspective: n.source_perspective,
          articleIndices: [],
          count: 0,
        };
      }
      sourceMap[key].articleIndices.push(n.index);
      sourceMap[key].count++;
    });

    var sourceNodes = Object.values(sourceMap);
    if (sourceNodes.length < 2) return;

    var indexToSource = {};
    data.nodes.forEach(function (n) {
      indexToSource[n.index] = n.source_name;
    });

    // ── Aggregate edges between sources ──
    var edgeSums = {};
    data.edges.forEach(function (e) {
      var sA = indexToSource[e.source];
      var sB = indexToSource[e.target];
      if (!sA || !sB || sA === sB) return;
      var key = sA < sB ? sA + "|" + sB : sB + "|" + sA;
      if (!edgeSums[key]) edgeSums[key] = { total: 0, count: 0 };
      edgeSums[key].total += e.similarity;
      edgeSums[key].count++;
    });

    var sourceByName = {};
    sourceNodes.forEach(function (s) { sourceByName[s.source_name] = s; });

    var allEdges = [];
    Object.keys(edgeSums).forEach(function (key) {
      var parts = key.split("|");
      var avg = edgeSums[key].total / edgeSums[key].count;
      allEdges.push({
        sourceNode: sourceByName[parts[0]],
        targetNode: sourceByName[parts[1]],
        similarity: avg,
      });
    });

    // Filter weak edges (< 0.65 similarity)
    allEdges = allEdges.filter(function (e) { return e.similarity >= 0.65; });

    if (allEdges.length === 0) return;

    // ── Color helper ──
    function nodeColor(d) {
      if (isSports && d.source_region) {
        return REGION_COLORS[d.source_region] || "#888";
      }
      if (isTech && d.source_perspective) {
        return PERSPECTIVE_COLORS[d.source_perspective] || "#888";
      }
      return BIAS_COLORS[d.source_bias] || "#888";
    }

    // ── Sort sources by bias/region/perspective ──
    var BIAS_ORDER = ["left", "lean-left", "center", "lean-right", "right"];
    var REGION_ORDER = ["us", "canada", "mexico", "uk", "australia", "india", "japan", "korea", "international"];
    var PERSPECTIVE_ORDER = ["consumer", "enterprise", "academic", "culture"];

    function sourceOrder(s) {
      if (isSports && s.source_region) {
        var idx = REGION_ORDER.indexOf(s.source_region);
        return idx >= 0 ? idx : 99;
      }
      if (isTech && s.source_perspective) {
        var idx = PERSPECTIVE_ORDER.indexOf(s.source_perspective);
        return idx >= 0 ? idx : 99;
      }
      var idx = BIAS_ORDER.indexOf(s.source_bias);
      return idx >= 0 ? idx : 99;
    }

    var sortedSources = sourceNodes.slice().sort(function (a, b) {
      return sourceOrder(a) - sourceOrder(b) || a.source_name.localeCompare(b.source_name);
    });

    // ── Container & responsive check ──
    var container = document.getElementById("similarity-web");
    if (!container) return;

    var width = container.clientWidth || 600;
    var isMobile = width < 600;

    // ── Build similarity lookup for both views ──
    var simLookup = {};
    allEdges.forEach(function (e) {
      var a = e.sourceNode.source_name;
      var b = e.targetNode.source_name;
      var key = a < b ? a + "|" + b : b + "|" + a;
      simLookup[key] = e.similarity;
    });

    // ── Branch: matrix on mobile, force graph on desktop ──
    if (isMobile) {
      renderMatrix();
    } else {
      renderForceGraph();
    }

    // ════════════════════════════════════════
    //  MOBILE: Adjacency Matrix
    // ════════════════════════════════════════

    function renderMatrix() {
      var n = sortedSources.length;

      // Color scale: read from CSS variables for theme support
      var rootStyles = getComputedStyle(document.documentElement);
      var matrixLow = rootStyles.getPropertyValue('--matrix-low').trim() || "#e8e4f0";
      var matrixMid = rootStyles.getPropertyValue('--matrix-mid').trim() || "#d97706";
      var matrixHigh = rootStyles.getPropertyValue('--matrix-high').trim() || "#7e22ce";
      var colorScale = d3.scaleLinear()
        .domain([0.65, 0.8, 0.95])
        .range([matrixLow, matrixMid, matrixHigh])
        .clamp(true);

      var wrapper = d3.select(container)
        .append("div")
        .attr("class", "sim-matrix-wrapper");

      var table = wrapper.append("div").attr("class", "sim-matrix");

      // Header row with corner + column headers
      var headerRow = table.append("div").attr("class", "sim-matrix-header-row");
      headerRow.append("div").attr("class", "sim-matrix-corner");

      headerRow.selectAll(".sim-matrix-col-header")
        .data(sortedSources)
        .enter()
        .append("div")
        .attr("class", "sim-matrix-col-header")
        .append("span")
        .text(function (d) { return d.source_name; })
        .style("color", function (d) { return nodeColor(d); });

      // Data rows
      var rows = table.selectAll(".sim-matrix-row")
        .data(sortedSources)
        .enter()
        .append("div")
        .attr("class", "sim-matrix-row");

      // Row header
      rows.append("div")
        .attr("class", "sim-matrix-row-header")
        .text(function (d) { return d.source_name; })
        .style("color", function (d) { return nodeColor(d); });

      // Cells
      rows.each(function (rowSource) {
        var row = d3.select(this);
        row.selectAll(".sim-matrix-cell")
          .data(sortedSources)
          .enter()
          .append("div")
          .attr("class", "sim-matrix-cell")
          .style("background-color", function (colSource) {
            if (rowSource.source_name === colSource.source_name) return "var(--bg-secondary, #eee)";
            var a = rowSource.source_name;
            var b = colSource.source_name;
            var key = a < b ? a + "|" + b : b + "|" + a;
            var sim = simLookup[key];
            if (!sim) return "transparent";
            return colorScale(sim);
          })
          .on("click", function (event, colSource) {
            event.stopPropagation();
            showMatrixInfo(rowSource, colSource);
          });
      });

      // Info panel
      var matrixInfo = d3.select(container)
        .append("div")
        .attr("class", "sim-info-panel")
        .style("display", "none");

      document.addEventListener("click", function () {
        matrixInfo.style("display", "none");
      });

      function showMatrixInfo(rowSrc, colSrc) {
        if (rowSrc.source_name === colSrc.source_name) {
          matrixInfo.html(
            "<strong>" + rowSrc.source_name + "</strong> &middot; " +
            rowSrc.count + " article" + (rowSrc.count > 1 ? "s" : "")
          ).style("display", "block");
          return;
        }
        var a = rowSrc.source_name;
        var b = colSrc.source_name;
        var key = a < b ? a + "|" + b : b + "|" + a;
        var sim = simLookup[key];
        var pairCount = edgeSums[key] ? edgeSums[key].count : 0;

        if (!sim) {
          matrixInfo.html(
            "<strong>" + a + "</strong> &amp; <strong>" + b + "</strong> &middot; No connection"
          ).style("display", "block");
          return;
        }

        matrixInfo.html(
          "<strong>" + a + "</strong> &amp; <strong>" + b + "</strong> &middot; " +
          Math.round(sim * 100) + "% similar &middot; " +
          pairCount + " pair" + (pairCount !== 1 ? "s" : "")
        ).style("display", "block");
      }

      // Hint
      var hint = d3.select(container)
        .append("div")
        .attr("class", "sim-zoom-hint")
        .text("Tap a cell to see source pair details");
      setTimeout(function () {
        hint.transition().duration(600).style("opacity", 0).remove();
      }, 3500);
    }

    // ════════════════════════════════════════
    //  DESKTOP: Force-Directed Graph
    // ════════════════════════════════════════

    function renderForceGraph() {
      var rootStyles = getComputedStyle(document.documentElement);

      // Prune: keep only top 4 strongest edges per source node
      allEdges.sort(function (a, b) { return b.similarity - a.similarity; });
      var edgeCount = {};
      var keptKeys = {};
      var sourceEdges = [];
      var MAX_PER_NODE = 4;

      allEdges.forEach(function (e) {
        var sA = e.sourceNode.source_name;
        var sB = e.targetNode.source_name;
        if ((edgeCount[sA] || 0) >= MAX_PER_NODE && (edgeCount[sB] || 0) >= MAX_PER_NODE) return;
        var key = sA < sB ? sA + "|" + sB : sB + "|" + sA;
        if (keptKeys[key]) return;
        keptKeys[key] = true;
        edgeCount[sA] = (edgeCount[sA] || 0) + 1;
        edgeCount[sB] = (edgeCount[sB] || 0) + 1;
        sourceEdges.push({
          source: e.sourceNode,
          target: e.targetNode,
          similarity: e.similarity,
        });
      });

      if (sourceEdges.length === 0) return;

      var height = Math.min(width * 0.7, 450);

      // Scale forces to viewport
      var scaleFactor = Math.max(width / 600, 0.5);
      var chargeStrength = -500 * scaleFactor;
      var linkBaseDist = 140 * scaleFactor;

      var svg = d3
        .select(container)
        .append("svg")
        .attr("viewBox", "0 0 " + width + " " + height)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .style("width", "100%")
        .style("height", "auto");

      // Zoom/pan
      var graphGroup = svg.append("g").attr("class", "sim-graph-group");

      var zoom = d3.zoom()
        .scaleExtent([0.5, 4])
        .on("zoom", function (event) {
          graphGroup.attr("transform", event.transform);
        });

      svg.call(zoom);

      svg.on("dblclick.zoom", function () {
        svg.transition().duration(300).call(zoom.transform, d3.zoomIdentity);
      });

      // Node size scale
      var maxCount = d3.max(sourceNodes, function (d) { return d.count; });
      var radiusScale = d3.scaleSqrt().domain([1, Math.max(maxCount, 2)]).range([6, 18]);

      // Edge scales
      var simExtent = d3.extent(sourceEdges, function (d) { return d.similarity; });
      var strokeScale = d3.scaleLinear().domain([simExtent[0], simExtent[1]]).range([1, 5]).clamp(true);
      var opacityScale = d3.scaleLinear().domain([simExtent[0], simExtent[1]]).range([0.3, 0.9]).clamp(true);

      // Tooltip
      var tooltip = d3
        .select(container)
        .append("div")
        .attr("class", "sim-tooltip")
        .style("opacity", 0);

      // Focus state
      var focusedNode = null;

      function focusNode(d) {
        if (focusedNode === d) {
          clearFocus();
          return;
        }
        focusedNode = d;

        var neighbors = {};
        sourceEdges.forEach(function (e) {
          if (e.source === d || e.source.source_name === d.source_name) {
            neighbors[(e.target.source_name || e.target)] = true;
          }
          if (e.target === d || e.target.source_name === d.source_name) {
            neighbors[(e.source.source_name || e.source)] = true;
          }
        });

        node.select("circle")
          .transition().duration(200)
          .attr("opacity", function (n) {
            return (n === d || neighbors[n.source_name]) ? 1 : 0.15;
          });

        node.select("text")
          .transition().duration(200)
          .attr("opacity", function (n) {
            return (n === d || neighbors[n.source_name]) ? 1 : 0;
          });

        link
          .transition().duration(200)
          .attr("stroke-opacity", function (e) {
            var eSrc = e.source.source_name || e.source;
            var eTgt = e.target.source_name || e.target;
            return (eSrc === d.source_name || eTgt === d.source_name) ? 0.9 : 0.05;
          });

        // Build info panel content with pair counts
        var connList = [];
        sourceEdges.forEach(function (e) {
          var eSrc = e.source.source_name || e.source;
          var eTgt = e.target.source_name || e.target;
          var pairKey = eSrc < eTgt ? eSrc + "|" + eTgt : eTgt + "|" + eSrc;
          var pairCount = edgeSums[pairKey] ? edgeSums[pairKey].count : 0;
          if (eSrc === d.source_name) connList.push({ name: eTgt, sim: e.similarity, pairs: pairCount });
          if (eTgt === d.source_name) connList.push({ name: eSrc, sim: e.similarity, pairs: pairCount });
        });
        connList.sort(function (a, b) { return b.sim - a.sim; });

        var html = "<strong>" + d.source_name + "</strong> &middot; " +
          d.count + " article" + (d.count > 1 ? "s" : "");
        if (connList.length > 0) {
          html += "<span class='sim-info-connections'>";
          connList.forEach(function (c) {
            html += " &middot; " + c.name + " (" + Math.round(c.sim * 100) + "%, " +
              c.pairs + " pair" + (c.pairs !== 1 ? "s" : "") + ")";
          });
          html += "</span>";
        }

        tooltip
          .html(html)
          .style("opacity", 1)
          .style("left", "0.5rem")
          .style("top", "0.5rem");
      }

      function clearFocus() {
        focusedNode = null;

        node.select("circle")
          .transition().duration(200)
          .attr("opacity", 1);

        node.select("text")
          .transition().duration(200)
          .attr("opacity", 1);

        link
          .transition().duration(200)
          .attr("stroke-opacity", function (d) { return opacityScale(d.similarity); });

        tooltip.style("opacity", 0);
      }

      // Force simulation
      var simulation = d3
        .forceSimulation(sourceNodes)
        .force(
          "link",
          d3
            .forceLink(sourceEdges)
            .distance(function (d) {
              return linkBaseDist * (1.1 - d.similarity);
            })
        )
        .force("charge", d3.forceManyBody().strength(chargeStrength))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide(function (d) {
          return radiusScale(d.count) + 8;
        }));

      // Draw links
      var link = graphGroup
        .append("g")
        .selectAll("line")
        .data(sourceEdges)
        .enter()
        .append("line")
        .attr("stroke", rootStyles.getPropertyValue('--link-stroke').trim() || "#bbb")
        .attr("stroke-opacity", function (d) { return opacityScale(d.similarity); })
        .attr("stroke-width", function (d) {
          return strokeScale(d.similarity);
        });

      // Draw nodes
      var node = graphGroup
        .append("g")
        .selectAll("g")
        .data(sourceNodes)
        .enter()
        .append("g")
        .attr("class", "sim-node")
        .call(
          d3
            .drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended)
        );

      node
        .append("circle")
        .attr("r", function (d) { return radiusScale(d.count); })
        .attr("fill", nodeColor)
        .attr("stroke", rootStyles.getPropertyValue('--node-stroke').trim() || "#fff")
        .attr("stroke-width", 1.5);

      node
        .append("text")
        .attr("class", "sim-label")
        .attr("dx", function (d) { return radiusScale(d.count) + 4; })
        .attr("dy", 4)
        .text(function (d) { return d.source_name; });

      // Desktop hover + click-to-focus
      node
        .on("mouseover", function (event, d) {
          if (focusedNode) return;
          tooltip
            .html(
              "<strong>" + d.source_name + "</strong><br>" +
              d.count + " article" + (d.count > 1 ? "s" : "")
            )
            .style("opacity", 1);
        })
        .on("mousemove", function (event) {
          if (focusedNode) return;
          var rect = container.getBoundingClientRect();
          tooltip
            .style("left", event.clientX - rect.left + 12 + "px")
            .style("top", event.clientY - rect.top - 10 + "px");
        })
        .on("mouseout", function () {
          if (focusedNode) return;
          tooltip.style("opacity", 0);
        })
        .on("click", function (event, d) {
          event.stopPropagation();
          focusNode(d);
        });

      svg.on("click", function () {
        clearFocus();
      });

      // Tick
      simulation.on("tick", function () {
        link
          .attr("x1", function (d) { return d.source.x; })
          .attr("y1", function (d) { return d.source.y; })
          .attr("x2", function (d) { return d.target.x; })
          .attr("y2", function (d) { return d.target.y; });

        node.attr("transform", function (d) {
          var r = radiusScale(d.count);
          d.x = Math.max(r, Math.min(width - r, d.x));
          d.y = Math.max(r, Math.min(height - r, d.y));
          return "translate(" + d.x + "," + d.y + ")";
        });
      });

      function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }

      function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      }

      function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }
    }
  }

  // Initialize when the Sources tab is shown
  tabRadio.addEventListener("change", function () {
    if (tabRadio.checked) {
      requestAnimationFrame(function () {
        requestAnimationFrame(init);
      });
    }
  });
})();
