import { getHighlightNodes, allNodeData, selectedProgramId, setSelectedProgramId, lastDataStr } from './main.js';
import { width, height } from './state.js';
import { openInNewTab, showSidebarContent, sidebarSticky, showSidebar, setSidebarSticky, hideSidebar } from './sidebar.js';
import { renderNodeList, selectListNodeById } from './list.js';

export function scrollAndSelectNodeById(nodeId) {
    // Helper to get edges from lastDataStr (as in main.js resize)
    function getCurrentEdges() {
        let edges = [];
        if (typeof lastDataStr === 'string') {
            try {
                const parsed = JSON.parse(lastDataStr);
                edges = parsed.edges || [];
            } catch {}
        }
        return edges;
    }
    const container = document.getElementById('node-list-container');
    if (container) {
        const rows = Array.from(container.children);
        const target = rows.find(div => div.getAttribute('data-node-id') === nodeId);
        if (target) {
            target.scrollIntoView({behavior: 'smooth', block: 'center'});
            setSelectedProgramId(nodeId);
            renderNodeList(allNodeData);
            showSidebarContent(allNodeData.find(n => n.id == nodeId));
            showSidebar();
            setSidebarSticky(true);
            selectProgram(selectedProgramId);
            renderGraph({ nodes: allNodeData, edges: getCurrentEdges() }, { centerNodeId: nodeId });
            updateGraphNodeSelection();
            return true;
        }
    }
    const node = allNodeData.find(n => n.id == nodeId);
    if (node) {
        setSelectedProgramId(nodeId);
        showSidebarContent(node);
        showSidebar();
        setSidebarSticky(true);
        selectProgram(selectedProgramId);
        renderGraph({ nodes: allNodeData, edges: getCurrentEdges() }, { centerNodeId: nodeId });
        updateGraphNodeSelection();
        return true;
    }
    return false;
}

export function updateGraphNodeSelection() {
    if (!g) return;
    g.selectAll('circle')
        .attr('stroke', d => selectedProgramId === d.id ? 'red' : '#333')
        .attr('stroke-width', d => selectedProgramId === d.id ? 3 : 1.5)
        .classed('node-selected', d => selectedProgramId === d.id);
    updateGraphEdgeSelection(); // update edge highlight when node selection changes
}

export function getNodeColor(d) {
    if (d.island !== undefined) return d3.schemeCategory10[d.island % 10];
    return getComputedStyle(document.documentElement)
        .getPropertyValue('--node-default').trim() || "#fff";
}

function getSelectedMetric() {
    const metricSelect = document.getElementById('metric-select');
    return metricSelect ? metricSelect.value : 'overall_score';
}

export function getNodeRadius(d) {
    let minScore = Infinity, maxScore = -Infinity;
    let minR = 10, maxR = 32;
    const metric = getSelectedMetric();

    if (Array.isArray(allNodeData) && allNodeData.length > 0) {
        allNodeData.forEach(n => {
            if (n.metrics && typeof n.metrics[metric] === "number") {
                if (n.metrics[metric] < minScore) minScore = n.metrics[metric];
                if (n.metrics[metric] > maxScore) maxScore = n.metrics[metric];
            }
        });
        if (minScore === Infinity) minScore = 0;
        if (maxScore === -Infinity) maxScore = 1;
    } else {
        minScore = 0;
        maxScore = 1;
    }

    let score = d.metrics && typeof d.metrics[metric] === "number" ? d.metrics[metric] : null;
    if (score === null || isNaN(score)) {
        return minR / 2;
    }
    if (maxScore === minScore) return (minR + maxR) / 2;
    score = Math.max(minScore, Math.min(maxScore, score));
    return minR + (maxR - minR) * (score - minScore) / (maxScore - minScore);
}

export function selectProgram(programId) {
    const nodes = g.selectAll("circle");
    nodes.each(function(d) {
        const nodeElem = d3.select(this);
        if (d.id === programId) {
            nodeElem.classed("node-selected", true);
        } else {
            nodeElem.classed("node-selected", false);
        }
        nodeElem.classed("node-hovered", false);
    });
    // Dispatch event for list view sync
    window.dispatchEvent(new CustomEvent('node-selected', { detail: { id: programId } }));
    updateGraphEdgeSelection(); // update edge highlight on selection
}

let svg = null;
let g = null;
let simulation = null; // Keep simulation alive
let zoomBehavior = null; // Ensure zoomBehavior is available for locator

// Ensure window.g is always up to date for static export compatibility
Object.defineProperty(window, 'g', {
    get: function() { return g; },
    set: function(val) { g = val; }
});

// Recenter Button Overlay
function showRecenterButton(onClick) {
    let btn = document.getElementById('graph-recenter-btn');
    if (!btn) {
        btn = document.createElement('button');
        btn.id = 'graph-recenter-btn';
        btn.textContent = 'Recenter';
        btn.style.position = 'absolute';
        btn.style.left = '50%';
        btn.style.top = '50%';
        btn.style.transform = 'translate(-50%, -50%)';
        btn.style.zIndex = 1000;
        btn.style.fontSize = '2em';
        btn.style.padding = '0.5em 1.5em';
        btn.style.background = '#fff';
        btn.style.border = '2px solid #2196f3';
        btn.style.borderRadius = '12px';
        btn.style.boxShadow = '0 2px 16px #0002';
        btn.style.cursor = 'pointer';
        btn.style.display = 'block';
        document.getElementById('graph').appendChild(btn);
    }
    btn.style.display = 'block';
    btn.onclick = function() {
        btn.style.display = 'none';
        if (typeof onClick === 'function') onClick();
    };
}

function hideRecenterButton() {
    const btn = document.getElementById('graph-recenter-btn');
    if (btn) btn.style.display = 'none';
}

function ensureGraphSvg() {
    // Get latest width/height from state.js
    let svgEl = d3.select('#graph').select('svg');
    if (svgEl.empty()) {
        svgEl = d3.select('#graph').append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('id', 'graph-svg');
    } else {
        svgEl.attr('width', width).attr('height', height);
    }
    let gEl = svgEl.select('g');
    if (gEl.empty()) {
        gEl = svgEl.append('g');
    }
    return { svg: svgEl, g: gEl };
}

function applyDragHandlersToAllNodes() {
    if (!g) return;
    g.selectAll('circle').each(function() {
        d3.select(this).on('.drag', null);
        d3.select(this).call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    });
}

function renderGraph(data, options = {}) {
    const { svg: svgEl, g: gEl } = ensureGraphSvg();
    svg = svgEl;
    g = gEl;
    window.g = g; // Ensure global assignment for static export
    if (!g) {
        console.warn('D3 group (g) is null in renderGraph. Aborting render.');
        return;
    }
    // Preserve zoom/pan
    let prevTransform = null;
    if (!svg.empty()) {
        const gZoom = svg.select('g');
        if (!gZoom.empty()) {
            const transform = gZoom.attr('transform');
            if (transform) prevTransform = transform;
        }
    }
    g.selectAll("*").remove();

    // Keep simulation alive and update nodes/links
    if (!simulation) {
        simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.edges).id(d => d.id).distance(80))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2));
    } else {
        simulation.nodes(data.nodes);
        simulation.force("link").links(data.edges);
        simulation.alpha(0.7).restart();
    }

    const link = g.append("g")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .selectAll("line")
        .data(data.edges)
        .enter().append("line")
        .attr("stroke-width", 2);

    const metric = getSelectedMetric();
    const highlightFilter = document.getElementById('highlight-select').value;
    const highlightNodes = getHighlightNodes(data.nodes, highlightFilter, metric);
    const highlightIds = new Set(highlightNodes.map(n => n.id));

    const node = g.append("g")
        .attr("stroke", getComputedStyle(document.documentElement).getPropertyValue('--node-stroke').trim() || "#fff")
        .attr("stroke-width", 1.5)
        .selectAll("circle")
        .data(data.nodes)
        .enter().append("circle")
        .attr("r", d => getNodeRadius(d))
        .attr("fill", d => getNodeColor(d))
        .attr("class", d => [
            highlightIds.has(d.id) ? 'node-highlighted' : '',
            selectedProgramId === d.id ? 'node-selected' : ''
        ].join(' ').trim())
        .attr('stroke', d => selectedProgramId === d.id ? 'red' : (highlightIds.has(d.id) ? '#2196f3' : '#333'))
        .attr('stroke-width', d => selectedProgramId === d.id ? 3 : 1.5)
        .on("click", function(event, d) {
            setSelectedProgramId(d.id);
            setSidebarSticky(true);
            selectListNodeById(d.id); // sync list selection
            g.selectAll('circle')
                .classed('node-hovered', false)
                .classed('node-selected', false)
                .classed('node-highlighted', nd => highlightIds.has(nd.id))
                .classed('node-selected', nd => selectedProgramId === nd.id);
            d3.select(this).classed('node-selected', true);
            showSidebarContent(d, false);
            showSidebar();
            selectProgram(selectedProgramId);
            event.stopPropagation();
            updateGraphNodeSelection(); // Ensure all nodes update selection border
        })
        .on("dblclick", openInNewTab)
        .on("mouseover", function(event, d) {
            if (!sidebarSticky && (!selectedProgramId || selectedProgramId !== d.id)) {
                showSidebarContent(d, true);
                showSidebar();
            }
            d3.select(this)
                .classed('node-hovered', true)
                .attr('stroke', '#FFD600').attr('stroke-width', 4);
        })
        .on("mouseout", function(event, d) {
            d3.select(this)
                .classed('node-hovered', false)
                .attr('stroke', selectedProgramId === d.id ? 'red' : (highlightIds.has(d.id) ? '#2196f3' : '#333'))
                .attr('stroke-width', selectedProgramId === d.id ? 3 : 1.5);
            if (!selectedProgramId) {
                hideSidebar();
            }
        });

    node.append("title").text(d => d.id);

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
        updateGraphEdgeSelection(); // update edge highlight on tick
    });

    // Intelligent zoom/pan
    const zoomBehavior = d3.zoom()
        .scaleExtent([0.2, 10])
        .on('zoom', function(event) {
            g.attr('transform', event.transform);
            // Check if all content is out of view
            setTimeout(() => {
                try {
                    const svgRect = svg.node().getBoundingClientRect();
                    const allCircles = g.selectAll('circle').nodes();
                    if (allCircles.length === 0) { hideRecenterButton(); return; }
                    let anyVisible = false;
                    for (const c of allCircles) {
                        const bbox = c.getBoundingClientRect();
                        if (
                            bbox.right > svgRect.left &&
                            bbox.left < svgRect.right &&
                            bbox.bottom > svgRect.top &&
                            bbox.top < svgRect.bottom
                        ) {
                            anyVisible = true;
                            break;
                        }
                    }
                    if (!anyVisible) {
                        showRecenterButton(() => {
                            // Reset zoom/pan
                            svg.transition().duration(400).call(zoomBehavior.transform, d3.zoomIdentity);
                        });
                    } else {
                        hideRecenterButton();
                    }
                } catch {}
            }, 0);
        });
    svg.call(zoomBehavior);
    if (prevTransform) {
        g.attr('transform', prevTransform);
        const t = d3.zoomTransform(g.node());
        svg.call(zoomBehavior.transform, t);
    } else if (options.fitToNodes) {
        setTimeout(() => {
            try {
                const allCircles = g.selectAll('circle').nodes();
                if (allCircles.length > 0) {
                    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                    allCircles.forEach(c => {
                        const bbox = c.getBBox();
                        minX = Math.min(minX, bbox.x);
                        minY = Math.min(minY, bbox.y);
                        maxX = Math.max(maxX, bbox.x + bbox.width);
                        maxY = Math.max(maxY, bbox.y + bbox.height);
                    });
                    const pad = 40;
                    minX -= pad; minY -= pad; maxX += pad; maxY += pad;
                    const graphW = svg.attr('width');
                    const graphH = svg.attr('height');
                    const scale = Math.min(graphW / (maxX - minX), graphH / (maxY - minY), 1);
                    const tx = (graphW - scale * (minX + maxX)) / 2;
                    const ty = (graphH - scale * (minY + maxY)) / 2;
                    const t = d3.zoomIdentity.translate(tx, ty).scale(scale);
                    svg.transition().duration(400).call(zoomBehavior.transform, t);
                }
            } catch {}
        }, 0);
    } else if (options.centerNodeId) {
        setTimeout(() => {
            try {
                const node = g.selectAll('circle').filter(d => d.id == options.centerNodeId).node();
                if (node) {
                    const bbox = node.getBBox();
                    const graphW = svg.attr('width');
                    const graphH = svg.attr('height');
                    const scale = Math.min(graphW / (bbox.width * 6), graphH / (bbox.height * 6), 1.5);
                    const tx = graphW/2 - scale * (bbox.x + bbox.width/2);
                    const ty = graphH/2 - scale * (bbox.y + bbox.height/2);
                    const t = d3.zoomIdentity.translate(tx, ty).scale(scale);
                    svg.transition().duration(400).call(zoomBehavior.transform, t);
                }
            } catch {}
        }, 0);
    }

    selectProgram(selectedProgramId);
    updateGraphEdgeSelection(); // update edge highlight after render
    applyDragHandlersToAllNodes();

    svg.on("click", function(event) {
        if (event.target === svg.node()) {
            setSelectedProgramId(null);
            setSidebarSticky(false);
            hideSidebar();
            g.selectAll("circle")
                .classed("node-selected", false)
                .classed("node-hovered", false)
                .attr("stroke", function(d) { return (highlightIds.has(d.id) ? '#2196f3' : '#333'); })
                .attr("stroke-width", 1.5);
            selectListNodeById(null);
        }
    });
}

export function animateGraphNodeAttributes() {
    if (!g) return;
    const metric = getSelectedMetric();
    const filter = document.getElementById('highlight-select').value;
    const highlightNodes = getHighlightNodes(allNodeData, filter, metric);
    const highlightIds = new Set(highlightNodes.map(n => n.id));
    g.selectAll('circle')
        .transition().duration(400)
        .attr('r', d => getNodeRadius(d))
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', d => selectedProgramId === d.id ? 'red' : (highlightIds.has(d.id) ? '#2196f3' : '#333'))
        .attr('stroke-width', d => selectedProgramId === d.id ? 3 : 1.5)
        .attr('opacity', 1)
        .on('end', null)
        .selection()
        .each(function(d) {
            d3.select(this)
                .classed('node-highlighted', highlightIds.has(d.id))
                .classed('node-selected', selectedProgramId === d.id);
        });
    setTimeout(applyDragHandlersToAllNodes, 420);
}

export function centerAndHighlightNodeInGraph(nodeId) {
    if (!g || !svg) return;
    // Ensure zoomBehavior is available and is a function
    if (!zoomBehavior || typeof zoomBehavior !== 'function') {
        zoomBehavior = d3.zoom()
            .scaleExtent([0.2, 10])
            .on('zoom', function(event) {
                g.attr('transform', event.transform);
            });
        svg.call(zoomBehavior);
    }
    const nodeSel = g.selectAll('circle').filter(d => d.id == nodeId);
    if (!nodeSel.empty()) {
        // Pan/zoom to node
        const node = nodeSel.node();
        const bbox = node.getBBox();
        const graphW = svg.attr('width');
        const graphH = svg.attr('height');
        const scale = Math.min(graphW / (bbox.width * 6), graphH / (bbox.height * 6), 1.5);
        const tx = graphW/2 - scale * (bbox.x + bbox.width/2);
        const ty = graphH/2 - scale * (bbox.y + bbox.height/2);
        const t = d3.zoomIdentity.translate(tx, ty).scale(scale);
        // Use the correct D3 v7 API for programmatic zoom
        svg.transition().duration(400).call(zoomBehavior.transform, t);
        // Yellow shadow highlight
        nodeSel.each(function() {
            const el = d3.select(this);
            el.classed('node-locator-highlight', true)
                .style('filter', 'drop-shadow(0 0 16px 8px #FFD600)');
            el.transition().duration(350).style('filter', 'drop-shadow(0 0 24px 16px #FFD600)')
                .transition().duration(650).style('filter', null)
                .on('end', function() { el.classed('node-locator-highlight', false); });
        });
    }
}

export function updateGraphEdgeSelection() {
    if (!g) return;
    g.selectAll('line')
        .attr('stroke', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 'red' : '#999')
        .attr('stroke-width', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 4 : 2)
        .attr('stroke-opacity', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 0.95 : 0.6);
}

function dragstarted(event, d) {
    if (!event.active && simulation) simulation.alphaTarget(0.3).restart(); // Keep simulation alive
    d.fx = d.x;
    d.fy = d.y;
}
function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}
function dragended(event, d) {
    if (!event.active && simulation) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

window.addEventListener('node-selected', function(e) {
    // When node selection changes (e.g., from list view), update graph node selection
    updateGraphNodeSelection();
});

export { renderGraph, g };
