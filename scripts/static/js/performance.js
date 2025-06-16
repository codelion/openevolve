import { allNodeData, archiveProgramIds, formatMetrics, renderMetricBar, getHighlightNodes, getSelectedMetric, selectedProgramId, setSelectedProgramId } from './main.js';
import { getNodeRadius, getNodeColor, selectProgram, scrollAndSelectNodeById } from './graph.js';
import { hideSidebar, sidebarSticky, showSidebarContent, showSidebar, setSidebarSticky } from './sidebar.js';
import { selectListNodeById } from './list.js';

(function() {
    window.addEventListener('DOMContentLoaded', function() {
        const perfDiv = document.getElementById('view-performance');
        if (!perfDiv) return;
        let toggleDiv = document.getElementById('perf-island-toggle');
        if (!toggleDiv) {
            toggleDiv = document.createElement('div');
            toggleDiv.id = 'perf-island-toggle';
            toggleDiv.style = 'display:flex;align-items:center;gap:0.7em;margin-left:3em;';
            toggleDiv.innerHTML = `
            <label class="toggle-switch">
                <input type="checkbox" id="show-islands-toggle">
                <span class="toggle-slider"></span>
            </label>
            <span style="font-weight:500;font-size:1.08em;">Show islands</span>
            `;
            perfDiv.insertBefore(toggleDiv, perfDiv.firstChild);
        }
        function animatePerformanceGraphAttributes() {
            const svg = d3.select('#performance-graph');
            if (svg.empty()) return;
            const g = svg.select('g.zoom-group');
            if (g.empty()) return;
            const metric = getSelectedMetric();
            const highlightFilter = document.getElementById('highlight-select').value;
            const showIslands = document.getElementById('show-islands-toggle')?.checked;
            const nodes = allNodeData;
            const validNodes = nodes.filter(n => n.metrics && typeof n.metrics[metric] === 'number');
            const undefinedNodes = nodes.filter(n => !n.metrics || n.metrics[metric] == null || isNaN(n.metrics[metric]));
            let islands = [];
            if (showIslands) {
                islands = Array.from(new Set(nodes.map(n => n.island))).sort((a,b)=>a-b);
            } else {
                islands = [null];
            }
            const yExtent = d3.extent(nodes, d => d.generation);
            const minGen = 0;
            const maxGen = yExtent[1];
            const margin = {top: 60, right: 40, bottom: 40, left: 60};
            let undefinedBoxWidth = 70;
            const undefinedBoxPad = 54;
            const graphXOffset = undefinedBoxWidth + undefinedBoxPad;
            const width = +svg.attr('width');
            const height = +svg.attr('height');
            const graphHeight = Math.max(400, (maxGen - minGen + 1) * 48 + margin.top + margin.bottom);
            let yScales = {};
            islands.forEach((island, i) => {
                yScales[island] = d3.scaleLinear()
                    .domain([minGen, maxGen]).nice()
                    .range([margin.top + i*graphHeight, margin.top + (i+1)*graphHeight - margin.bottom]);
            });
            const xExtent = d3.extent(validNodes, d => d.metrics[metric]);
            const x = d3.scaleLinear()
                .domain([xExtent[0], xExtent[1]]).nice()
                .range([margin.left+graphXOffset, width - margin.right]);
            const highlightNodes = getHighlightNodes(nodes, highlightFilter, metric);
            const highlightIds = new Set(highlightNodes.map(n => n.id));
            // Animate valid nodes
            g.selectAll('circle')
                .filter(function(d) { return validNodes.includes(d); })
                .transition().duration(400)
                .attr('cx', d => x(d.metrics[metric]))
                .attr('cy', d => showIslands ? yScales[d.island](d.generation) : yScales[null](d.generation))
                .attr('r', d => getNodeRadius(d))
                .attr('fill', d => getNodeColor(d))
                .attr('stroke', d => selectedProgramId === d.id ? 'red' : (highlightIds.has(d.id) ? '#2196f3' : '#333'))
                .attr('stroke-width', d => selectedProgramId === d.id ? 3 : 1.5)
                .attr('opacity', 0.85)
                .on('end', null)
                .selection()
                .each(function(d) {
                    d3.select(this)
                        .classed('node-highlighted', highlightIds.has(d.id))
                        .classed('node-selected', selectedProgramId === d.id);
                });
            // Animate undefined nodes (NaN box)
            g.selectAll('circle')
                .filter(function(d) { return undefinedNodes.includes(d); })
                .transition().duration(400)
                .attr('cx', d => d._nanX || (margin.left + undefinedBoxWidth/2))
                .attr('cy', d => yScales[showIslands ? d.island : null](d.generation))
                .attr('r', d => getNodeRadius(d))
                .attr('fill', d => getNodeColor(d))
                .attr('stroke', d => selectedProgramId === d.id ? 'red' : '#333')
                .attr('stroke-width', d => selectedProgramId === d.id ? 3 : 1.5)
                .attr('opacity', 0.85)
                .on('end', null)
                .selection()
                .each(function(d) {
                    d3.select(this)
                        .classed('node-selected', selectedProgramId === d.id);
                });
            // Animate edges
            const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));
            const edges = nodes.filter(n => n.parent_id && nodeById[n.parent_id]).map(n => {
                return {
                    source: nodeById[n.parent_id],
                    target: n
                };
            });
            g.selectAll('line.performance-edge')
                .data(edges, d => d.target.id)
                .transition().duration(400)
                .attr('x1', d => {
                    const m = d.source.metrics && typeof d.source.metrics[metric] === 'number' ? d.source.metrics[metric] : null;
                    if (m === null || isNaN(m)) {
                        return margin.left + undefinedBoxWidth/2;
                    } else {
                        return x(m);
                    }
                })
                .attr('y1', d => {
                    const m = d.source.metrics && typeof d.source.metrics[metric] === 'number' ? d.source.metrics[metric] : null;
                    const island = showIslands ? d.source.island : null;
                    return yScales[island](d.source.generation);
                })
                .attr('x2', d => {
                    const m = d.target.metrics && typeof d.target.metrics[metric] === 'number' ? d.target.metrics[metric] : null;
                    if (m === null || isNaN(m)) {
                        return margin.left + undefinedBoxWidth/2;
                    } else {
                        return x(m);
                    }
                })
                .attr('y2', d => {
                    const m = d.target.metrics && typeof d.target.metrics[metric] === 'number' ? d.target.metrics[metric] : null;
                    const island = showIslands ? d.target.island : null;
                    return yScales[island](d.target.generation);
                })
                .attr('stroke', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 'red' : '#888')
                .attr('stroke-width', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 3 : 1.5)
                .attr('opacity', 0.5);
        }
        const metricSelect = document.getElementById('metric-select');
        metricSelect.addEventListener('change', function() {
            updatePerformanceGraph(allNodeData);
            setTimeout(updateEdgeHighlighting, 0); // ensure edges update after node positions change
        });
        const highlightSelect = document.getElementById('highlight-select');
        highlightSelect.addEventListener('change', function() {
            animatePerformanceGraphAttributes();
            setTimeout(updateEdgeHighlighting, 0); // ensure edges update after animation
        });
        document.getElementById('tab-performance').addEventListener('click', function() {
            if (typeof allNodeData !== 'undefined' && allNodeData.length) {
                updatePerformanceGraph(allNodeData, {autoZoom: true});
                setTimeout(() => { zoomPerformanceGraphToFit(); }, 0);
            }
        });
        // Show islands yes/no toggle event
        document.getElementById('show-islands-toggle').addEventListener('change', function() {
            updatePerformanceGraph(allNodeData);
        });
        // Responsive resize
        window.addEventListener('resize', function() {
            if (typeof allNodeData !== 'undefined' && allNodeData.length && perfDiv.style.display !== 'none') {
                updatePerformanceGraph(allNodeData);
            }
        });
        window.updatePerformanceGraph = updatePerformanceGraph;

        // Initial render
        if (typeof allNodeData !== 'undefined' && allNodeData.length) {
            updatePerformanceGraph(allNodeData);
            // Zoom to fit after initial render
            setTimeout(() => {
                zoomPerformanceGraphToFit();
            }, 0);
        }
    });
})();

// Recenter Button Overlay
function showRecenterButton(onClick) {
    let btn = document.getElementById('performance-recenter-btn');
    if (!btn) {
        btn = document.createElement('button');
        btn.id = 'performance-recenter-btn';
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
        document.getElementById('view-performance').appendChild(btn);
    }
    btn.style.display = 'block';
    btn.onclick = function() {
        btn.style.display = 'none';
        if (typeof onClick === 'function') onClick();
    };
}
function hideRecenterButton() {
    const btn = document.getElementById('performance-recenter-btn');
    if (btn) btn.style.display = 'none';
}

// Select a node by ID and update graph and sidebar
export function selectPerformanceNodeById(id, opts = {}) {
    setSelectedProgramId(id);
    setSidebarSticky(true);
    // Dispatch event for list view sync
    window.dispatchEvent(new CustomEvent('node-selected', { detail: { id } }));
    if (typeof allNodeData !== 'undefined' && allNodeData.length) {
        updatePerformanceGraph(allNodeData, opts);
        const node = allNodeData.find(n => n.id == id);
        if (node) showSidebarContent(node, false);
    }
}

export function centerAndHighlightNodeInPerformanceGraph(nodeId) {
    if (!g || !svg) return;
    // Ensure zoomBehavior is available and is a function
    if (!zoomBehavior || typeof zoomBehavior !== 'function') {
        zoomBehavior = d3.zoom()
            .scaleExtent([0.2, 10])
            .on('zoom', function(event) {
                g.attr('transform', event.transform);
                lastTransform = event.transform;
            });
        svg.call(zoomBehavior);
    }
    // Try both valid and NaN nodes
    let nodeSel = g.selectAll('circle.performance-node').filter(d => d.id == nodeId);
    if (nodeSel.empty()) {
        nodeSel = g.selectAll('circle.performance-nan').filter(d => d.id == nodeId);
    }
    if (!nodeSel.empty()) {
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

let svg = null;
let g = null;
let zoomBehavior = null;
let lastTransform = null;

function autoZoomPerformanceGraph(nodes, x, yScales, islands, graphHeight, margin, undefinedBoxWidth, width, svg, g) {
    // Compute bounding box for all nodes (including NaN box)
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    // Valid nodes
    nodes.forEach(n => {
        let cx, cy;
        if (n.metrics && typeof n.metrics[getSelectedMetric()] === 'number') {
            cx = x(n.metrics[getSelectedMetric()]);
            cy = yScales[document.getElementById('show-islands-toggle')?.checked ? n.island : null](n.generation);
        } else if (typeof n._nanX === 'number') {
            cx = n._nanX;
            cy = yScales[document.getElementById('show-islands-toggle')?.checked ? n.island : null](n.generation);
        }
        if (typeof cx === 'number' && typeof cy === 'number') {
            minX = Math.min(minX, cx);
            maxX = Math.max(maxX, cx);
            minY = Math.min(minY, cy);
            maxY = Math.max(maxY, cy);
        }
    });
    // Include NaN box
    minX = Math.min(minX, margin.left);
    // Add some padding
    const padX = 60, padY = 60;
    minX -= padX; maxX += padX; minY -= padY; maxY += padY;
    const svgW = +svg.attr('width');
    const svgH = +svg.attr('height');
    const scale = Math.min(svgW / (maxX - minX), svgH / (maxY - minY), 1.5);
    const tx = svgW/2 - scale * (minX + (maxX-minX)/2);
    const ty = svgH/2 - scale * (minY + (maxY-minY)/2);
    const t = d3.zoomIdentity.translate(tx, ty).scale(scale);
    svg.transition().duration(500).call(zoomBehavior.transform, t);
}

function updatePerformanceGraph(nodes, options = {}) {
    // Get or create SVG
    if (!svg) {
        svg = d3.select('#performance-graph');
        if (svg.empty()) {
            svg = d3.select('#view-performance')
                .append('svg')
                .attr('id', 'performance-graph')
                .style('display', 'block');
        }
    }
    // Get or create group
    g = svg.select('g.zoom-group');
    if (g.empty()) {
        g = svg.append('g').attr('class', 'zoom-group');
    }
    // Setup zoom behavior only once
    if (!zoomBehavior) {
        zoomBehavior = d3.zoom()
            .scaleExtent([0.2, 10])
            .on('zoom', function(event) {
                g.attr('transform', event.transform);
                lastTransform = event.transform;
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
    }
    // Reapply last transform after update
    if (lastTransform) {
        svg.call(zoomBehavior.transform, lastTransform);
    }
    // Add SVG background click handler for unselect
    svg.on('click', function(event) {
        if (event.target === svg.node()) {
            setSelectedProgramId(null);
            setSidebarSticky(false);
            hideSidebar();
            // Remove selection from all nodes
            g.selectAll('circle.performance-node, circle.performance-nan')
                .classed('node-selected', false)
                .attr('stroke', function(d) {
                    // Use highlight color if highlighted, else default
                    const highlightFilter = document.getElementById('highlight-select').value;
                    const highlightNodes = getHighlightNodes(nodes, highlightFilter, getSelectedMetric());
                    const highlightIds = new Set(highlightNodes.map(n => n.id));
                    return highlightIds.has(d.id) ? '#2196f3' : '#333';
                })
                .attr('stroke-width', 1.5);
            selectListNodeById(null);
            setTimeout(updateEdgeHighlighting, 0); // ensure edges update after selectedProgramId is null
        }
    });
    // Sizing
    const sidebarEl = document.getElementById('sidebar');
    const padding = 32;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const toolbarHeight = document.getElementById('toolbar').offsetHeight;
    const sidebarWidth = sidebarEl.offsetWidth || 400;
    const width = Math.max(windowWidth - sidebarWidth - padding, 400);
    const metric = getSelectedMetric();
    const validNodes = nodes.filter(n => n.metrics && typeof n.metrics[metric] === 'number');
    const undefinedNodes = nodes.filter(n => !n.metrics || n.metrics[metric] == null || isNaN(n.metrics[metric]));
    const showIslands = document.getElementById('show-islands-toggle')?.checked;
    let islands = [];
    if (showIslands) {
        islands = Array.from(new Set(nodes.map(n => n.island))).sort((a,b)=>a-b);
    } else {
        islands = [null];
    }
    const yExtent = d3.extent(nodes, d => d.generation);
    const minGen = 0;
    const maxGen = yExtent[1];
    const margin = {top: 60, right: 40, bottom: 40, left: 60};
    let undefinedBoxWidth = 70;
    const undefinedBoxPad = 54;
    const genCount = (maxGen - minGen + 1) || 1;
    const graphHeight = Math.max(400, genCount * 48 + margin.top + margin.bottom);
    const totalGraphHeight = showIslands ? (graphHeight * islands.length) : graphHeight;
    const svgHeight = Math.max(windowHeight - toolbarHeight - 24, totalGraphHeight);
    const graphXOffset = undefinedBoxWidth + undefinedBoxPad;
    svg.attr('width', width).attr('height', svgHeight);
    // Remove old axes/labels
    g.selectAll('.axis, .axis-label, .island-label, .nan-label, .nan-box').remove();
    // Y scales per island
    let yScales = {};
    islands.forEach((island, i) => {
        yScales[island] = d3.scaleLinear()
            .domain([minGen, maxGen]).nice()
            .range([margin.top + i*graphHeight, margin.top + (i+1)*graphHeight - margin.bottom]);
        // Y axis
        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(${margin.left+graphXOffset},0)`)
            .call(d3.axisLeft(yScales[island]).ticks(Math.min(12, genCount)));
        // Y axis label (always at start of main graph)
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', `rotate(-90)`) // vertical
            .attr('y', margin.left + graphXOffset + 8)
            .attr('x', -(margin.top + i*graphHeight + (graphHeight - margin.top - margin.bottom)/2))
            .attr('dy', '-2.2em')
            .attr('text-anchor', 'middle')
            .attr('font-size', '1em')
            .attr('fill', '#888')
            .text('Generation');
        // Island label
        if (showIslands) {
            g.append('text')
                .attr('class', 'island-label')
                .attr('x', (width + undefinedBoxWidth) / 2)
                .attr('y', margin.top + i*graphHeight + 38)
                .attr('text-anchor', 'middle')
                .attr('font-size', '2.1em')
                .attr('font-weight', 700)
                .attr('fill', '#444')
                .attr('pointer-events', 'none')
                .text(`Island ${island}`);
        }
    });
    // X axis
    const xExtent = d3.extent(validNodes, d => d.metrics[metric]);
    const x = d3.scaleLinear()
        .domain([xExtent[0], xExtent[1]]).nice()
        .range([margin.left+graphXOffset, width - margin.right]);
    // Remove old x axis and label only
    g.selectAll('.x-axis, .x-axis-label').remove();
    // Add x axis group
    g.append('g')
        .attr('class', 'axis x-axis')
        .attr('transform', `translate(0,${margin.top})`)
        .call(d3.axisTop(x));
    // Add x axis label
    g.append('text')
        .attr('class', 'x-axis-label')
        .attr('x', (width + undefinedBoxWidth) / 2)
        .attr('y', margin.top - 28) // just below the axis
        .attr('fill', '#888')
        .attr('text-anchor', 'middle')
        .attr('font-size', '1.1em')
        .text(metric);
    // NaN box
    if (undefinedNodes.length) {
        // Group NaN nodes by (generation, island)
        const nanGroups = {};
        undefinedNodes.forEach(n => {
            const key = `${n.generation}|${showIslands ? n.island : ''}`;
            if (!nanGroups[key]) nanGroups[key] = [];
            nanGroups[key].push(n);
        });
        // Find max group size
        const maxGroupSize = Math.max(...Object.values(nanGroups).map(g => g.length));
        // Box width should be based on the full intended spread, not the reduced spread
        const spreadWidth = Math.max(38, 24 * maxGroupSize);
        undefinedBoxWidth = spreadWidth/2 + 32; // 16px padding on each side
        // Add a fixed offset so the NaN box is further left of the main graph
        const nanBoxGap = 64; // px gap between NaN box and main graph
        const nanBoxRight = margin.left + graphXOffset - nanBoxGap;
        const nanBoxLeft = nanBoxRight - undefinedBoxWidth;
        const boxTop = margin.top;
        const boxBottom = showIslands ? (margin.top + islands.length*graphHeight - margin.bottom) : (margin.top + graphHeight - margin.bottom);
        g.append('text')
            .attr('class', 'nan-label')
            .attr('x', nanBoxLeft + undefinedBoxWidth/2)
            .attr('y', boxTop - 10)
            .attr('text-anchor', 'middle')
            .attr('font-size', '0.92em')
            .attr('fill', '#888')
            .text('NaN');
        g.append('rect')
            .attr('class', 'nan-box')
            .attr('x', nanBoxLeft)
            .attr('y', boxTop)
            .attr('width', undefinedBoxWidth)
            .attr('height', boxBottom - boxTop)
            .attr('fill', 'none')
            .attr('stroke', '#bbb')
            .attr('stroke-width', 1.5)
            .attr('rx', 12);
        // Assign x offset for each NaN node (spread only in the center half of the box)
        undefinedNodes.forEach(n => {
            const key = `${n.generation}|${showIslands ? n.island : ''}`;
            const group = nanGroups[key];
            if (!group) return;
            if (group.length === 1) {
                n._nanX = nanBoxLeft + undefinedBoxWidth/2;
            } else {
                const idx = group.indexOf(n);
                const innerSpread = spreadWidth / 2; // only use half the box for node spread
                const innerStart = nanBoxLeft + (undefinedBoxWidth - innerSpread) / 2;
                n._nanX = innerStart + innerSpread * (idx + 0.5) / group.length;
            }
        });
    }
    // Data join for edges
    const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));
    const edges = nodes.filter(n => n.parent_id && nodeById[n.parent_id]).map(n => ({ source: nodeById[n.parent_id], target: n }));
    // Remove all old edges before re-adding (fixes missing/incorrect edges after metric change)
    g.selectAll('line.performance-edge').remove();
    // Helper to get x/y for a node (handles NaN and valid nodes)
    function getNodeXY(node, x, yScales, showIslands, metric) {
        // Returns [x, y] for a node, handling both valid and NaN nodes
        if (!node) return [null, null];
        const y = yScales[showIslands ? node.island : null](node.generation);
        if (node.metrics && typeof node.metrics[metric] === 'number') {
            return [x(node.metrics[metric]), y];
        } else if (typeof node._nanX === 'number') {
            return [node._nanX, y];
        } else {
            // fallback: center of NaN box if _nanX not set
            // This should not happen, but fallback for safety
            return [x.range()[0] - 100, y];
        }
    }
    g.selectAll('line.performance-edge')
        .data(edges, d => d.target.id)
        .enter()
        .append('line')
        .attr('class', 'performance-edge')
        .attr('stroke', '#888')
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.5)
        .attr('x1', d => getNodeXY(d.source, x, yScales, showIslands, metric)[0])
        .attr('y1', d => getNodeXY(d.source, x, yScales, showIslands, metric)[1])
        .attr('x2', d => getNodeXY(d.target, x, yScales, showIslands, metric)[0])
        .attr('y2', d => getNodeXY(d.target, x, yScales, showIslands, metric)[1])
        .attr('stroke', d => {
            if (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) {
                return 'red';
            }
            return '#888';
        })
        .attr('stroke-width', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 3 : 1.5)
        .attr('opacity', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 0.9 : 0.5);
    // Ensure edge highlighting updates after node selection
    function updateEdgeHighlighting() {
        g.selectAll('line.performance-edge')
            .attr('stroke', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 'red' : '#888')
            .attr('stroke-width', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 3 : 1.5)
            .attr('opacity', d => (selectedProgramId && (d.source.id === selectedProgramId || d.target.id === selectedProgramId)) ? 0.9 : 0.5);
    }
    updateEdgeHighlighting();

    // Data join for nodes
    const highlightFilter = document.getElementById('highlight-select').value;
    const highlightNodes = getHighlightNodes(nodes, highlightFilter, metric);
    const highlightIds = new Set(highlightNodes.map(n => n.id));
    const nodeSel = g.selectAll('circle.performance-node')
        .data(validNodes, d => d.id);
    nodeSel.enter()
        .append('circle')
        .attr('class', 'performance-node')
        .attr('cx', d => x(d.metrics[metric]))
        .attr('cy', d => showIslands ? yScales[d.island](d.generation) : yScales[null](d.generation))
        .attr('r', d => getNodeRadius(d))
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', d => selectedProgramId === d.id ? 'red' : (highlightIds.has(d.id) ? '#2196f3' : '#333'))
        .attr('stroke-width', d => selectedProgramId === d.id ? 3 : 1.5)
        .attr('opacity', 0.85)
        .on('mouseover', function(event, d) {
            if (!sidebarSticky && (!selectedProgramId || selectedProgramId !== d.id)) {
                showSidebarContent(d, true);
                showSidebar();
            }
            d3.select(this)
                .classed('node-hovered', true)
                .attr('stroke', '#FFD600').attr('stroke-width', 4);
        })
        .on('mouseout', function(event, d) {
            d3.select(this)
                .classed('node-hovered', false)
                .attr('stroke', selectedProgramId === d.id ? 'red' : (highlightIds.has(d.id) ? '#2196f3' : '#333'))
                .attr('stroke-width', selectedProgramId === d.id ? 3 : 1.5);
            if (!selectedProgramId) {
                hideSidebar();
            }
        })
        .on('click', function(event, d) {
            event.preventDefault();
            setSelectedProgramId(d.id);
            window._lastSelectedNodeData = d;
            setSidebarSticky(true);
            selectListNodeById(d.id);
            g.selectAll('circle.performance-node').classed('node-hovered', false).classed('node-selected', false)
                .attr('stroke', function(nd) {
                    return selectedProgramId === nd.id ? 'red' : (highlightIds.has(nd.id) ? '#2196f3' : '#333');
                })
                .attr('stroke-width', function(nd) {
                    return selectedProgramId === nd.id ? 3 : 1.5;
                });
            d3.select(this).classed('node-selected', true);
            showSidebarContent(d, false);
            showSidebar();
            selectProgram(selectedProgramId);
            updateEdgeHighlighting();
        })
        .merge(nodeSel)
        .transition().duration(500)
        .attr('cx', d => x(d.metrics[metric]))
        .attr('cy', d => showIslands ? yScales[d.island](d.generation) : yScales[null](d.generation))
        .attr('r', d => getNodeRadius(d))
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', d => selectedProgramId === d.id ? 'red' : (highlightIds.has(d.id) ? '#2196f3' : '#333'))
        .attr('stroke-width', d => selectedProgramId === d.id ? 3 : 1.5)
        .attr('opacity', 0.85)
        .on('end', null)
        .selection()
        .each(function(d) {
            d3.select(this)
                .classed('node-highlighted', highlightIds.has(d.id))
                .classed('node-selected', selectedProgramId === d.id);
        });
    nodeSel.exit().transition().duration(300).attr('opacity', 0).remove();
    // Data join for NaN nodes
    const nanSel = g.selectAll('circle.performance-nan')
        .data(undefinedNodes, d => d.id);
    nanSel.enter()
        .append('circle')
        .attr('class', 'performance-nan')
        .attr('cx', d => d._nanX)
        .attr('cy', d => yScales[showIslands ? d.island : null](d.generation))
        .attr('r', d => getNodeRadius(d))
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', d => selectedProgramId === d.id ? 'red' : '#333')
        .attr('stroke-width', d => selectedProgramId === d.id ? 3 : 1.5)
        .attr('opacity', 0.85)
        .on('mouseover', function(event, d) {
            if (!sidebarSticky && (!selectedProgramId || selectedProgramId !== d.id)) {
                showSidebarContent(d, true);
                showSidebar();
            }
            d3.select(this)
                .classed('node-hovered', true)
                .attr('stroke', '#FFD600').attr('stroke-width', 4);
        })
        .on('mouseout', function(event, d) {
            d3.select(this)
                .classed('node-hovered', false)
                .attr('stroke', selectedProgramId === d.id ? 'red' : '#333')
                .attr('stroke-width', selectedProgramId === d.id ? 3 : 1.5);
            if (!selectedProgramId) {
                hideSidebar();
            }
        })
        .on('click', function(event, d) {
            event.preventDefault();
            setSelectedProgramId(d.id);
            window._lastSelectedNodeData = d;
            setSidebarSticky(true);
            selectListNodeById(d.id);
            g.selectAll('circle.performance-nan').classed('node-hovered', false).classed('node-selected', false)
                .attr('stroke', function(nd) {
                    return selectedProgramId === nd.id ? 'red' : '#333';
                })
                .attr('stroke-width', function(nd) {
                    return selectedProgramId === nd.id ? 3 : 1.5;
                });
            d3.select(this).classed('node-selected', true);
            showSidebarContent(d, false);
            showSidebar();
            selectProgram(selectedProgramId);
            updateEdgeHighlighting();
        })
        .merge(nanSel)
        .transition().duration(500)
        .attr('cx', d => d._nanX)
        .attr('cy', d => yScales[showIslands ? d.island : null](d.generation))
        .attr('r', d => getNodeRadius(d))
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', d => selectedProgramId === d.id ? 'red' : '#333')
        .attr('stroke-width', d => selectedProgramId === d.id ? 3 : 1.5)
        .attr('opacity', 0.85)
        .on('end', null)
        .selection()
        .each(function(d) {
            d3.select(this)
                .classed('node-selected', selectedProgramId === d.id);
        });
    nanSel.exit().transition().duration(300).attr('opacity', 0).remove();
    // Auto-zoom to fit on initial render or when requested
    if (options.autoZoom || (!lastTransform && nodes.length)) {
        autoZoomPerformanceGraph(nodes, x, yScales, islands, graphHeight, margin, undefinedBoxWidth, width, svg, g);
    }
}

// Zoom-to-fit helper
function zoomPerformanceGraphToFit() {
    if (!svg || !g) return;
    // Get all node positions (valid and NaN)
    const nodeCircles = g.selectAll('circle.performance-node, circle.performance-nan').nodes();
    if (!nodeCircles.length) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    nodeCircles.forEach(node => {
        const bbox = node.getBBox();
        minX = Math.min(minX, bbox.x);
        minY = Math.min(minY, bbox.y);
        maxX = Math.max(maxX, bbox.x + bbox.width);
        maxY = Math.max(maxY, bbox.y + bbox.height);
    });
    // Also include the NaN box if present
    const nanBox = g.select('rect.nan-box').node();
    if (nanBox) {
        const bbox = nanBox.getBBox();
        minX = Math.min(minX, bbox.x);
        minY = Math.min(minY, bbox.y);
        maxX = Math.max(maxX, bbox.x + bbox.width);
        maxY = Math.max(maxY, bbox.y + bbox.height);
    }
    // Add some padding
    const pad = 32;
    minX -= pad; minY -= pad; maxX += pad; maxY += pad;
    const graphW = svg.attr('width');
    const graphH = svg.attr('height');
    // Bias the center to the left so the left edge is always visible
    // Instead of centering on the middle, center at 35% from the left
    const centerFrac = 0.35;
    const centerX = minX + (maxX - minX) * centerFrac;
    const centerY = minY + (maxY - minY) / 2;
    const scale = Math.min(graphW / (maxX - minX), graphH / (maxY - minY), 1.5);
    const tx = graphW/2 - scale * centerX;
    const ty = graphH/2 - scale * centerY;
    const t = d3.zoomIdentity.translate(tx, ty).scale(scale);
    svg.transition().duration(400).call(zoomBehavior.transform, t);
    lastTransform = t;
}
