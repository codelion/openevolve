import { allNodeData, archiveProgramIds, formatMetrics, renderMetricBar, getHighlightNodes, getSelectedMetric, setAllNodeData, selectedProgramId, setSelectedProgramId } from './main.js';
import { showSidebar, setSidebarSticky, showSidebarContent } from './sidebar.js';
import { selectProgram, scrollAndSelectNodeById } from './graph.js';
import { selectPerformanceNodeById } from './performance.js';

export function renderNodeList(nodes) {
    setAllNodeData(nodes);
    const container = document.getElementById('node-list-container');
    if (!container) return;
    const search = document.getElementById('list-search').value.trim().toLowerCase();
    const sort = document.getElementById('list-sort').value;
    let filtered = nodes;
    if (search) {
        filtered = nodes.filter(n => (n.id + '').toLowerCase().includes(search));
    }
    const metric = getSelectedMetric();
    if (sort === 'id') {
        filtered = filtered.slice().sort((a, b) => (a.id + '').localeCompare(b.id + ''));
    } else if (sort === 'generation') {
        filtered = filtered.slice().sort((a, b) => (a.generation || 0) - (b.generation || 0));
    } else if (sort === 'island') {
        filtered = filtered.slice().sort((a, b) => (a.island || 0) - (b.island || 0));
    } else if (sort === 'score') {
        filtered = filtered.slice().sort((a, b) => {
            const aScore = a.metrics && typeof a.metrics[metric] === 'number' ? a.metrics[metric] : -Infinity;
            const bScore = b.metrics && typeof b.metrics[metric] === 'number' ? b.metrics[metric] : -Infinity;
            return bScore - aScore;
        });
    }
    const highlightFilter = document.getElementById('highlight-select').value;
    const highlightNodes = getHighlightNodes(nodes, highlightFilter, metric);
    const highlightIds = new Set(highlightNodes.map(n => n.id));
    const allScores = nodes.map(n => (n.metrics && typeof n.metrics[metric] === 'number') ? n.metrics[metric] : null).filter(x => x !== null && !isNaN(x));
    const minScore = allScores.length ? Math.min(...allScores) : 0;
    const maxScore = allScores.length ? Math.max(...allScores) : 1;
    const topScore = allScores.length ? Math.max(...allScores) : 0;
    const avgScore = allScores.length ? (allScores.reduce((a, b) => a + b, 0) / allScores.length) : 0;

    let summaryBar = document.getElementById('list-summary-bar');
    if (!summaryBar) {
        summaryBar = document.createElement('div');
        summaryBar.id = 'list-summary-bar';
        summaryBar.className = 'list-summary-bar';
        container.parentElement.insertBefore(summaryBar, container);
    }
    summaryBar.innerHTML = `
      <div class="summary-block">
        <span class="summary-icon">🏆</span>
        <span class="summary-label">Top score</span>
        <span class="summary-value">${topScore.toFixed(4)}</span>
        ${renderMetricBar(topScore, minScore, maxScore)}
      </div>
      <div class="summary-block">
        <span class="summary-icon">📊</span>
        <span class="summary-label">Average</span>
        <span class="summary-value">${avgScore.toFixed(4)}</span>
        ${renderMetricBar(avgScore, minScore, maxScore)}
        <span style="margin-left:1.2em;font-size:0.98em;color:#888;vertical-align:middle;">
          <span title="Total programs, generations, islands">📦</span> Total: ${nodes.length} programs, ${new Set(nodes.map(n => n.generation)).size} generations, ${new Set(nodes.map(n => n.island)).size} islands
        </span>
      </div>
    `;
    container.innerHTML = '';
    filtered.forEach((node, idx) => {
        const row = document.createElement('div');
        row.className = 'node-list-item' + (selectedProgramId === node.id ? ' selected' : '') + (highlightIds.has(node.id) ? ' highlighted' : '');
        row.setAttribute('data-node-id', node.id);
        row.tabIndex = 0;

        const numDiv = document.createElement('div');
        numDiv.textContent = `#${idx + 1}`;
        numDiv.style.fontSize = '2.2em';
        numDiv.style.fontWeight = 'bold';
        numDiv.style.color = '#444';
        numDiv.style.flex = '0 0 70px';
        numDiv.style.display = 'flex';
        numDiv.style.alignItems = 'center';
        numDiv.style.justifyContent = 'center';
        row.appendChild(numDiv);
        let selectedMetricRow = '';
        if (node.metrics && metric in node.metrics) {
            let val = (typeof node.metrics[metric] === 'number' && isFinite(node.metrics[metric])) ? node.metrics[metric].toFixed(4) : node.metrics[metric];
            let allVals = nodes.map(n => (n.metrics && typeof n.metrics[metric] === 'number') ? n.metrics[metric] : null).filter(x => x !== null && isFinite(x));
            let minV = allVals.length ? Math.min(...allVals) : 0;
            let maxV = allVals.length ? Math.max(...allVals) : 1;
            selectedMetricRow = `<div class="node-info-row">
                <span class="node-info-label" style="font-weight:bold;margin-bottom:1.5em;">${metric}:</span>
                <span class="node-info-value" style="margin-bottom:1.5em;display:inline-block;">
                  <span style="margin-right:0.7em;">${val}</span>
                  <span style="display:inline-block;vertical-align:middle;min-width:60px;">${renderMetricBar(node.metrics[metric], minV, maxV)}</span>
                </span>
            </div>`;
        }
        const infoBlock = document.createElement('div');
        infoBlock.className = 'node-info-block';
        infoBlock.innerHTML = `
            <div class="node-info-table">
                ${selectedMetricRow}
                <div class="node-info-row"><span class="node-info-label">ID:</span><span class="node-info-value">${node.id}</span></div>
                <div class="node-info-row"><span class="node-info-label">Gen:</span><span class="node-info-value">${node.generation ?? ''}</span></div>
                <div class="node-info-row"><span class="node-info-label">Island:</span><span class="node-info-value">${node.island ?? ''}</span></div>
                <div class="node-info-row"><span class="node-info-label">Parent:</span><span class="node-info-value"><a href="#" class="parent-link" data-parent="${node.parent_id ?? ''}">${node.parent_id ?? 'None'}</a></span></div>
            </div>
        `;
        let metricsHtml = '<div class="metrics-block">';
        if (node.metrics) {
            Object.entries(node.metrics).forEach(([k, v]) => {
                if (k === metric) return; // skip selected metric
                let val = (typeof v === 'number' && isFinite(v)) ? v.toFixed(4) : v;
                let allVals = nodes.map(n => (n.metrics && typeof n.metrics[k] === 'number') ? n.metrics[k] : null).filter(x => x !== null && isFinite(x));
                let minV = allVals.length ? Math.min(...allVals) : 0;
                let maxV = allVals.length ? Math.max(...allVals) : 1;
                metricsHtml += `<div class="metric-row"><span class="metric-label">${k}:</span> <span class="metric-value">${val}</span>${renderMetricBar(v, minV, maxV)}</div>`;
            });
        }
        metricsHtml += '</div>';
        // Flexbox layout: info block | metrics block
        row.style.display = 'flex';
        row.style.alignItems = 'stretch';
        row.style.gap = '32px';
        row.style.padding = '12px 8px 0 2em';
        row.style.margin = '0 0 10px 0';
        row.style.borderRadius = '8px';
        row.style.border = selectedProgramId === node.id ? '2.5px solid red' : '1.5px solid #4442';
        row.style.boxShadow = highlightIds.has(node.id) ? '0 0 0 2px #2196f3' : 'none';
        row.style.background = '';
        infoBlock.style.flex = '0 0 auto';
        const metricsBlock = document.createElement('div');
        metricsBlock.innerHTML = metricsHtml;
        metricsBlock.className = 'metrics-block-outer';
        metricsBlock.style.flex = '1 1 0%';
        row.appendChild(infoBlock);

        let openLink = `<a href="/program/${node.id}" target="_blank" class="open-in-new" style="font-size:0.95em;">[open in new window]</a>`;
        const openDiv = document.createElement('div');
        openDiv.style.textAlign = 'center';
        openDiv.style.margin = '-0.5em 0 0.5em 0';
        openDiv.innerHTML = openLink;
        row.appendChild(openDiv);

        row.appendChild(metricsBlock);

        row.onclick = (e) => {
            if (e.target.tagName === 'A') return;
            if (selectedProgramId !== node.id) {
                setSelectedProgramId(node.id);
                window._lastSelectedNodeData = node;
                setSidebarSticky(true);
                renderNodeList(allNodeData);
                showSidebarContent(node, false);
                showSidebarListView();
                selectProgram(node.id);
                selectPerformanceNodeById(node.id);
            }
        };
        // Parent link logic for list
        setTimeout(() => {
            const parentLink = row.querySelector('.parent-link');
            if (parentLink && parentLink.dataset.parent && parentLink.dataset.parent !== 'None' && parentLink.dataset.parent !== '') {
                parentLink.onclick = function(e) {
                    e.preventDefault();
                    scrollAndSelectNodeById(parentLink.dataset.parent);
                };
            }
        }, 0);
        container.appendChild(row);
    });
    container.focus();
    // Scroll to selected node if present
    const selected = container.querySelector('.node-list-item.selected');
    if (selected) {
        selected.scrollIntoView({behavior: 'smooth', block: 'center'});
    }
}
export function selectListNodeById(id) {
    setSelectedProgramId(id);
    renderNodeList(allNodeData);
    const node = allNodeData.find(n => n.id == id);
    if (node) {
        window._lastSelectedNodeData = node;
        setSidebarSticky(true);
        showSidebarContent(node, false);
        showSidebarListView();
    }
}

// List search/sort events
if (document.getElementById('list-search')) {
    document.getElementById('list-search').addEventListener('input', () => renderNodeList(allNodeData));
}
if (document.getElementById('list-sort')) {
    document.getElementById('list-sort').addEventListener('change', () => renderNodeList(allNodeData));
}

// Highlight select event
const highlightSelect = document.getElementById('highlight-select');
highlightSelect.addEventListener('change', function() {
    renderNodeList(allNodeData);
});

if (document.getElementById('list-sort')) {
    document.getElementById('list-sort').value = 'score';
}

const viewList = document.getElementById('view-list');
const sidebarEl = document.getElementById('sidebar');
export function updateListSidebarLayout() {
    if (viewList.style.display !== 'none') {
        sidebarEl.style.transform = 'translateX(0)';
        viewList.style.marginRight = (sidebarEl.offsetWidth+100) + 'px';
    } else {
        viewList.style.marginRight = '0';
    }
}

function showSidebarListView() {
    if (viewList.style.display !== 'none') {
        sidebarEl.style.transform = 'translateX(0)';
        viewList.style.marginRight = (sidebarEl.offsetWidth+100) + 'px';
    } else {
        showSidebar();
    }
}

// Sync selection when switching to list tab
const tabListBtn = document.getElementById('tab-list');
if (tabListBtn) {
    tabListBtn.addEventListener('click', () => {
        renderNodeList(allNodeData);
    });
}

// Keyboard navigation for up/down in list view
const nodeListContainer = document.getElementById('node-list-container');
if (nodeListContainer) {
    nodeListContainer.tabIndex = 0;
    nodeListContainer.addEventListener('keydown', function(e) {
        if (!['ArrowUp', 'ArrowDown'].includes(e.key)) return;
        e.preventDefault(); // Always prevent default to avoid browser scroll
        const items = Array.from(nodeListContainer.querySelectorAll('.node-list-item'));
        if (!items.length) return;
        let idx = items.findIndex(item => item.classList.contains('selected'));
        if (idx === -1) idx = 0;
        if (e.key === 'ArrowUp' && idx > 0) idx--;
        if (e.key === 'ArrowDown' && idx < items.length - 1) idx++;
        const nextItem = items[idx];
        if (nextItem) {
            const nodeId = nextItem.getAttribute('data-node-id');
            selectListNodeById(nodeId);
            nextItem.focus();
            nextItem.scrollIntoView({behavior: 'smooth', block: 'center'});
            // Also scroll the page if needed
            const rect = nextItem.getBoundingClientRect();
            if (rect.top < 0 || rect.bottom > window.innerHeight) {
                window.scrollTo({top: window.scrollY + rect.top - 100, behavior: 'smooth'});
            }
        }
    });
    // Focus container on click to enable keyboard nav
    nodeListContainer.addEventListener('click', function() {
        nodeListContainer.focus();
    });
}

// Listen for node selection events from other views and sync selection in the list view
window.addEventListener('node-selected', function(e) {
    // e.detail should contain the selected node id
    if (e.detail && e.detail.id) {
        setSelectedProgramId(e.detail.id);
        renderNodeList(allNodeData);
    }
});