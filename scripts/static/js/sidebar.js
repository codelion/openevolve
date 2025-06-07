import { allNodeData, archiveProgramIds, formatMetrics, renderMetricBar, getHighlightNodes, selectedProgramId, setSelectedProgramId } from './main.js';
import { scrollAndSelectNodeById } from './graph.js';

const sidebar = document.getElementById('sidebar');
export let sidebarSticky = false;

export function showSidebar() {
    sidebar.style.transform = 'translateX(0)';
}
export function hideSidebar() {
    sidebar.style.transform = 'translateX(100%)';
    sidebarSticky = false;
}

export function showSidebarContent(d, fromHover = false) {
    const sidebarContent = document.getElementById('sidebar-content');
    if (!sidebarContent) return;
    if (fromHover && sidebarSticky) return;
    if (!d) {
        sidebarContent.innerHTML = '';
        return;
    }
    let starHtml = '';
    if (archiveProgramIds && archiveProgramIds.includes(d.id)) {
        starHtml = '<span style="position:relative;top:0.05em;left:0.15em;font-size:1.6em;color:#FFD600;z-index:10;" title="MAP-elites member" aria-label="MAP-elites member">★</span>';
    }
    // Locator icon button (left of close X)
    let locatorBtn = '<button id="sidebar-locator-btn" title="Locate selected node" aria-label="Locate selected node" style="position:absolute;top:0.05em;right:2.5em;font-size:1.5em;background:none;border:none;color:#FFD600;cursor:pointer;z-index:10;line-height:1;filter:drop-shadow(0 0 2px #FFD600);">⦿</button>';
    let closeBtn = '<button id="sidebar-close-btn" style="position:absolute;top:0.05em;right:0.15em;font-size:1.6em;background:none;border:none;color:#888;cursor:pointer;z-index:10;line-height:1;">&times;</button>';
    let openLink = '<div style="text-align:center;margin:-1em 0 1.2em 0;"><a href="/program/' + d.id + '" target="_blank" class="open-in-new" style="font-size:0.95em;">[open in new window]</a></div>';
    let tabHtml = '';
    let tabContentHtml = '';
    let tabNames = [];
    if (d.code && typeof d.code === 'string' && d.code.trim() !== '') tabNames.push('Code');
    if (d.prompts && typeof d.prompts === 'object' && Object.keys(d.prompts).length > 0) tabNames.push('Prompts');
    if (tabNames.length > 0) {
        tabHtml = '<div id="sidebar-tab-bar" style="display:flex;gap:0.7em;margin-bottom:0.7em;">' +
            tabNames.map((name, i) => `<span class="sidebar-tab${i===0?' active':''}" data-tab="${name}">${name}</span>`).join('') + '</div>';
        tabContentHtml = '<div id="sidebar-tab-content">';
        if (tabNames[0] === 'Code') tabContentHtml += `<pre class="sidebar-code-pre">${d.code}</pre>`;
        if (tabNames[0] === 'Prompts') {
            for (const [k, v] of Object.entries(d.prompts)) {
                tabContentHtml += `<div style="margin-bottom:0.7em;"><b>${k}:</b><pre class="sidebar-pre">${v}</pre></div>`;
            }
        }
        tabContentHtml += '</div>';
    }
    let parentIslandHtml = '';
    if (d.parent_id && d.parent_id !== 'None') {
        const parent = allNodeData.find(n => n.id == d.parent_id);
        if (parent && parent.island !== undefined) {
            parentIslandHtml = ` <span style="color:#888;font-size:0.92em;">(island ${parent.island})</span>`;
        }
    }
    sidebarContent.innerHTML =
        `<div style="position:relative;min-height:2em;">
            ${starHtml}
            ${locatorBtn}
            ${closeBtn}
            ${openLink}
            <b>Program ID:</b> ${d.id}<br>
            <b>Island:</b> ${d.island}<br>
            <b>Generation:</b> ${d.generation}<br>
            <b>Parent ID:</b> <a href="#" class="parent-link" data-parent="${d.parent_id || ''}">${d.parent_id || 'None'}</a>${parentIslandHtml}<br><br>
            <b>Metrics:</b><br>${formatMetrics(d.metrics)}<br><br>
            ${tabHtml}${tabContentHtml}
        </div>`;
    if (tabNames.length > 1) {
        const tabBar = document.getElementById('sidebar-tab-bar');
        Array.from(tabBar.children).forEach(tabEl => {
            tabEl.onclick = function() {
                Array.from(tabBar.children).forEach(e => e.classList.remove('active'));
                tabEl.classList.add('active');
                const tabName = tabEl.dataset.tab;
                const tabContent = document.getElementById('sidebar-tab-content');
                if (tabName === 'Code') tabContent.innerHTML = `<pre class="sidebar-code-pre">${d.code}</pre>`;
                if (tabName === 'Prompts') {
                    let html = '';
                    for (const [k, v] of Object.entries(d.prompts)) {
                        html += `<div style="margin-bottom:0.7em;"><b>${k}:</b><pre class="sidebar-pre">${v}</pre></div>`;
                    }
                    tabContent.innerHTML = html;
                }
            };
        });
    }
    const closeBtnEl = document.getElementById('sidebar-close-btn');
    if (closeBtnEl) closeBtnEl.onclick = function() {
        setSelectedProgramId(null);
        sidebarSticky = false;
        hideSidebar();
    };
    // Locator button logic
    const locatorBtnEl = document.getElementById('sidebar-locator-btn');
    if (locatorBtnEl) {
        locatorBtnEl.onclick = function(e) {
            e.preventDefault();
            // Use view display property for active view detection
            const viewBranching = document.getElementById('view-branching');
            const viewPerformance = document.getElementById('view-performance');
            const viewList = document.getElementById('view-list');
            if (viewBranching && viewBranching.style.display !== 'none') {
                import('./graph.js').then(mod => {
                    mod.centerAndHighlightNodeInGraph(d.id);
                });
            } else if (viewPerformance && viewPerformance.style.display !== 'none') {
                import('./performance.js').then(mod => {
                    mod.centerAndHighlightNodeInPerformanceGraph(d.id);
                });
            } else if (viewList && viewList.style.display !== 'none') {
                // Scroll to list item
                const container = document.getElementById('node-list-container');
                if (container) {
                    const rows = Array.from(container.children);
                    const target = rows.find(div => div.getAttribute('data-node-id') === d.id);
                    if (target) {
                        target.scrollIntoView({behavior: 'smooth', block: 'center'});
                        // Optionally add a yellow highlight effect
                        target.classList.add('node-locator-highlight');
                        setTimeout(() => target.classList.remove('node-locator-highlight'), 1000);
                    }
                }
            }
        };
    }
    // Parent link logic
    const parentLink = sidebarContent.querySelector('.parent-link');
    if (parentLink && parentLink.dataset.parent && parentLink.dataset.parent !== 'None' && parentLink.dataset.parent !== '') {
        parentLink.onclick = function(e) {
            e.preventDefault();
            const parentNode = allNodeData.find(n => n.id == parentLink.dataset.parent);
            if (parentNode) {
                window._lastSelectedNodeData = parentNode;
            }
            const perfTabBtn = document.getElementById('tab-performance');
            const perfTabView = document.getElementById('view-performance');
            if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                import('./performance.js').then(mod => {
                    mod.selectPerformanceNodeById(parentLink.dataset.parent);
                    showSidebar();
                });
            } else {
                scrollAndSelectNodeById(parentLink.dataset.parent);
            }
        };
    }
}

export function openInNewTab(event, d) {
    const url = `/program/${d.id}`;
    window.open(url, '_blank');
    event.stopPropagation();
}

export function setSidebarSticky(val) {
    sidebarSticky = val;
}