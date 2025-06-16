import { allNodeData, archiveProgramIds, formatMetrics, renderMetricBar, getHighlightNodes, selectedProgramId, setSelectedProgramId } from './main.js';
import { scrollAndSelectNodeById } from './graph.js';

const sidebar = document.getElementById('sidebar');
export let sidebarSticky = false;
let lastSidebarTab = null;

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
    let locatorBtn = '<button id="sidebar-locator-btn" title="Locate selected node" aria-label="Locate selected node" style="position:absolute;top:0.05em;right:2.5em;font-size:1.5em;background:none;border:none;color:#FFD600;cursor:pointer;z-index:10;line-height:1;filter:drop-shadow(0 0 2px #FFD600);">⦿</button>';
    let closeBtn = '<button id="sidebar-close-btn" style="position:absolute;top:0.05em;right:0.15em;font-size:1.6em;background:none;border:none;color:#888;cursor:pointer;z-index:10;line-height:1;">&times;</button>';
    let openLink = '<div style="text-align:center;margin:-1em 0 1.2em 0;"><a href="/program/' + d.id + '" target="_blank" class="open-in-new" style="font-size:0.95em;">[open in new window]</a></div>';
    let tabHtml = '';
    let tabContentHtml = '';
    let tabNames = [];
    if (d.code && typeof d.code === 'string' && d.code.trim() !== '') tabNames.push('Code');
    if ((d.prompts && typeof d.prompts === 'object' && Object.keys(d.prompts).length > 0) || (d.artifacts_json && typeof d.artifacts_json === 'object' && Object.keys(d.artifacts_json).length > 0)) tabNames.push('Prompts');
    const children = allNodeData.filter(n => n.parent_id === d.id);
    if (children.length > 0) tabNames.push('Children');

    // Handle nodes with "-copyN" IDs
    function getBaseId(id) {
        return id.includes('-copy') ? id.split('-copy')[0] : id;
    }
    const baseId = getBaseId(d.id);
    const clones = allNodeData.filter(n => getBaseId(n.id) === baseId && n.id !== d.id);
    if (clones.length > 0) tabNames.push('Clones');

    let activeTab = lastSidebarTab && tabNames.includes(lastSidebarTab) ? lastSidebarTab : tabNames[0];

    // Helper to render tab content
    function renderSidebarTabContent(tabName, d, children) {
        if (tabName === 'Code') {
            return `<pre class="sidebar-code-pre">${d.code}</pre>`;
        }
        if (tabName === 'Prompts') {
            // Prompt select logic
            let promptOptions = [];
            let promptMap = {};
            if (d.prompts && typeof d.prompts === 'object') {
                for (const [k, v] of Object.entries(d.prompts)) {
                    if (v && typeof v === 'object' && !Array.isArray(v)) {
                        for (const [subKey, subVal] of Object.entries(v)) {
                            const optLabel = `${k} - ${subKey}`;
                            promptOptions.push(optLabel);
                            promptMap[optLabel] = subVal;
                        }
                    } else {
                        const optLabel = `${k}`;
                        promptOptions.push(optLabel);
                        promptMap[optLabel] = v;
                    }
                }
            }
            // Artifacts
            if (d.artifacts_json) {
                const optLabel = `artifacts`;
                promptOptions.push(optLabel);
                promptMap[optLabel] = d.artifacts_json;
            }
            // Get last selected prompt from localStorage, or default to first
            let lastPromptKey = localStorage.getItem('sidebarPromptSelect') || promptOptions[0] || '';
            if (!promptMap[lastPromptKey]) lastPromptKey = promptOptions[0] || '';
            // Build select box
            let selectHtml = '';
            if (promptOptions.length > 1) {
                selectHtml = `<select id="sidebar-prompt-select" style="margin-bottom:0.7em;max-width:100%;font-size:1em;">
                    ${promptOptions.map(opt => `<option value="${opt}"${opt===lastPromptKey?' selected':''}>${opt}</option>`).join('')}
                </select>`;
            }
            // Show only the selected prompt
            let promptVal = promptMap[lastPromptKey];
            let promptHtml = `<pre class="sidebar-pre">${promptVal ?? ''}</pre>`;
            return selectHtml + promptHtml;
        }
        if (tabName === 'Children') {
            const metric = (document.getElementById('metric-select') && document.getElementById('metric-select').value) || 'combined_score';
            let min = 0, max = 1;
            const vals = children.map(child => (child.metrics && typeof child.metrics[metric] === 'number') ? child.metrics[metric] : null).filter(x => x !== null);
            if (vals.length > 0) {
                min = Math.min(...vals);
                max = Math.max(...vals);
            }
            return `<div><ul style='margin:0.5em 0 0 1em;padding:0;'>` +
                children.map(child => {
                    let val = (child.metrics && typeof child.metrics[metric] === 'number') ? child.metrics[metric].toFixed(4) : '(no value)';
                    let bar = (child.metrics && typeof child.metrics[metric] === 'number') ? renderMetricBar(child.metrics[metric], min, max) : '';
                    return `<li style='margin-bottom:0.3em;'><a href="#" class="child-link" data-child="${child.id}">${child.id}</a><br /><br /> <span style='margin-left:0.5em;'>${val}</span> ${bar}</li>`;
                }).join('') +
                `</ul></div>`;
        }
        if (tabName === 'Clones') {
            return `<div><ul style='margin:0.5em 0 0 1em;padding:0;'>` +
                clones.map(clone =>
                    `<li style='margin-bottom:0.3em;'><a href="#" class="clone-link" data-clone="${clone.id}">${clone.id}</a></li>`
                ).join('') +
                `</ul></div>`;
        }
        return '';
    }

    if (tabNames.length > 0) {
        tabHtml = '<div id="sidebar-tab-bar" style="display:flex;gap:0.7em;margin-bottom:0.7em;">' +
            tabNames.map((name) => `<span class="sidebar-tab${name===activeTab?' active':''}" data-tab="${name}">${name}</span>`).join('') + '</div>';
        tabContentHtml = `<div id="sidebar-tab-content">${renderSidebarTabContent(activeTab, d, children)}</div>`;
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

    // Helper to attach prompt select handler
    function attachPromptSelectHandler() {
        const promptSelect = document.getElementById('sidebar-prompt-select');
        if (promptSelect) {
            promptSelect.onchange = function() {
                localStorage.setItem('sidebarPromptSelect', promptSelect.value);
                // Only re-render the Prompts tab, not the whole sidebar
                const tabContent = document.getElementById('sidebar-tab-content');
                if (tabContent) {
                    tabContent.innerHTML = renderSidebarTabContent('Prompts', d, children);
                    attachPromptSelectHandler();
                }
            };
        }
    }
    attachPromptSelectHandler();

    if (tabNames.length > 1) {
        const tabBar = document.getElementById('sidebar-tab-bar');
        Array.from(tabBar.children).forEach(tabEl => {
            tabEl.onclick = function() {
                Array.from(tabBar.children).forEach(e => e.classList.remove('active'));
                tabEl.classList.add('active');
                const tabName = tabEl.dataset.tab;
                lastSidebarTab = tabName;
                const tabContent = document.getElementById('sidebar-tab-content');
                tabContent.innerHTML = renderSidebarTabContent(tabName, d, children);
                if (tabName === 'Prompts') {
                    attachPromptSelectHandler();
                }
                setTimeout(() => {
                    document.querySelectorAll('.child-link').forEach(link => {
                        link.onclick = function(e) {
                            e.preventDefault();
                            const childNode = allNodeData.find(n => n.id == link.dataset.child);
                            if (childNode) {
                                window._lastSelectedNodeData = childNode;
                                const perfTabBtn = document.getElementById('tab-performance');
                                const perfTabView = document.getElementById('view-performance');
                                if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                                    import('./performance.js').then(mod => {
                                        mod.selectPerformanceNodeById(childNode.id);
                                        showSidebar();
                                    });
                                } else {
                                    scrollAndSelectNodeById(childNode.id);
                                }
                            }
                        };
                    });
                    document.querySelectorAll('.clone-link').forEach(link => {
                        link.onclick = function(e) {
                            e.preventDefault();
                            const cloneNode = allNodeData.find(n => n.id == link.dataset.clone);
                            if (cloneNode) {
                                window._lastSelectedNodeData = cloneNode;
                                const perfTabBtn = document.getElementById('tab-performance');
                                const perfTabView = document.getElementById('view-performance');
                                if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                                    import('./performance.js').then(mod => {
                                        mod.selectPerformanceNodeById(cloneNode.id);
                                        showSidebar();
                                    });
                                } else {
                                    scrollAndSelectNodeById(cloneNode.id);
                                }
                            }
                        };
                    });
                }, 0);
            };
        });
    }
    setTimeout(() => {
        attachPromptSelectHandler();
        document.querySelectorAll('.child-link').forEach(link => {
            link.onclick = function(e) {
                e.preventDefault();
                const childNode = allNodeData.find(n => n.id == link.dataset.child);
                if (childNode) {
                    window._lastSelectedNodeData = childNode;
                    // Check if performance tab is active
                    const perfTabBtn = document.getElementById('tab-performance');
                    const perfTabView = document.getElementById('view-performance');
                    if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                        import('./performance.js').then(mod => {
                            mod.selectPerformanceNodeById(childNode.id);
                            showSidebar();
                        });
                    } else {
                        scrollAndSelectNodeById(childNode.id);
                    }
                }
            };
        });
        document.querySelectorAll('.clone-link').forEach(link => {
            link.onclick = function(e) {
                e.preventDefault();
                const cloneNode = allNodeData.find(n => n.id == link.dataset.clone);
                if (cloneNode) {
                    window._lastSelectedNodeData = cloneNode;
                    const perfTabBtn = document.getElementById('tab-performance');
                    const perfTabView = document.getElementById('view-performance');
                    if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                        import('./performance.js').then(mod => {
                            mod.selectPerformanceNodeById(cloneNode.id);
                            showSidebar();
                        });
                    } else {
                        scrollAndSelectNodeById(cloneNode.id);
                    }
                }
            };
        });
    }, 0);
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