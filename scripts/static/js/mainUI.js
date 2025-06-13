import { width, height } from './state.js';
import { selectedProgramId } from './main.js';
import { selectProgram } from './graph.js';
import { showSidebarContent } from './sidebar.js';

const darkToggleContainer = document.getElementById('darkmode-toggle').parentElement;
const darkToggleInput = document.getElementById('darkmode-toggle');
const darkToggleLabel = document.getElementById('darkmode-label');

if (!document.getElementById('custom-dark-toggle')) {
    const wrapper = document.createElement('label');
    wrapper.className = 'toggle-switch';
    wrapper.id = 'custom-dark-toggle';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.id = 'darkmode-toggle';
    input.checked = darkToggleInput.checked;
    const slider = document.createElement('span');
    slider.className = 'toggle-slider';
    wrapper.appendChild(input);
    wrapper.appendChild(slider);
    darkToggleContainer.replaceChild(wrapper, darkToggleInput);

    darkToggleContainer.appendChild(darkToggleLabel);
    input.addEventListener('change', function() {
        setTheme(this.checked ? 'dark' : 'light');
    });
}

// Tab switching logic
const tabs = ["branching", "performance", "list"];
tabs.forEach(tab => {
    document.getElementById(`tab-${tab}`).addEventListener('click', function() {
        tabs.forEach(t => {
            document.getElementById(`tab-${t}`).classList.remove('active');
            const view = document.getElementById(`view-${t}`);
            if (view) view.style.display = 'none';
        });
        this.classList.add('active');
        const view = document.getElementById(`view-${tab}`);
        if (view) view.style.display = 'block';
        // Synchronize node selection when switching tabs
        if (tab === 'list' || tab === 'branching') {
            if (selectedProgramId) {
                selectProgram(selectedProgramId);
                showSidebarContent(window._lastSelectedNodeData || null);
            }
        }
        // Disable page scroll for graph tabs
        if (tab === 'branching' || tab === 'performance') {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
    });
});

// Dark mode logic
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    document.getElementById('darkmode-toggle').checked = (theme === 'dark');
    document.getElementById('darkmode-label').textContent = theme === 'dark' ? '🌙' : '☀️';
}
function getSystemTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
// On load, use localStorage or system default to determine theme
(function() {
    let theme = localStorage.getItem('theme');
    if (!theme) theme = getSystemTheme();
    setTheme(theme);
})();
document.getElementById('darkmode-toggle').addEventListener('change', function() {
    setTheme(this.checked ? 'dark' : 'light');
});

// Canvas size and zoom setup
let toolbarHeight = document.getElementById('toolbar').offsetHeight;

const svg = d3.select("#graph").append("svg")
    .attr("width", width)
    .attr("height", height)
    .call(d3.zoom()
        .scaleExtent([0.1, 10])
        .on("zoom", (event) => {
            g.attr("transform", event.transform);
        }))
    .on("dblclick.zoom", null);

const g = svg.append("g");
