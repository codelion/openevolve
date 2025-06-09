---
applyTo: 'scripts/**/*.js'
---
- In this program, a dataset with nodes and edges is visualized in different graphs and lists. These view modes are selectable in tabs.
- Nodes are parametrized with meta data including program ID, island number, generation nunmber, parent ID (which is used to determine the edge connections), a metric dataset with flexible keys, a code string, a dict with prompts and more. All data except program ID are optional.
- A sidebar shows detailed node information. Its format is the same across all view modes.
- The sidebar in this program is designed to show up dynamically when a node is selected in one of the graphs or lists. It appears on hover of the node and hides when the node is not hovered anymore.
- A single node can be selected to turn it "sticky". When a node is sticky, its information remains visible in the sidebar and the sidebar remains open until the user clicks in the background. Hovering another node will not change the sidebar content if a node is already sticky.
- The selected node is highlighted with a red border and synchronized across all graphs and lists. I.e., clicking a node in a list will also highlight it in the graphs.

- A select box #highlight-select configures a filter logic that allows to highlight multiple nodes. Nodes are highlighted with a blue shadow in the graphs and lists.
- A select box #metric-select shows the available metrics (determined dynamically from the dataset), and the selected metric may be used in the graph creation, filter and sorting logic.