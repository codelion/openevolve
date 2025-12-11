"""
Huawei CodeCraft 2017 - CDN Server Deployment Optimization
Initial baseline solution using greedy heuristics
"""
import sys
from typing import List, Tuple, Dict, Set
import heapq


class NetworkGraph:
    """Network graph representation"""
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.edges: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(num_nodes)}

    def add_edge(self, u: int, v: int, bandwidth: int, cost: int):
        """Add bidirectional edge with bandwidth and cost"""
        self.edges[u].append((v, bandwidth, cost))
        self.edges[v].append((u, bandwidth, cost))

    def dijkstra(self, start: int, end: int) -> Tuple[List[int], int]:
        """Find shortest path by cost"""
        dist = [float('inf')] * self.num_nodes
        parent = [-1] * self.num_nodes
        dist[start] = 0

        pq = [(0, start)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == end:
                break

            for v, bw, cost in self.edges[u]:
                if dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    parent[v] = u
                    heapq.heappush(pq, (dist[v], v))

        # Reconstruct path
        if parent[end] == -1 and start != end:
            return [], float('inf')

        path = []
        curr = end
        while curr != -1:
            path.append(curr)
            curr = parent[curr]
        path.reverse()

        return path, dist[end]


# EVOLVE-BLOCK-START
def solve_cdn_deployment(
    num_nodes: int,
    edges: List[Tuple[int, int, int, int]],  # (u, v, bandwidth, cost_per_unit)
    consumers: List[Tuple[int, int, int]],  # (consumer_id, connected_node, demand)
    server_types: List[Tuple[int, int, int]],  # (type_id, capacity, hardware_cost)
    node_deploy_costs: List[Tuple[int, int]]  # (node_id, deploy_cost)
) -> Tuple[int, List[Tuple[List[int], int, int, int]]]:
    """
    Solve CDN server deployment problem using simple greedy strategy

    Strategy: Deploy one server per consumer at their connected node
    This avoids bandwidth constraint issues by minimizing path lengths

    Returns:
        total_cost: Total deployment and bandwidth cost
        paths: List of (path, consumer_id, bandwidth, server_type)
    """
    # Create deploy cost map
    deploy_cost_map = {node_id: cost for node_id, cost in node_deploy_costs}

    # Create server type map
    server_map = {type_id: (capacity, hw_cost) for type_id, capacity, hw_cost in server_types}

    total_cost = 0
    paths = []
    deployed_servers: Dict[int, Tuple[int, int, int]] = {}  # node -> (server_type, capacity, used)

    # Sort consumers by demand (highest first) for better server type selection
    sorted_consumers = sorted(consumers, key=lambda x: x[2], reverse=True)

    for consumer_id, connected_node, demand in sorted_consumers:
        server_node = connected_node

        # Check if we already have a server at this node
        if server_node not in deployed_servers:
            # Choose appropriate server type
            # Pick the smallest server type that can handle this consumer
            chosen_type = None
            for type_id, (capacity, hw_cost) in sorted(server_map.items(), key=lambda x: x[1][0]):
                if capacity >= demand:
                    chosen_type = type_id
                    break

            if chosen_type is None:
                # Use largest server if none fit
                chosen_type = max(server_map.keys(), key=lambda t: server_map[t][0])

            capacity, hw_cost = server_map[chosen_type]
            deploy_cost = deploy_cost_map.get(server_node, 0)

            # Add deployment cost
            total_cost += hw_cost + deploy_cost

            # Record deployed server
            deployed_servers[server_node] = (chosen_type, capacity, 0)

        # Get server info
        server_type, capacity, used = deployed_servers[server_node]

        # Check if server can handle this consumer
        if used + demand > capacity:
            # Need to upgrade or add another server
            # For simplicity, deploy a new larger server
            # Find a server type that can handle total demand
            total_demand = used + demand

            for type_id, (cap, hw_cost) in sorted(server_map.items(), key=lambda x: x[1][0]):
                if cap >= total_demand:
                    # Only pay additional cost if upgrading
                    old_hw_cost = server_map[server_type][1]
                    if hw_cost > old_hw_cost:
                        total_cost += (hw_cost - old_hw_cost)

                    server_type = type_id
                    capacity = cap
                    break

            deployed_servers[server_node] = (server_type, capacity, total_demand)
        else:
            # Server can handle it
            deployed_servers[server_node] = (server_type, capacity, used + demand)

        # Path is just the server node itself (server and consumer at same node)
        path = [server_node]

        # No bandwidth cost since server is co-located with consumer
        # Record path
        paths.append((path, consumer_id, demand, server_type))

    return total_cost, paths
# EVOLVE-BLOCK-END


def parse_input(input_text: str) -> dict:
    """Parse input file format"""
    lines = [line.strip() for line in input_text.strip().split('\n') if line.strip()]
    idx = 0

    # First line: num_nodes, num_edges, num_consumers
    parts = lines[idx].split()
    num_nodes, num_edges, num_consumers = int(parts[0]), int(parts[1]), int(parts[2])
    idx += 1

    # Server types
    server_types = []
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            type_id, capacity, hw_cost = int(parts[0]), int(parts[1]), int(parts[2])
            server_types.append((type_id, capacity, hw_cost))
            idx += 1
        else:
            break

    # Node deploy costs
    node_deploy_costs = []
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            node_id, deploy_cost = int(parts[0]), int(parts[1])
            node_deploy_costs.append((node_id, deploy_cost))
            idx += 1
        else:
            break

    # Edges
    edges = []
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) == 4 and all(p.isdigit() for p in parts):
            u, v, bw, cost = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            edges.append((u, v, bw, cost))
            idx += 1
        else:
            break

    # Consumers
    consumers = []
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            consumer_id, connected_node, demand = int(parts[0]), int(parts[1]), int(parts[2])
            consumers.append((consumer_id, connected_node, demand))
            idx += 1
        else:
            break

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_consumers': num_consumers,
        'server_types': server_types,
        'node_deploy_costs': node_deploy_costs,
        'edges': edges,
        'consumers': consumers
    }


def format_output(paths: List[Tuple[List[int], int, int, int]]) -> str:
    """Format output according to specification"""
    output_lines = [str(len(paths)), ""]

    for path, consumer_id, bandwidth, server_type in paths:
        path_str = " ".join(map(str, path))
        line = f"{path_str} {consumer_id} {bandwidth} {server_type}"
        output_lines.append(line)

    return "\n".join(output_lines)


def run_solution(input_text: str) -> Tuple[int, str]:
    """Main entry point for evolution"""
    data = parse_input(input_text)

    total_cost, paths = solve_cdn_deployment(
        data['num_nodes'],
        data['edges'],
        data['consumers'],
        data['server_types'],
        data['node_deploy_costs']
    )

    output_text = format_output(paths)
    return total_cost, output_text


if __name__ == "__main__":
    # Read from stdin or file
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        input_text = sys.stdin.read()

    total_cost, output_text = run_solution(input_text)
    print(f"Total cost: {total_cost}")
    print(output_text)
