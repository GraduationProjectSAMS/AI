from common_imports import plt, nx
from constants import USAGE_NODES, USAGE_EDGES

def draw_usage_diagram():
    G = nx.DiGraph()
    G.add_nodes_from(USAGE_NODES)
    G.add_edges_from(USAGE_EDGES)
    
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="orange", edge_color="black", node_size=3000, font_size=10)
    plt.title("Application Usage Diagram")
    plt.show(block=False)


if __name__ == "__main__":
    draw_usage_diagram()
