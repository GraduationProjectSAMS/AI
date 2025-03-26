from common_imports import plt, nx
from constants import COMMUNICATION_NODES, COMMUNICATION_EDGES

def draw_communication_diagram():
    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(COMMUNICATION_NODES)
    G.add_edges_from(COMMUNICATION_EDGES)

    # Draw the graph
    plt.figure(figsize=(15, 8))
    pos = nx.spring_layout(G, seed=48)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=5000, font_size=12, font_weight="bold")
    plt.title("Application Communication Diagram")
    plt.show(block=False)

if __name__ == "__main__":
    draw_communication_diagram()
