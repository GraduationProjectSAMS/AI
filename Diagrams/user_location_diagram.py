from common_imports import plt, nx
from constants import LOCATION_NODES, LOCATION_EDGES

# Create User Location Diagram
def draw_user_location_diagram():
    G = nx.DiGraph()
    G.add_nodes_from(LOCATION_NODES)
    G.add_edges_from(LOCATION_EDGES)

    plt.figure(figsize=(12, 6))
    pos2 = nx.spring_layout(G, seed=42)
    nx.draw(G, pos2, with_labels=True, node_color="lightgreen", edge_color="black", node_size=3000, font_size=10, font_weight="bold")
    plt.title("Application & User Location Diagram")
    plt.show(block=False)

if __name__ == "__main__":
    draw_user_location_diagram()