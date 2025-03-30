from common_imports import plt, nx
from constants import LOCATION_NODES, LOCATION_EDGES

def draw_user_location_diagram(ax=None):
    G = nx.DiGraph()
    G.add_nodes_from(LOCATION_NODES)
    G.add_edges_from(LOCATION_EDGES)

    pos = nx.spring_layout(G, seed=42)

    if ax:
        ax.set_title("Application & User Location Diagram")
        nx.draw(G, pos, with_labels=True, node_color="lightgreen", edge_color="black",
                node_size=3000, font_size=10, font_weight="bold", ax=ax)
    else:
        plt.figure(figsize=(12, 6))
        nx.draw(G, pos, with_labels=True, node_color="lightgreen", edge_color="black",
                node_size=3000, font_size=10, font_weight="bold")
        plt.title("Application & User Location Diagram")
        plt.show()

if __name__ == "__main__":
    draw_user_location_diagram()
