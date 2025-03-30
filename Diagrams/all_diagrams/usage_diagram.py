from common_imports import plt, nx
from constants import USAGE_NODES, USAGE_EDGES

def draw_usage_diagram(ax=None):
    G = nx.DiGraph()
    G.add_nodes_from(USAGE_NODES)
    G.add_edges_from(USAGE_EDGES)

    pos = nx.spring_layout(G, seed=42)

    if ax:
        ax.set_title("Application Usage Diagram")
        nx.draw(G, pos, with_labels=True, node_color="orange", edge_color="black", 
                node_size=3000, font_size=10, ax=ax)
    else:
        plt.figure(figsize=(12, 6))
        nx.draw(G, pos, with_labels=True, node_color="orange", edge_color="black",
                node_size=3000, font_size=10)
        plt.title("Application Usage Diagram")
        plt.show()

if __name__ == "__main__":
    draw_usage_diagram()
