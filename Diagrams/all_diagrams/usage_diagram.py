from common_imports import plt, nx
from constants import USAGE_NODES, USAGE_EDGES

def draw_usage_diagram(ax):
    G = nx.DiGraph()
    G.add_nodes_from(USAGE_NODES)
    G.add_edges_from(USAGE_EDGES)

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="orange", edge_color="black",
            node_size=3000, font_size=10, font_weight="bold", arrows=True, ax=ax)

    ax.set_title("Application Usage Diagram")
