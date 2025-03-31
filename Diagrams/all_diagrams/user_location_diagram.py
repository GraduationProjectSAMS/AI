from common_imports import plt, nx
from constants import LOCATION_NODES, LOCATION_EDGES

def draw_user_location_diagram(ax):
    G = nx.DiGraph()
    G.add_nodes_from(LOCATION_NODES)
    G.add_edges_from(LOCATION_EDGES)

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightgreen", edge_color="black",
            node_size=3000, font_size=10, font_weight="bold", arrows=True, ax=ax)

    ax.set_title("Application & User Location Diagram")
