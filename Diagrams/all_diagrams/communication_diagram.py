from common_imports import plt, nx
from constants import COMMUNICATION_NODES, COMMUNICATION_EDGES

def draw_communication_diagram(ax):
    G = nx.DiGraph()
    G.add_nodes_from(COMMUNICATION_NODES)
    G.add_edges_from(COMMUNICATION_EDGES)

    pos = nx.spring_layout(G, seed=48)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
            node_size=4000, font_size=10, font_weight="bold", arrows=True, ax=ax)

    ax.set_title("Application Communication Diagram")
    