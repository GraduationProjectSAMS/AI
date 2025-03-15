# Re-import necessary libraries after execution state reset
import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define nodes (system modules)
nodes = [
    "Mobile App (Flutter)", "Web Interface", "Backend (Laravel)", "Database (MySQL)",
    "AI Recommendation Engine", "Payment Gateway", "Order Management System",
    "Inventory Management System", "Shipping & Logistics API", "Supplier API"
]

# Add nodes to the graph
G.add_nodes_from(nodes)

# Define edges (interactions between components)
edges = [
    ("Mobile App (Flutter)", "Backend (Laravel)"),
    ("Web Interface", "Backend (Laravel)"),
    ("Backend (Laravel)", "Database (MySQL)"),
    ("Backend (Laravel)", "AI Recommendation Engine"),
    ("Backend (Laravel)", "Payment Gateway"),
    ("Backend (Laravel)", "Order Management System"),
    ("Order Management System", "Inventory Management System"),
    ("Order Management System", "Shipping & Logistics API"),
    ("Inventory Management System", "Supplier API")
]

# Add edges to the graph
G.add_edges_from(edges)

# Draw the graph
plt.figure(figsize=(15, 8))
pos = nx.spring_layout(G, seed=48)  # Layout for positioning
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=5000, font_size=12, font_weight="bold")
plt.title("Application Communication Diagram")
plt.show()
