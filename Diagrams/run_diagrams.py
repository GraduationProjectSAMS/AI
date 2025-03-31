import matplotlib.pyplot as plt
from all_diagrams.communication_diagram import draw_communication_diagram
from all_diagrams.interaction_matrix import draw_interaction_matrix
from all_diagrams.usage_diagram import draw_usage_diagram
from all_diagrams.user_location_diagram import draw_user_location_diagram

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Assign each function to a subplot
draw_communication_diagram(axes[0, 0])  # Top-left
draw_interaction_matrix(axes[0, 1])     # Top-right
draw_usage_diagram(axes[1, 0])          # Bottom-left
draw_user_location_diagram(axes[1, 1])  # Bottom-right

plt.tight_layout()
plt.show()
