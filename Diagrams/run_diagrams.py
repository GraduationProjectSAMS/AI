import matplotlib.pyplot as plt
from all_diagrams.communication_diagram import draw_communication_diagram
from all_diagrams.interaction_matrix import draw_interaction_matrix
from all_diagrams.usage_diagram import draw_usage_diagram
from all_diagrams.user_location_diagram import draw_user_location_diagram

# Create a figure with subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Call each function and pass the corresponding subplot axis
draw_communication_diagram(axes[0, 0])  # Top-left
draw_interaction_matrix(axes[0, 1])  # Top-right
draw_usage_diagram(axes[1, 0])  # Bottom-left
draw_user_location_diagram(axes[1, 1])  # Bottom-right

# Adjust layout to prevent overlap
plt.tight_layout()

# Save as a single file
plt.savefig("all_diagrams.png", dpi=300)

# Show the combined figure
plt.show()
