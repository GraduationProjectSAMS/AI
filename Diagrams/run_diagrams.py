import matplotlib.pyplot as plt
from communication_diagram import draw_communication_diagram
from interaction_matrix import draw_interaction_matrix
from usage_diagram import draw_usage_diagram
from user_location_diagram import draw_user_location_diagram

# Disable interactive mode to prevent blocking issues
plt.ioff()

# Call all functions sequentially
draw_communication_diagram()
draw_interaction_matrix()
draw_usage_diagram()
draw_user_location_diagram()

# Show all figures at the end
plt.show()
