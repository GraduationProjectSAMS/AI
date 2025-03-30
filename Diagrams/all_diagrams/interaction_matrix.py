from common_imports import plt, pd
from constants import INTERACTION_COMPONENTS, INTERACTION_MATRIX

def draw_interaction_matrix(ax=None):
    # Convert the matrix to a DataFrame
    interaction_df = pd.DataFrame(INTERACTION_MATRIX, columns=INTERACTION_COMPONENTS, index=INTERACTION_COMPONENTS)

    # Plot as heatmap
    if ax:
        ax.set_title("Application Interaction Matrix")
        im = ax.imshow(interaction_df, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(len(INTERACTION_COMPONENTS)))
        ax.set_yticks(range(len(INTERACTION_COMPONENTS)))
        ax.set_xticklabels(INTERACTION_COMPONENTS, rotation=90)
        ax.set_yticklabels(INTERACTION_COMPONENTS)
        plt.colorbar(im, ax=ax, label="Interaction (1 = Yes, 0 = No)")
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(interaction_df, cmap="Blues", interpolation="nearest")
        plt.xticks(range(len(INTERACTION_COMPONENTS)), INTERACTION_COMPONENTS, rotation=90)
        plt.yticks(range(len(INTERACTION_COMPONENTS)), INTERACTION_COMPONENTS)
        plt.colorbar(label="Interaction (1 = Yes, 0 = No)")
        plt.title("Application Interaction Matrix")
        plt.show()

if __name__ == "__main__":
    draw_interaction_matrix()
