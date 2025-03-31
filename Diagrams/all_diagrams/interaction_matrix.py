from common_imports import plt, pd, sns
from constants import INTERACTION_COMPONENTS, INTERACTION_MATRIX

def draw_interaction_matrix(ax):
    interaction_df = pd.DataFrame(INTERACTION_MATRIX, columns=INTERACTION_COMPONENTS, index=INTERACTION_COMPONENTS)

    sns.heatmap(interaction_df, annot=True, cmap="Blues", fmt="d", cbar=True, linewidths=0.5, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title("Application Interaction Matrix")