import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import get_recommendations

# Run the test
print(get_recommendations(item_id=12, top_n=10))
