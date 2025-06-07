import time
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import get_recommendations

start = time.time()

print(get_recommendations(item_id=12, top_n=10))

print(f"Done in {time.time() - start:.2f} seconds")
