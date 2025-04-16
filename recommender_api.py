# Flask Wrapper

from flask import Flask, request, jsonify
from engine import get_recommendations

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    item_id = int(request.args.get("item_id"))
    top_n = int(request.args.get("top_n", 10))

    recommendations = get_recommendations(item_id, top_n)
    return jsonify({"recommended_ids": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
