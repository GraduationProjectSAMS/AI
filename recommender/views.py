from rest_framework.decorators import api_view
from rest_framework.response import Response
from .ai.engine import get_recommendations_for_user

@api_view(["GET"])
def recommend_view(request):
    user_id = request.GET.get("user_id")
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return Response({"error": "Invalid or missing user_id"}, status=400)

    try:
        recommendations = get_recommendations_for_user(user_id)
        return Response(recommendations)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
