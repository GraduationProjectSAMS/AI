# recommender/urls.py
from django.urls import path
from .views import recommend_view

urlpatterns = [
    path('recommendations/', recommend_view),
]
