from django.urls import path, include
from .views import get_routes

app_name = 'accounts'
urlpatterns = [
    path('api/', get_routes, name='api'),
]