from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from accounts.views import (
    organization_apiview,
    organization_list_apiview,
    user_list_apiview,
    user_apiview,
    image_apiview
)

app_name = 'accounts'
urlpatterns = [
    path('organization', organization_list_apiview.as_view()),
    path('organization/<str:name>', organization_apiview.as_view()),
    path('user', user_list_apiview.as_view()),
    path('user/<str:phone>', user_apiview.as_view()),
    path('img', image_apiview.as_view()),
]