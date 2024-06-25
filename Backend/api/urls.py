# api/urls.py

from django.urls import path
from .views import ImageUploadView, image_list_view

urlpatterns = [
    path('upload/', ImageUploadView.as_view(), name='image_upload'),  # Class-based view
    path('images/', image_list_view, name='image_list'),  # Function-based view
]
