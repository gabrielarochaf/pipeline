from django.conf import settings
from django.urls import  re_path
from django.conf.urls.static import static
from api import views

urlpatterns = [
    re_path(r'^images$', views.imagesApi),
    re_path(r'^images/([0-9]+)$',views.imagesApi),
    re_path(r'^images/SaveFile$', views.Save_Files, name='save_files'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)