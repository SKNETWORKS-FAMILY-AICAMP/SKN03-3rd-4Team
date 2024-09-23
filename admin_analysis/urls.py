from django.urls import path
from . import views


urlpatterns = [
    path('', views.admin_analysis, name='admin_analysis')

]

