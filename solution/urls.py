from django.urls import path
from . import views

urlpatterns = [
    # path('', views.solution_test, name='solution'),
    path('', views.prediction_solution_view, name='solution'),
]
