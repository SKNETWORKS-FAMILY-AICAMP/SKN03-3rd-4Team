from django.urls import path
from . import views


urlpatterns = [
    path('', views.prediction, name='prediction'),
    path('predict_churn', views.predict_churn, name='predict_churn'),

]
