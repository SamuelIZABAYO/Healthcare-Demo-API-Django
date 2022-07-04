from django.urls import path
from .views import pneumonia_prediction_infer


app_name = "pneumonia"


urlpatterns = [
    path('pneumonia-pre-infer/', view=pneumonia_prediction_infer, name="pneumonia-prediction-infer"),
]
