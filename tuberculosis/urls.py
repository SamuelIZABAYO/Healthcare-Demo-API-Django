from django.urls import path
from .views import tuberculosis_prediction_infer


app_name = "tuberculosis"


urlpatterns = [
    path('tuberculosis-pre-infer/', view=tuberculosis_prediction_infer, name="tuberculosis-prediction-infer"),
]
