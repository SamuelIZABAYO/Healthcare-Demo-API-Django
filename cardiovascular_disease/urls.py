from django.urls import path
from .views import infer

app_name = "cardiovascular_disease"

urlpatterns = [
    path('cardiovascular-disease-infer/', infer, name="cardiovascular-disease-infer"),
]
