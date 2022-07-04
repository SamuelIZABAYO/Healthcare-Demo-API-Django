from django.urls import path
from .views import infer

app_name = "liver_disease"

urlpatterns = [
    path('liver-disease-infer/', infer, name="liver-disease-infer"),
]
