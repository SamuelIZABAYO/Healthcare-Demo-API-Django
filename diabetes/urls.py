from django.urls import path
from .views import infer

app_name = "diabetes"

urlpatterns = [
    path('diabetes-infer/', infer, name="diabetes-infer"),
]
