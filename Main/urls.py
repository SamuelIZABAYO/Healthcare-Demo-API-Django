from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('brain_abnormality.urls')),
    path('', include('diabetes.urls')),
    path('', include('pneumonia.urls')),
    path('', include('tuberculosis.urls')),
    path('', include('liver_disease.urls')),
    path('', include('cardiovascular_disease.urls')),
]
