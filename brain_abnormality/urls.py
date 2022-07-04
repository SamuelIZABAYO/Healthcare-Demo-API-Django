from django.urls import path
from .views import brain_mri_abnormality_segment_infer


app_name = "brain_abnormality"


urlpatterns = [
    path('brain-mri-abnormality-infer/', view=brain_mri_abnormality_segment_infer, name="brain-mri-abnormality-segment-infer"),
]
