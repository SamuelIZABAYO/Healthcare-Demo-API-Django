import json
import numpy as np

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from static.utils import clean_data, infer_diabetes_model


@csrf_exempt
def infer(request):
    if request.method == "POST":
        
        if request.POST.get("web_client_data") is not None:
            JSONData = request.POST.get("web_client_data")

            pregnancies = float(json.loads(JSONData)["pregnancies"])
            plasma_glucose = float(json.loads(JSONData)["plasma_glucose"])
            diastolic_blood_pressure = float(json.loads(JSONData)["diastolic_blood_pressure"])
            triceps_thickness = float(json.loads(JSONData)["triceps_thickness"])
            serum_insulin = float(json.loads(JSONData)["serum_insulin"])
            bmi = float(json.loads(JSONData)["bmi"])
            diabetes_pedigree = float(json.loads(JSONData)["diabetes_pedigree"])
            age = float(json.loads(JSONData)["age"])
            
            new_data = np.array([pregnancies, 
                                 plasma_glucose, 
                                 diastolic_blood_pressure, 
                                 triceps_thickness, 
                                 serum_insulin, 
                                 bmi, 
                                 diabetes_pedigree, 
                                 age]).reshape(1, -1)

        else:
            old_data = request.POST.get("data")
            new_data = clean_data(old_data)
        
            del old_data

        label, prob = infer_diabetes_model(new_data)
        
        return JsonResponse({
            "label": label,
            "probability" : str(prob),
        })
    
    return HttpResponse("Diabetes Inference Endpoint")
