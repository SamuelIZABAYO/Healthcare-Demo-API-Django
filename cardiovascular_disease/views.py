import json
import numpy as np

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from static.utils import clean_data, infer_cardiovascular_disease_model


@csrf_exempt
def infer(request):
    if request.method == "POST":
        
        if request.POST.get("web_client_data") is not None:
            JSONData = request.POST.get("web_client_data")

            age = float(json.loads(JSONData)["age"])
            gender = float(json.loads(JSONData)["gender"])
            height = float(json.loads(JSONData)["height"])
            weight = float(json.loads(JSONData)["weight"])
            ap_high = float(json.loads(JSONData)["ap_high"])
            ap_low = float(json.loads(JSONData)["ap_low"])
            cholestrol = float(json.loads(JSONData)["cholestrol"])
            glucose = float(json.loads(JSONData)["glucose"])
            smoke = float(json.loads(JSONData)["smoke"])
            alcohol = float(json.loads(JSONData)["alcohol"])
            active = float(json.loads(JSONData)["active"])

            new_data = np.array([age, 
                                 gender, 
                                 height, 
                                 weight,
                                 ap_high,
                                 ap_low,
                                 cholestrol,
                                 glucose,
                                 smoke,
                                 alcohol,
                                 active,
                                 ]).reshape(1, -1)

        else:
            old_data = request.POST.get("data")
            new_data = clean_data(old_data)
        
            del old_data

        label, prob = infer_cardiovascular_disease_model(new_data)
        
        return JsonResponse({
            "label": label,
            "probability" : str(prob),
        })
    
    return HttpResponse("Cardiovascular Disease Inference Endpoint")
