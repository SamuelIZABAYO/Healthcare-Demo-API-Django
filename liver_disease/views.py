import json
import numpy as np

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from static.utils import clean_data, infer_liver_disease_model


@csrf_exempt
def infer(request):
    if request.method == "POST":
        
        if request.POST.get("web_client_data") is not None:
            JSONData = request.POST.get("web_client_data")

            age = float(json.loads(JSONData)["age"])
            gender = float(json.loads(JSONData)["gender"])
            total_billrubin = float(json.loads(JSONData)["total_billrubin"])
            direct_billrubin = float(json.loads(JSONData)["direct_billrubin"])
            alkphos_alkaline_phosphotase = float(json.loads(JSONData)["alkphos_alkaline_phosphotase"])
            sgpt_alamine_aminotransferase = float(json.loads(JSONData)["sgpt_alamine_aminotransferase"])
            sgot_aspartate_aminotransferase = float(json.loads(JSONData)["sgot_aspartate_aminotransferase"])
            total_protiens = float(json.loads(JSONData)["total_protiens"])
            alb_albumin = float(json.loads(JSONData)["alb_albumin"])
            ag_ratio = float(json.loads(JSONData)["ag_ratio"])
            
            new_data = np.array([age, 
                                 gender, 
                                 total_billrubin, 
                                 direct_billrubin,
                                 alkphos_alkaline_phosphotase,
                                 sgpt_alamine_aminotransferase,
                                 sgot_aspartate_aminotransferase,
                                 total_protiens,
                                 alb_albumin,
                                 ag_ratio,
                                 ]).reshape(1, -1)

        else:
            old_data = request.POST.get("data")
            new_data = clean_data(old_data)
        
            del old_data

        label, prob = infer_liver_disease_model(new_data)
        
        return JsonResponse({
            "label": label,
            "probability" : str(prob),
        })
    
    return HttpResponse("Liver Disease Inference Endpoint")
