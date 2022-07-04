import json

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from static.utils import CFG, decode_data, decode_image


@csrf_exempt
def tuberculosis_prediction_infer(request):
    if request.method == "POST":
        cfg = CFG(mode="tuberculosis")
        cfg.setup()

        if request.POST.get("data") is not None:
            imageData  = json.loads(request.POST.get("data"))["imageData"]
            _, image = decode_image(imageData)
        else:
            image = decode_data(request.FILES["image"].read())
            
        probability = cfg.infer(image)

        return JsonResponse({
            "probability": str(probability)
        })

    return HttpResponse("Tuberculosis Prediction Inference Endpoint")
