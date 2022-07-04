import json
import numpy as np

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from static.utils import CFG, decode_data, encode_image_to_base64, decode_image


@csrf_exempt
def brain_mri_abnormality_segment_infer(request):
    if request.method == "POST":
        cfg = CFG(mode="brain-abnormality")
        cfg.setup()

        if request.POST.get("data") is not None:
            imageData  = json.loads(request.POST.get("data"))["imageData"]
            _, image = decode_image(imageData)
        else:
            image = decode_data(request.FILES["image"].read())

        image = cfg.infer(image)
        image = np.clip((image*255), 0, 255).astype("uint8")
        image = np.concatenate((np.expand_dims(image, axis=2), np.expand_dims(image, axis=2), np.expand_dims(image, axis=2)), axis=2)

        result_imageData = encode_image_to_base64(image=image)

        return JsonResponse({
            "imageData" : result_imageData,
        }) 

    return HttpResponse("Brain MRI Segmentation Inference Endpoint")
