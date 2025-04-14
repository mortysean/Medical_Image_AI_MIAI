from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from utils.predict import predict_mask_with_meta  # ⬅️ 使用新函数

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "No image file provided"}, status=400)

        print(f"[✓] Received image: {image_file.name}")

        result = predict_mask_with_meta(image_file)  # ⬅️ 返回 dict

        print(f"[✓] Prediction done, returning result with {len(result.get('lesions', []))} lesions")
        return Response(result)
