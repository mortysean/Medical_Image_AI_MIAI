from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from utils.predict import predict_mask

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        image_file = request.FILES['image']
        print(f"[✓] Received image: {image_file.name}")  # <-- 打印文件名

        encoded_mask = predict_mask(image_file)

        print(f"[✓] Prediction done, returning result")  # <-- 打印状态
        return Response({'mask': encoded_mask})
