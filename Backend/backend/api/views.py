from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse

from api.models import Images
from api.serializers import ImagesSerializer
from django.core.files.storage import default_storage
# Create your views here.

@csrf_exempt
def imagesApi(request, id=0):
    if request.method =='GET':
        images = Images.objects.all()
        images_serializer = ImagesSerializer(images, many=True)
        return JsonResponse(images_serializer.data, safe=False)
    elif request.method == 'POST':
        image_data = JSONParser().parse(request)
        image_serializer = ImagesSerializer(data=image_data)
        if image_serializer.is_valid():
            image_serializer.save()
            return JsonResponse("Add Successfully", safe=False)
        return JsonResponse('Error to post', safe=False)


@csrf_exempt
def Save_Files(request):
    file = request.FILES['MyFile']
    file_name = default_storage.save(file.name, file)

    return JsonResponse(file_name, safe=False)


