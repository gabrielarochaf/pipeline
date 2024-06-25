from django.shortcuts import render
from django.http import JsonResponse
from .models import ImageUpload
from django.conf import settings
from supabase import create_client, Client
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import uuid
import os

def get_supabase_client() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

@method_decorator(csrf_exempt, name='dispatch')
class ImageUploadView(View):
    def post(self, request, *args, **kwargs):
   
        file = request.FILES.get('image')

       
        # Save image locally first (ensure proper handling)
        image_instance = ImageUpload(title=file.name, image=file)
        image_instance.save()

        try:
            # Read file content in bytes
            file.seek(0)  # Ensure we're reading from the start of the file
            file_content = file.read()
            file_extension = file.name.split('.')[-1]

            # Check if filename starts with a specific prefix
            filename = os.path.splitext(file.name)[0]  # Remove extension
            # prefixes = ['cachorros', 'gatos', 'pássaros']  # Defina os prefixos desejados

            folder = filename  # Pasta padrão se não corresponder a nenhum prefixo
            # for prefix in prefixes:
            #     if filename.lower().startswith(prefix):
            #         folder = prefix.capitalize()  # Cria a pasta com a primeira letra em maiúsculo
            #         break

            # Use original filename with a unique identifier
            unique_file_name = f"{folder}/{filename}.{file_extension}"

            # Upload to Supabase
            supabase: Client = get_supabase_client()
            response = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(unique_file_name, file_content)

            if response.status_code == 200:
                # Construct the public URL manually
                public_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/{settings.SUPABASE_BUCKET_NAME}/{unique_file_name}"

                if public_url:
                    image_instance.image_url = public_url
                    image_instance.save()
                    return JsonResponse({'image_url': public_url})
                else:
                    return JsonResponse({'error': 'Failed to get public URL from Supabase.'}, status=500)
            else:
                return JsonResponse({'error': 'Failed to upload to Supabase.'}, status=500)
        
        except Exception as e:
            # Handle any exceptions that occur during file handling or upload
            return JsonResponse({'error': f'Error uploading file to Supabase: {str(e)}'}, status=500)

def image_list_view(request):
    images = ImageUpload.objects.all()
    images_data = [{'title': image.title, 'image_url': image.image_url} for image in images]
    return JsonResponse(images_data, safe=False)
