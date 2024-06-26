# # gallery/views.py
# from django.shortcuts import render
# from django.http import JsonResponse
# from .models import ImageUpload
# from django.conf import settings
# from supabase import create_client, Client
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from django.views import View
# import os
# import zipfile
# import logging

# logger = logging.getLogger(__name__)

# def get_supabase_client() -> Client:
#     logger.debug("Creating Supabase client.")
#     return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# @method_decorator(csrf_exempt, name='dispatch')
# class ImageUploadView(View):
#     def post(self, request, *args, **kwargs):
#         file = request.FILES.get('image')

#         if not file:
#             logger.error("No file provided in the request.")
#             return JsonResponse({'error': 'Image file is required.'}, status=400)

#         if not file.name.endswith('.zip'):
#             logger.error("File provided is not a ZIP file.")
#             return JsonResponse({'error': 'Only ZIP files are allowed.'}, status=400)

#         title = os.path.splitext(file.name)[0]
#         file_urls = []

#         try:
#             with zipfile.ZipFile(file, 'r') as zip_ref:
#                 for zip_info in zip_ref.infolist():
#                     if not zip_info.is_dir():
#                         with zip_ref.open(zip_info) as file_in_zip:
#                             file_content = file_in_zip.read()

#                             if not file_content:
#                                 logger.error(f"Failed to read file content for {zip_info.filename}")
#                                 continue

#                             supabase: Client = get_supabase_client()

#                             try:
#                                 logger.debug(f"Uploading file {zip_info.filename} to Supabase.")
#                                 response = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(f"{title}/{zip_info.filename}", file_content)

#                                 if hasattr(response, 'status_code') and response.status_code == 200:
#                                     public_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/{settings.SUPABASE_BUCKET_NAME}/{title}/{zip_info.filename}"
#                                     file_urls.append(public_url)
#                                     logger.debug(f"Uploaded file {zip_info.filename} to Supabase successfully.")
#                                 else:
#                                     error_status = response.status_code if hasattr(response, 'status_code') else 'Unknown'
#                                     logger.error(f"Failed to upload file {zip_info.filename} to Supabase. Status code: {error_status}")
#                                     return JsonResponse({'error': f'Failed to upload file {zip_info.filename} to Supabase. Status code: {error_status}'}, status=500)
#                             except Exception as upload_error:
#                                 logger.error(f"Exception during file upload to Supabase: {str(upload_error)}", exc_info=True)
#                                 return JsonResponse({'error': f'Exception during file upload to Supabase: {str(upload_error)}'}, status=500)

#             return JsonResponse({'file_urls': file_urls}, status=200)
        
#         except zipfile.BadZipFile:
#             logger.error("Bad ZIP file provided.")
#             return JsonResponse({'error': 'Provided file is not a valid ZIP file.'}, status=400)
        
#         except Exception as e:
#             logger.error(f"Error processing ZIP file: {str(e)}", exc_info=True)
#             return JsonResponse({'error': f'Error processing ZIP file: {str(e)}'}, status=500)

# def image_list_view(request):
#     images = ImageUpload.objects.all()
#     images_data = [{'title': image.title, 'image_url': image.image_url} for image in images]
#     return JsonResponse(images_data, safe=False)


# gallery/views.py
import os
import zipfile
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from .models import ImageUpload

logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class ImageUploadView(View):
    def post(self, request, *args, **kwargs):
        file = request.FILES.get('image')

        if not file:
            logger.error("No file provided in the request.")
            return JsonResponse({'error': 'Image file is required.'}, status=400)

        if not file.name.endswith('.zip'):
            logger.error("File provided is not a ZIP file.")
            return JsonResponse({'error': 'Only ZIP files are allowed.'}, status=400)

        title = os.path.splitext(file.name)[0]
        file_urls = []
        media_path = os.path.join(settings.MEDIA_ROOT, title)  # Path to save files in 'media' folder

        try:
            with zipfile.ZipFile(file, 'r') as zip_ref:
                for zip_info in zip_ref.infolist():
                    if not zip_info.is_dir():
                        file_path_in_zip = zip_info.filename
                        dest_path = os.path.join(media_path, os.path.dirname(file_path_in_zip))
                        dest_file_path = os.path.join(media_path, file_path_in_zip)

                        os.makedirs(dest_path, exist_ok=True)  # Create directories if not exist

                        with zip_ref.open(zip_info) as file_in_zip:
                            with open(dest_file_path, 'wb') as f:
                                f.write(file_in_zip.read())

                        public_url = os.path.join(settings.MEDIA_URL, title, file_path_in_zip)
                        file_urls.append(public_url)

            return JsonResponse({'file_urls': file_urls}, status=200)
        
        except zipfile.BadZipFile:
            logger.error("Bad ZIP file provided.")
            return JsonResponse({'error': 'Provided file is not a valid ZIP file.'}, status=400)
        
        except Exception as e:
            logger.error(f"Error processing ZIP file: {str(e)}", exc_info=True)
            return JsonResponse({'error': f'Error processing ZIP file: {str(e)}'}, status=500)

def image_list_view(request):
    images = ImageUpload.objects.all()
    images_data = [{'title': image.title, 'image_url': image.image_url} for image in images]
    return JsonResponse(images_data, safe=False)