from django.db import models


def upload_to(instance, filename):
    return 'images/{filename}'.format(filename=filename)

class ImageUpload(models.Model):
    title = models.CharField(max_length=100)
    image = models.ImageField(upload_to=upload_to)
    image_url = models.URLField(max_length=200, blank=True)