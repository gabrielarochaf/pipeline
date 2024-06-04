from django.db import models

# Create your models here.

class Images(models.Model):
    ImageId = models.AutoField(primary_key=True)
    ImageName = models.CharField(max_length=100)
    ImageFile = models.CharField(max_length=100)