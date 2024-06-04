from rest_framework import serializers
from api.models import Images

class ImagesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Images
        fields = ('ImageId', 'ImageName','ImageFile')