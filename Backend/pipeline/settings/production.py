from .base import *

DEBUG = False

ALLOWED_HOSTS = ['yourdomain.com', 'localhost', '127.0.0.1']


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backend.postgresql',
        'NAME': 'pipline',
        'USER': 'postgres.vztlaqszhhqpfunwzyhd',
        'PASSWORD': 'Deinha@201927',
        'HOST': 'aws-0-sa-east-1.pooler.supabase.com',
        'PORT': '6543',
    }
}

CORS_ORIGIN_ALLOW_ALL = False

CORS_ALLOWED_ORIGINS = [
    'https://domain.com.br'
]