from decouple import config


DEBUG = config('DEBUG', default=False, cast=bool)
SECRET_KEY = config('SECRET_KEY')

environment = config('DJANGO_SETTINGS_MODULE', default='backend.settings.devolpment')

if environment == 'backend.settings.production':
    from .production import *
else:
    from .development import *
