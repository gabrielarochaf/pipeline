from decouple import config

environment = config('DJANGO_SETTINGS_MODULE', default='pipeline.settings.devolpment')

if environment == 'pipeline.settings.production':
    from .production import *
else:
    from .development import *
