from decouple import config

environment = config('DJANGO_SETTINGS_MODULE', default='backend.settings.devolpment')

if environment == 'backend.settings.production':
    from .production import *
else:
    from .development import *
