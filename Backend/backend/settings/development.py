# myproject/settings/development.py
from .base import *

DEBUG = True

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# CORS settings if needed

# Django Debug Toolbar settings if used

# Next.js frontend configuration
NEXT_FRONTEND_URL = 'http://localhost:3000'  # Adjust to match your Next.js development server
