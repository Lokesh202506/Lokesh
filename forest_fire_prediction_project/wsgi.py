"""
WSGI config for forest_fire_prediction_project.

It exposes the WSGI callable as a module-level variable named `application`.

For more information on this file, see
https://docs.djangoproject.com/en/stable/howto/deployment/wsgi/
"""
import os

# Create a flag file on every server restart
with open('server_restart.flag', 'w') as f:
    f.write('restarted')

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forest_fire_prediction_project.settings')

application = get_wsgi_application()


import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forest_fire_prediction_project.settings')

application = get_wsgi_application()