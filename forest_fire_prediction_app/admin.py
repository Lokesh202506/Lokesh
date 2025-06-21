from django.contrib import admin
from .models import Dataset, Prediction, GraphicalReport

admin.site.register(Dataset)
admin.site.register(Prediction)
admin.site.register(GraphicalReport)