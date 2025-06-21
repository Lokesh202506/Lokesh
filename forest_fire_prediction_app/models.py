from django.db import models
from django.contrib.auth.models import User

class Dataset(models.Model):
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, default=1)

class Prediction(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    prediction_result = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

class GraphicalReport(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    report_file = models.FileField(upload_to='reports/')
    created_at = models.DateTimeField(auto_now_add=True)