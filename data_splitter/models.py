# data_splitter/models.py
from django.db import models

class DataSplit(models.Model):
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    total_samples = models.IntegerField()
    train_samples = models.IntegerField()
    val_samples = models.IntegerField()
    test_samples = models.IntegerField()
    stratify_column = models.CharField(max_length=255, null=True, blank=True)
    
    def __str__(self):
        return self.name