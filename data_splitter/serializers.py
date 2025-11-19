# data_splitter/serializers.py
from rest_framework import serializers
from .models import DataSplit

class DataSplitSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSplit
        fields = '__all__'