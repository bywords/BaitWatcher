from django.db import models

class Article(models.Model):
    link = models.URLField()
    headline = models.CharField(max_length=200)
    body = models.CharField(max_length=5000)
    score = models.FloatField()