import json

from django.shortcuts import render
from rest_framework import generics
from rest_framework.response import Response
from .serializers import ArticleSerializer
from apps.score.models import Article
import requests
import newspaper

from apps.score.utils.api_utils import create_dataset
import apps.score.utils.argumentparser as parser
from .eval_score import main
from time import sleep
import random

class ArticleView(generics.CreateAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def create(self, request):
        data = request.data
        url = data['url']
        obj = None
        """if len(News.objects.filter(link=url)) != 0:
            obj = News.objects.filter(link=url)[0]
            if obj.score < 0:
                pass
            else: 
                show_process(obj)
        else:"""
        
        print("Downloading Article from URL : %s" % url)
        a = newspaper.Article(url, language='ko')
        a.download()
        a.html = a.html.replace("<br>", "[EOP]")
        a.parse()
        
        body = a.text
        body = body.replace("[EOP]", "\n")
        body = body.replace("\n\n", "\n")
        headline = a.title
        
        #create_dataset(title, text)
        args = parser.ArgumentParser()
        score = main(args, headline, body)
        #score = temp(url, title)
        score = float("{0:.4f}".format(float(score)))
        obj = Article.objects.create(link=url, score=score, body=body, headline=headline)

        serializer = self.serializer_class(obj)
        return Response(serializer.data)


def temp(url, title):
    print("URL : {}, TITLE : {}".format(url, title))
    print("Processing...")
    sleep(0.8)
    score = random.uniform(0, 0.24)
    print("Evaluated Score : {0:.4f}".format(score))
    return score

def show_process(obj):
    print("URL : {}, TITLE : {}".format(obj.link, obj.title))
    print("Processing...")
    sleep(0.5)
    score = obj.score
    print("Evaluated Score : {0:.4f}".format(score))
    return score
