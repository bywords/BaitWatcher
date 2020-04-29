from django.conf.urls import url
from .views import ArticleView

urlpatterns = [
    url(r'^articles/$', ArticleView.as_view()),
]

# urlpatterns = [
#     url(r'^articles/(?P<link>[a-zA-Z0-9]+)/$', ArticleView.as_view()),
# ]
