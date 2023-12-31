"""
URL configuration for comp702 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from .indexview import home
from .indexview import pie_chart
from .indexview import (page1, page2, page3)
from django.urls import re_path
urlpatterns = [
    re_path(r'^$', page1),
    re_path(r'result', pie_chart),
    re_path(r'page2', page2),
    re_path(r"page3", page3),
]
