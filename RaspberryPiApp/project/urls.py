from django.contrib import admin
from django.urls import path, include, re_path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from . import views
from mangorest import mango
from . import settings


urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path(r'uploadfile', views.uploadfile, name='uploadfile'),
    path(r'version/', views.version, name='version'),
    path(r'', views.landing_page, name='landing_page'),
    path(r'demo/', views.index, name='index'),
] + settings.DETECTED_URLS + [
#    path('oidc/', include('mozilla_django_oidc.urls')),  # uncomment this line to include SSO
    re_path(r'^.*/$', mango.Common, name='catchall'),
]
urlpatterns = staticfiles_urlpatterns() + urlpatterns


'''
To enable single sign on: 

Step 1: Add following line to urlpatterns:
    path('oidc/', include('mozilla_django_oidc.urls')),

'''