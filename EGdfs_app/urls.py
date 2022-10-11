from django.urls import path, include
from django.contrib.auth.views import LogoutView


from . import views

urlpatterns = [
    path('', views.homePage, name = 'home'),
    path('accounts/', include('allauth.urls')),
    path('logout', LogoutView.as_view()),
]


