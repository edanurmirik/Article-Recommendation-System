from django.urls import path
from . import views

urlpatterns = [
    path("",views.index,name=''),
    path("index",views.index,name='index'),
    path("login_page",views.login_page,name='login_page'),
    path("signup",views.signup,name='signup'),
    path("interests",views.interests,name='interests'),
    path("home",views.home,name='home'),
    path("logout_page",views.logout_page,name='logout_page'),
    path("profil_page/", views.profil_page, name='profil_page'),
    path("update_profil/", views.update_profil, name='update_profil'),
    path("update_interests/", views.update_interests, name='update_interests'),
    path("articles/", views.articles, name='articles'),
    path('toggle_like_dislike/', views.toggle_like_dislike, name='toggle_like_dislike'),
    path('ikinci_asama', views.ikinci_asama, name='ikinci_asama'),
    path('liked_articles/', views.liked_articles, name='liked_articles'),
    path('article/<str:mongo_id>/', views.article_detail, name='article_detail'),
    path('personalised_articles/', views.kisisellestirilmis_oneriler, name='personalised_articles'),
    path('algorithm_articles/', views.algoritma_oneriler, name='algorithm_articles'),
]