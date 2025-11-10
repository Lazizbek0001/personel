from django.urls import path
from . import views


urlpatterns =[
    path("", views.home, name="home"),
    path('api/movies/', views.movie_list, name='movie_list'),
    path('api/quotes/', views.quote_list, name='quote_list'),
    path('api/movies/add/', views.add_movie, name='add_movie'),
    path('api/quotes/add/', views.add_quote, name='add_quote'),
    path("movies/", views.movies, name="movies"),
]