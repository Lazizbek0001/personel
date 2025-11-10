from django.shortcuts import render
from django.http import JsonResponse
from .models import Movie, Quote
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import random

def movie_list(request):
    movies = list(Movie.objects.values('id', 'name', 'img', 'desc', 'rating'))
    movies = random.sample(movies, min(len(movies), 3))
    # Add full URL for images
    for movie in movies:
        if movie['img']:
            movie['img_url'] = request.build_absolute_uri(settings.MEDIA_URL + str(movie['img']))
        else:
            movie['img_url'] = None
    return JsonResponse(movies, safe=False)

def movies(request):
    movies = Movie.objects.all().order_by("name")
    return render(request, "movies.html", {"movies":movies})

def quote_list(request):
    quotes = list(Quote.objects.values('id', 'desc', 'img'))
    quotes = random.sample(quotes, min(len(quotes), 3))
    for quote in quotes:
        if quote['img']:
            quote['img_url'] = request.build_absolute_uri(settings.MEDIA_URL + str(quote['img']))
        else:
            quote['img_url'] = None
    return JsonResponse(quotes, safe=False)

@csrf_exempt
def add_movie(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        desc = request.POST.get('desc')
        rating = request.POST.get('rating')
        img = request.FILES.get('img')
        movie = Movie.objects.create(name=name, desc=desc, rating=rating, img=img)
        return JsonResponse({'message': 'Movie added!', 'id': movie.id})
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def add_quote(request):
    if request.method == 'POST':
        desc = request.POST.get('desc')
        img = request.FILES.get('img')
        quote = Quote.objects.create(desc=desc, img=img)
        return JsonResponse({'message': 'Quote added!', 'id': quote.id})
    return JsonResponse({'error': 'Invalid request'}, status=400)

def home(request):
    return render(request, "index.html")