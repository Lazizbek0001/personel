from django.contrib import admin

# Register your models here.
from .models import Movie, Quote


admin.site.register(Movie)

admin.site.register(Quote)

