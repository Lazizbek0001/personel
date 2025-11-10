from django.db import models

# Create your models here.


class Movie(models.Model):
    name = models.CharField(max_length=30)
    img = models.ImageField()
    desc = models.CharField(max_length=50, null=True)
    rating = models.FloatField()


    def __str__(self):
        return self.name
    


class Quote(models.Model):
    desc = models.TextField()
    img = models.ImageField(null=True, blank=True)



class Book(models.Model):
    name = models.CharField(max_length=100)


class Chapter(models.Model):
    name = models.CharField(max_length=100)
    book = models.ForeignKey(Book, related_name="chapters", on_delete=models.CASCADE)
    content = models.TextField()

    

    

    