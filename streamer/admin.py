from django.contrib import admin
from .models import Camera

@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['name', 'stream_type', 'url', 'is_active', 'created_at']
    list_filter = ['stream_type', 'is_active', 'created_at']
    search_fields = ['name', 'url', 'description']
    list_editable = ['is_active']
    date_hierarchy = 'created_at'