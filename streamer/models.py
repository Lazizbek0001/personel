from django.db import models

class Camera(models.Model):
    STREAM_TYPES = [
        ('rtsp', 'RTSP Stream'),
        ('http', 'HTTP Stream'),
        ('local', 'Local Camera'),
    ]
    
    name = models.CharField(max_length=100)
    url = models.CharField(
        max_length=255, 
        help_text="RTSP URL (rtsp://...), HTTP URL (http://...), or camera index (0, 1, 2...)"
    )
    stream_type = models.CharField(
        max_length=20,
        choices=STREAM_TYPES,
        default='rtsp'
    )
    description = models.TextField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.get_stream_type_display()})"
    
    def get_stream_url(self):
        """Return proper stream URL based on type"""
        if self.stream_type == 'local':
            return int(self.url) if self.url.isdigit() else 0
        return self.url
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('camera_list')