from django import forms
from .models import Camera

class CameraForm(forms.ModelForm):
    class Meta:
        model = Camera
        fields = ['name', 'url', 'stream_type', 'description', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter camera name'
            }),
            'url': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., rtsp://192.168.1.100:554/stream or 0 for local camera'
            }),
            'stream_type': forms.Select(attrs={
                'class': 'form-control'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional camera description'
            }),
            'is_active': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }
        help_texts = {
            'url': 'For local cameras, use numbers (0, 1, 2...). For IP cameras, use full RTSP/HTTP URLs.',
            'stream_type': 'Select the type of camera stream',
        }