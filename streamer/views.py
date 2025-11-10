from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.urls import reverse_lazy
from django.views.generic import CreateView, UpdateView, DeleteView, ListView
from .models import Camera
from .forms import CameraForm
from .rtsp_streamer import DynamicRTSPStreamer
import json

# Global streamer instance
streamer = DynamicRTSPStreamer()

class CameraListView(ListView):
    model = Camera
    template_name = 'stream/camera_list.html'
    context_object_name = 'cameras'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['current_stream'] = streamer.get_status()
        return context

class CameraCreateView(CreateView):
    model = Camera
    form_class = CameraForm
    template_name = 'stream/camera_form.html'
    success_url = reverse_lazy('camera_list')
    
    def form_valid(self, form):
        messages.success(self.request, f'Camera "{form.instance.name}" added successfully!')
        return super().form_valid(form)

class CameraUpdateView(UpdateView):
    model = Camera
    form_class = CameraForm
    template_name = 'stream/camera_form.html'
    success_url = reverse_lazy('camera_list')
    
    def form_valid(self, form):
        messages.success(self.request, f'Camera "{form.instance.name}" updated successfully!')
        return super().form_valid(form)

class CameraDeleteView(DeleteView):
    model = Camera
    template_name = 'stream/camera_confirm_delete.html'
    success_url = reverse_lazy('camera_list')
    
    def delete(self, request, *args, **kwargs):
        camera = self.get_object()
        # Stop stream if this camera is currently streaming
        if streamer.current_camera and streamer.current_camera.id == camera.id:
            streamer.stop_stream()
        messages.success(request, f'Camera "{camera.name}" deleted successfully!')
        return super().delete(request, *args, **kwargs)

# API views for streaming
@require_http_methods(["POST"])
@csrf_exempt
def start_stream(request, camera_id):
    """Start streaming from specific camera"""
    camera = get_object_or_404(Camera, id=camera_id, is_active=True)
    
    try:
        success = streamer.start_stream(camera)
        
        if success:
            stream_url = streamer.get_current_stream_url()
            return JsonResponse({
                'status': 'success',
                'message': f'Started streaming from {camera.name}',
                'stream_url': stream_url,
                'camera_name': camera.name,
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Failed to start stream'
            }, status=500)
            
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@require_http_methods(["POST"])
@csrf_exempt
def stop_stream(request):
    """Stop current stream"""
    try:
        streamer.stop_stream()
        return JsonResponse({
            'status': 'success',
            'message': 'Stream stopped'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error', 
            'message': str(e)
        }, status=500)

@require_http_methods(["GET"])
def stream_status(request):
    """Get current stream status"""
    return JsonResponse(streamer.get_status())