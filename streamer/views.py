# Django Core Imports
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators import gzip
from django.contrib import messages
from django.urls import reverse_lazy
from django.views.generic import CreateView, UpdateView, DeleteView, ListView
from django.conf import settings

# Model and Form Imports
from .models import Camera
from .forms import CameraForm

# Streaming Imports
import cv2
import threading
import time
import logging
import subprocess
import os
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)

# Import your custom streamers
from .rtsp_streamer import DynamicRTSPStreamer

# Global streamer instance
streamer = DynamicRTSPStreamer()

# View Classes
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
    template_name = 'stream/camera_delete.html'
    success_url = reverse_lazy('camera_list')
    
    def delete(self, request, *args, **kwargs):
        camera = self.get_object()
        # Stop stream if this camera is currently streaming
        if streamer.current_camera and streamer.current_camera.id == camera.id:
            streamer.stop_stream()
        messages.success(request, f'Camera "{camera.name}" deleted successfully!')
        return super().delete(request, *args, **kwargs)

# MJPEG Streamer Class
class MJPEGStreamer:
    def __init__(self):
        self.cameras = {}
        self.locks = {}
    
    def get_camera_stream(self, camera_id):
        """Get or create camera stream"""
        if camera_id not in self.cameras:
            try:
                camera = Camera.objects.get(id=camera_id)
                cap = cv2.VideoCapture(camera.get_stream_url())
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                
                self.cameras[camera_id] = cap
                self.locks[camera_id] = threading.Lock()
            except Exception as e:
                logger.error(f"Failed to open camera {camera_id}: {e}")
                return None
        return self.cameras[camera_id]
    
    def generate_frames(self, camera_id):
        """Generate MJPEG frames"""
        camera_stream = self.get_camera_stream(camera_id)
        if camera_stream is None:
            # Yield blank frames if camera is unavailable
            while True:
                yield from self.generate_blank_frame("Camera Unavailable")
                time.sleep(0.1)
        
        while True:
            with self.locks.get(camera_id, threading.Lock()):
                success, frame = camera_stream.read()
            
            if not success:
                # If frame reading fails, yield blank frame and try again
                yield from self.generate_blank_frame("No Signal")
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG
            try:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                else:
                    yield from self.generate_blank_frame("Encoding Error")
            except Exception as e:
                logger.error(f"Frame encoding error: {e}")
                yield from self.generate_blank_frame("Stream Error")
                time.sleep(0.1)
    
    def generate_blank_frame(self, message="No Signal"):
        """Generate a blank frame with message"""
        try:
            # Create a blank black frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add text
            cv2.putText(frame, message, (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        except Exception as e:
            # Fallback: yield a simple frame boundary
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n')

# Global MJPEG streamer instance
mjpeg_streamer = MJPEGStreamer()

# API Views
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
        logger.error(f"Error starting stream: {e}")
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
        logger.error(f"Error stopping stream: {e}")
        return JsonResponse({
            'status': 'error', 
            'message': str(e)
        }, status=500)

@require_http_methods(["GET"])
def stream_status(request):
    """Get current stream status"""
    status = streamer.get_status()
    # Add camera ID to status for MJPEG streaming
    if status['is_streaming'] and streamer.current_camera:
        status['current_camera_id'] = streamer.current_camera.id
    return JsonResponse(status)

@gzip.gzip_page
@require_http_methods(["GET"])
def mjpeg_stream(request, camera_id):
    """MJPEG stream endpoint"""
    try:
        response = StreamingHttpResponse(
            mjpeg_streamer.generate_frames(camera_id),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
        response['Cache-Control'] = 'no-cache'
        response['Pragma'] = 'no-cache'
        return response
    except Exception as e:
        logger.error(f"MJPEG stream error: {e}")
        # Return blank stream on error
        return StreamingHttpResponse(
            mjpeg_streamer.generate_blank_frame("Stream Error"),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )

def generate_blank_frame():
    """Generate a blank frame when camera is unavailable"""
    # Create a blank black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add error text
    cv2.putText(frame, "Camera Unavailable", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, jpeg = cv2.imencode('.jpg', frame)
    if ret:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')