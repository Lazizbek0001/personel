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

# Python Standard Library Imports
import threading
import time
import logging
import subprocess
import os

# Computer Vision Imports
import cv2
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)

# Import your custom streamer
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
        self.retry_count = {}
    
    def get_camera_stream(self, camera_id):
        """Get or create camera stream with retry logic"""
        if camera_id not in self.cameras:
            self.retry_count[camera_id] = 0
            max_retries = 5
            retry_delay = 2
            
            while self.retry_count[camera_id] < max_retries:
                try:
                    camera = Camera.objects.get(id=camera_id)
                    cap = cv2.VideoCapture(camera.get_stream_url())
                    
                    # Set camera properties with longer timeout
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Wait for camera to be ready
                    time.sleep(1)
                    
                    # Test if camera is actually readable
                    if cap.isOpened():
                        # Try to read a frame with timeout
                        for i in range(10):
                            ret, frame = cap.read()
                            if ret:
                                self.cameras[camera_id] = cap
                                self.locks[camera_id] = threading.Lock()
                                logger.info(f"Camera {camera_id} connected successfully")
                                return self.cameras[camera_id]
                            time.sleep(0.1)
                    
                    # If we get here, camera didn't work
                    cap.release()
                    
                except Exception as e:
                    logger.warning(f"Camera {camera_id} attempt {self.retry_count[camera_id] + 1} failed: {e}")
                
                self.retry_count[camera_id] += 1
                if self.retry_count[camera_id] < max_retries:
                    logger.info(f"Retrying camera {camera_id} in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            logger.error(f"Camera {camera_id} failed after {max_retries} attempts")
            return None
        
        return self.cameras[camera_id]
    
    def generate_frames(self, camera_id):
        """Generate MJPEG frames with automatic recovery"""
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while True:
            camera_stream = self.get_camera_stream(camera_id)
            if camera_stream is None:
                yield from self.generate_blank_frame("Camera Connecting...")
                time.sleep(2)
                continue
            
            try:
                with self.locks[camera_id]:
                    success, frame = camera_stream.read()
                
                if success:
                    consecutive_failures = 0
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                    else:
                        yield from self.generate_blank_frame("Encoding Error")
                        time.sleep(0.1)
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Camera {camera_id} has consecutive failures, restarting...")
                        self.restart_camera_stream(camera_id)
                        consecutive_failures = 0
                    yield from self.generate_blank_frame("No Signal")
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error reading from camera {camera_id}: {e}")
                yield from self.generate_blank_frame("Stream Error")
                time.sleep(1)
    
    def generate_blank_frame(self, message="No Signal"):
        """Generate a blank frame with message"""
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, message, (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        except Exception as e:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n')
    
    def restart_camera_stream(self, camera_id):
        """Restart camera connection"""
        if camera_id in self.cameras:
            try:
                self.cameras[camera_id].release()
            except:
                pass
            if camera_id in self.cameras:
                del self.cameras[camera_id]
        return self.get_camera_stream(camera_id)

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
        return StreamingHttpResponse(
            mjpeg_streamer.generate_blank_frame("Stream Error"),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )