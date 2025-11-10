# Django Core Imports
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators import gzip
from django.contrib import messages
from django.urls import reverse_lazy
from django.views.generic import CreateView, UpdateView, DeleteView, ListView
from django.conf import settings
import os

# Model and Form Imports
from .models import Camera
from .forms import CameraForm

# Python Standard Library Imports
import threading
import time
import logging
import subprocess
import uuid

# Computer Vision Imports
import cv2

# Initialize logger
logger = logging.getLogger(__name__)

# Import your custom RTSP restreamer
from .rtsp_streamer import DynamicRTSPStreamer

# ------------------------ Global RTSP streamer ------------------------
streamer = DynamicRTSPStreamer()

# ======================== UI Views ========================
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
        # Stop RTSP restream if this camera is currently streaming
        if streamer.current_camera and streamer.current_camera.id == camera.id:
            streamer.stop_stream()
        # Stop HLS stream if running
        stop_hls_stream(camera.id)
        messages.success(request, f'Camera "{camera.name}" deleted successfully!')
        return super().delete(request, *args, **kwargs)


# ======================== HLS Stream Manager ========================
class HLSStreamManager:
    """
    Manages HLS streams using FFmpeg to convert RTSP to HLS
    """
    def __init__(self):
        self.streams = {}  # camera_id -> ffmpeg_process
        self.media_base = getattr(settings, 'MEDIA_ROOT', 'media')
        self.hls_base_url = getattr(settings, 'HLS_BASE_URL', '/media/hls/')
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(self.media_base, 'hls'), exist_ok=True)
    
    def start_stream(self, camera_id, rtsp_url):
        """Start HLS stream from RTSP source"""
        if camera_id in self.streams:
            self.stop_stream(camera_id)
        
        # Create unique stream directory
        stream_dir = os.path.join(self.media_base, 'hls', f'camera_{camera_id}')
        os.makedirs(stream_dir, exist_ok=True)
        
        # HLS playlist path
        playlist_path = os.path.join(stream_dir, 'stream.m3u8')
        
        # FFmpeg command for RTSP → HLS conversion
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', rtsp_url,  # Input RTSP stream
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-hls_time', '2',           # Segment length in seconds
            '-hls_list_size', '5',      # Number of segments in playlist
            '-hls_flags', 'delete_segments',  # Delete old segments
            '-hls_segment_filename', os.path.join(stream_dir, 'segment_%03d.ts'),
            '-f', 'hls',
            playlist_path
        ]
        
        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            self.streams[camera_id] = {
                'process': process,
                'playlist_url': f'{self.hls_base_url}camera_{camera_id}/stream.m3u8',
                'start_time': time.time()
            }
            
            logger.info(f"Started HLS stream for camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HLS stream for camera {camera_id}: {e}")
            return False
    
    def stop_stream(self, camera_id):
        """Stop HLS stream"""
        if camera_id in self.streams:
            try:
                process_info = self.streams[camera_id]
                process = process_info['process']
                
                # Terminate FFmpeg process
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                
                # Clean up stream directory
                stream_dir = os.path.join(self.media_base, 'hls', f'camera_{camera_id}')
                if os.path.exists(stream_dir):
                    import shutil
                    shutil.rmtree(stream_dir)
                
                del self.streams[camera_id]
                logger.info(f"Stopped HLS stream for camera {camera_id}")
                
            except Exception as e:
                logger.error(f"Error stopping HLS stream for camera {camera_id}: {e}")
    
    def get_stream_url(self, camera_id):
        """Get HLS stream URL"""
        if camera_id in self.streams:
            return self.streams[camera_id]['playlist_url']
        return None
    
    def is_streaming(self, camera_id):
        """Check if HLS stream is running"""
        if camera_id not in self.streams:
            return False
        
        process_info = self.streams[camera_id]
        process = process_info['process']
        
        # Check if process is still running
        return process.poll() is None

# Global HLS manager
hls_manager = HLSStreamManager()

def stop_hls_stream(camera_id):
    """Stop HLS stream for camera"""
    hls_manager.stop_stream(camera_id)

# ======================== RTSP + HLS API ========================
@require_http_methods(["POST"])
@csrf_exempt
def start_stream(request, camera_id):
    """Start RTSP restream AND HLS stream from specific camera"""
    camera = get_object_or_404(Camera, id=camera_id, is_active=True)
    
    try:
        # Start RTSP stream first
        rtsp_success = streamer.start_stream(camera)
        
        if rtsp_success:
            # Get RTSP URL
            rtsp_url = streamer.get_current_stream_url()
            
            # Wait for RTSP to be ready
            time.sleep(3)
            
            # Start HLS stream from RTSP
            hls_success = hls_manager.start_stream(camera_id, rtsp_url)
            
            if hls_success:
                hls_url = hls_manager.get_stream_url(camera_id)
                return JsonResponse({
                    'status': 'success',
                    'message': f'Started streaming from {camera.name}',
                    'stream_url': rtsp_url,
                    'hls_url': hls_url,
                    'camera_name': camera.name,
                })
            else:
                # HLS failed, stop RTSP too
                streamer.stop_stream()
                return JsonResponse({
                    'status': 'error', 
                    'message': 'Failed to start HLS stream'
                }, status=500)
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Failed to start RTSP stream'
            }, status=500)
            
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        # Clean up on error
        streamer.stop_stream()
        hls_manager.stop_stream(camera_id)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["POST"])
@csrf_exempt
def stop_stream(request):
    """Stop current RTSP restream and HLS stream"""
    try:
        # Stop RTSP
        streamer.stop_stream()
        
        # Stop HLS for current camera if any
        if streamer.current_camera:
            hls_manager.stop_stream(streamer.current_camera.id)
        
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
    
    # Add HLS status
    if status['is_streaming'] and streamer.current_camera:
        camera_id = streamer.current_camera.id
        status['current_camera_id'] = camera_id
        status['hls_url'] = hls_manager.get_stream_url(camera_id)
        status['hls_streaming'] = hls_manager.is_streaming(camera_id)
    
    return JsonResponse(status)


@require_http_methods(["GET"])
def hls_playlist(request, camera_id, filename):
    """Serve HLS playlist and segments"""
    file_path = os.path.join(settings.MEDIA_ROOT, 'hls', f'camera_{camera_id}', filename)
    
    if os.path.exists(file_path):
        # Set appropriate content type
        if filename.endswith('.m3u8'):
            content_type = 'application/vnd.apple.mpegurl'
        elif filename.endswith('.ts'):
            content_type = 'video/MP2T'
        else:
            content_type = 'application/octet-stream'
        
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type=content_type)
            response['Cache-Control'] = 'no-cache'
            return response
    else:
        return HttpResponse(status=404)