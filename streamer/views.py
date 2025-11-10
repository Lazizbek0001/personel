# Django Core Imports
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, StreamingHttpResponse
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
import os
import queue
import weakref
from collections import deque

# Computer Vision Imports
import cv2
import numpy as np

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
        # Stop MJPEG worker for this camera
        _stop_worker(camera.id)
        messages.success(request, f'Camera "{camera.name}" deleted successfully!')
        return super().delete(request, *args, **kwargs)


# ======================== SIMPLE MJPEG System ========================
class SimpleCameraWorker:
    """
    Simple camera worker that directly streams without complex buffering
    """
    def __init__(self, camera_id, url, width=640, height=480, fps=15, jpeg_quality=70):
        self.camera_id = camera_id
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.cap = None
        self.running = False
        self.thread = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / fps

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, name=f"cam_worker_{self.camera_id}", daemon=True)
        self.thread.start()
        logger.info(f"Started camera worker for camera {self.camera_id}")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._cleanup()
        logger.info(f"Stopped camera worker for camera {self.camera_id}")

    def _cleanup(self):
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")

    def _create_placeholder(self, message="No Signal"):
        """Create a placeholder frame"""
        try:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, message, (30, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            success, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            return jpeg_data.tobytes() if success else b''
        except Exception:
            return b''

    def _open_camera(self):
        """Open camera with simple retry logic"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Try different backends
                backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
                
                for backend in backends:
                    cap = cv2.VideoCapture(self.url, backend)
                    if cap.isOpened():
                        # Set basic properties
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, self.fps)
                        
                        # Test read
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            logger.info(f"Camera {self.camera_id} opened with backend {backend}")
                            return cap
                        else:
                            cap.release()
                
                logger.warning(f"Camera {self.camera_id} attempt {attempt + 1} failed")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error opening camera {self.camera_id}: {e}")
                time.sleep(1)
        
        return None

    def generate_frames(self):
        """Generator that yields JPEG frames directly"""
        placeholder = self._create_placeholder("Connecting...")
        reconnect_delay = 1.0
        consecutive_failures = 0
        
        while self.running:
            try:
                # Open camera
                self.cap = self._open_camera()
                if not self.cap:
                    logger.error(f"Failed to open camera {self.camera_id}")
                    yield placeholder
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 5.0)
                    continue
                
                # Reset on successful connection
                reconnect_delay = 1.0
                consecutive_failures = 0
                logger.info(f"Camera {self.camera_id} connected, starting stream")
                
                # Stream frames
                while self.running and self.cap.isOpened():
                    current_time = time.time()
                    
                    # Rate limiting
                    if current_time - self.last_frame_time < self.frame_interval:
                        time.sleep(0.001)
                        continue
                    
                    # Read frame
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        consecutive_failures += 1
                        logger.warning(f"Camera {self.camera_id} frame read failed ({consecutive_failures})")
                        
                        if consecutive_failures > 10:
                            break  # Reconnect
                        
                        yield placeholder
                        time.sleep(0.1)
                        continue
                    
                    # Reset failure counter
                    consecutive_failures = 0
                    
                    # Resize if necessary
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    
                    # Encode to JPEG
                    success, jpeg_data = cv2.imencode('.jpg', frame, [
                        cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality
                    ])
                    
                    if success:
                        yield jpeg_data.tobytes()
                        self.last_frame_time = current_time
                    else:
                        yield placeholder
                    
                    # Small sleep to regulate CPU
                    time.sleep(0.01)
                
                # Cleanup for reconnection
                self._cleanup()
                yield placeholder
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, 5.0)
                
            except Exception as e:
                logger.error(f"Camera worker error for {self.camera_id}: {e}")
                self._cleanup()
                yield placeholder
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, 5.0)


# ======================== Worker Management ========================
_workers = {}
_workers_lock = threading.Lock()

def get_worker_for(camera_id):
    """Get or create worker for camera"""
    with _workers_lock:
        if camera_id in _workers:
            worker = _workers[camera_id]
            if worker.running:
                return worker
            else:
                # Worker died, create new one
                del _workers[camera_id]
        
        try:
            cam = Camera.objects.get(id=camera_id)
            url = cam.get_stream_url()
            
            worker = SimpleCameraWorker(
                camera_id=camera_id,
                url=url,
                width=640,
                height=480,
                fps=15,
                jpeg_quality=70
            )
            
            worker.start()
            _workers[camera_id] = worker
            logger.info(f"Created new worker for camera {camera_id}")
            return worker
            
        except Camera.DoesNotExist:
            logger.error(f"Camera {camera_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error creating worker for camera {camera_id}: {e}")
            return None

def _stop_worker(camera_id):
    """Stop and remove worker"""
    with _workers_lock:
        worker = _workers.pop(camera_id, None)
    
    if worker:
        try:
            worker.stop()
            logger.info(f"Stopped worker for camera {camera_id}")
        except Exception as e:
            logger.debug(f"Error stopping worker for camera {camera_id}: {e}")


# ======================== RTSP RESTREAM API ========================
@require_http_methods(["POST"])
@csrf_exempt
def start_stream(request, camera_id):
    """Start RTSP restream from specific camera"""
    camera = get_object_or_404(Camera, id=camera_id, is_active=True)
    try:
        # First start the RTSP stream
        success = streamer.start_stream(camera)
        
        if success:
            # Wait a bit for RTSP server to initialize
            time.sleep(2)
            
            stream_url = streamer.get_current_stream_url()
            return JsonResponse({
                'status': 'success',
                'message': f'Started streaming from {camera.name}',
                'stream_url': stream_url,
                'camera_name': camera.name,
            })
        else:
            return JsonResponse({'status': 'error', 'message': 'Failed to start stream'}, status=500)
            
    except Exception as e:
        logger.error(f"Error starting RTSP stream: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["POST"])
@csrf_exempt
def stop_stream(request):
    """Stop current RTSP restream"""
    try:
        streamer.stop_stream()
        return JsonResponse({'status': 'success', 'message': 'Stream stopped'})
    except Exception as e:
        logger.error(f"Error stopping RTSP stream: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["GET"])
def stream_status(request):
    """Get current RTSP restream status"""
    status = streamer.get_status()
    if status['is_streaming'] and streamer.current_camera:
        status['current_camera_id'] = streamer.current_camera.id
    return JsonResponse(status)


# ======================== SIMPLE MJPEG Stream ========================
@gzip.gzip_page
@require_http_methods(["GET"])
def mjpeg_stream(request, camera_id):
    """
    Simple MJPEG stream endpoint that works reliably
    """
    def generate_frames():
        """Generate frames with proper error handling"""
        worker = None
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get or create worker
                worker = get_worker_for(camera_id)
                if not worker:
                    yield from generate_error_frame("Camera not available")
                    time.sleep(2)
                    retry_count += 1
                    continue
                
                # Generate frames from worker
                frame_generator = worker.generate_frames()
                retry_count = 0  # Reset on success
                
                for frame_data in frame_generator:
                    if not worker.running:
                        break
                        
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                    
                    # Small delay to prevent overwhelming the client
                    time.sleep(0.03)
                
                # If we get here, the generator ended
                logger.warning(f"Frame generator ended for camera {camera_id}")
                break
                
            except Exception as e:
                logger.error(f"MJPEG stream error for camera {camera_id}: {e}")
                yield from generate_error_frame("Stream error")
                time.sleep(1)
                retry_count += 1
        
        # If all retries failed
        if retry_count >= max_retries:
            logger.error(f"Max retries exceeded for camera {camera_id}")
            while True:
                yield from generate_error_frame("Camera unavailable")
                time.sleep(2)

    def generate_error_frame(message):
        """Generate error frame"""
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, message, (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            success, jpeg_data = cv2.imencode('.jpg', frame)
            if success:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg_data.tobytes() + b'\r\n')
        except Exception:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n\r\n')

    try:
        response = StreamingHttpResponse(
            generate_frames(), 
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response
        
    except Exception as e:
        logger.error(f"MJPEG stream setup error: {e}")
        # Fallback response
        def fallback_generator():
            while True:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + 
                      SimpleCameraWorker(1, "")._create_placeholder("Stream Error") + b'\r\n')
                time.sleep(1)
        
        return StreamingHttpResponse(
            fallback_generator(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )