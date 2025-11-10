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
from concurrent.futures import ThreadPoolExecutor

# Computer Vision Imports
import cv2
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)

# Import your custom RTSP restreamer
from .rtsp_streamer import DynamicRTSPStreamer

# ------------------------ Global RTSP streamer ------------------------
streamer = DynamicRTSPStreamer()

# ------------------------ Global Thread Pool ------------------------
thread_pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="stream_worker")

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


# ======================== High-Performance MJPEG System ========================
class FrameBuffer:
    """Thread-safe circular buffer for storing frames"""
    def __init__(self, max_size=10):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_count = 0
    
    def put(self, frame_data):
        with self.lock:
            self.buffer.append(frame_data)
            self.latest_frame = frame_data
            self.frame_count += 1
    
    def get_latest(self):
        with self.lock:
            return self.latest_frame
    
    def get_all(self):
        with self.lock:
            return list(self.buffer)


class CameraWorker:
    """
    High-performance camera worker with:
    - FFmpeg backend for RTSP stability
    - Frame buffering to prevent freezing
    - Adaptive frame rate control
    - Connection pooling
    - Memory-efficient frame handling
    """
    def __init__(self, url, width=640, height=480, target_fps=15, jpeg_quality=70, buffer_size=5):
        self.url = url
        self.width = int(width)
        self.height = int(height)
        self.target_fps = max(1, int(target_fps))
        self.jpeg_quality = int(jpeg_quality)
        self.buffer_size = buffer_size
        
        self.frame_buffer = FrameBuffer(max_size=buffer_size)
        self.cap = None
        self.running = False
        self.thread = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / self.target_fps
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        # Performance metrics
        self.frames_processed = 0
        self.connection_errors = 0
        self.last_perf_log = time.time()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, name=f"cam_worker_{id(self)}", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._cleanup()

    def get_latest_frame(self):
        return self.frame_buffer.get_latest()

    def _initialize_camera(self):
        """Initialize camera with optimal settings"""
        try:
            # Try different backends for best compatibility
            backends = [
                cv2.CAP_FFMPEG,
                cv2.CAP_ANY
            ]
            
            for backend in backends:
                cap = cv2.VideoCapture(self.url, backend)
                
                if cap.isOpened():
                    # Optimize camera settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                    # Test frame read
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logger.info(f"Camera initialized successfully with backend {backend}")
                        return cap
                    else:
                        cap.release()
                
            logger.error("All camera backends failed")
            return None
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return None

    def _process_frame(self, frame):
        """Process and optimize frame"""
        try:
            # Resize if necessary (more efficient than letting OpenCV do it)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            
            # Convert to JPEG with quality control
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success, jpeg_data = cv2.imencode('.jpg', frame, encode_params)
            
            if success:
                return jpeg_data.tobytes()
            else:
                logger.warning("Frame encoding failed")
                return None
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None

    def _create_placeholder(self, message="Connecting..."):
        """Create optimized placeholder frame"""
        try:
            # Create a simple placeholder to minimize CPU
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, message, (30, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            success, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            return jpeg_data.tobytes() if success else b''
        except Exception:
            return b''

    def _processing_loop(self):
        """Main processing loop with adaptive performance"""
        placeholder = self._create_placeholder()
        reconnect_delay = 1.0
        last_successful_frame = time.time()
        
        while self.running:
            try:
                # Initialize camera connection
                self.cap = self._initialize_camera()
                if not self.cap or not self.cap.isOpened():
                    logger.warning(f"Failed to open camera: {self.url}")
                    self.frame_buffer.put(placeholder)
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 10.0)  # Exponential backoff
                    continue
                
                # Reset reconnection parameters on successful connection
                reconnect_delay = 1.0
                self.connection_attempts = 0
                logger.info(f"Camera connected: {self.url}")
                
                # Frame processing loop
                while self.running and self.cap.isOpened():
                    current_time = time.time()
                    
                    # Adaptive frame rate control
                    if current_time - self.last_frame_time < self.frame_interval:
                        time.sleep(0.001)  # Small sleep to prevent CPU spinning
                        continue
                    
                    # Read frame
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        logger.warning("Invalid frame received")
                        # Check if we've been without valid frames for too long
                        if time.time() - last_successful_frame > 10.0:
                            break  # Reconnect
                        time.sleep(0.1)
                        continue
                    
                    # Process frame in thread pool for better parallelism
                    future = thread_pool.submit(self._process_frame, frame)
                    try:
                        jpeg_data = future.result(timeout=2.0)  # 2 second timeout
                        if jpeg_data:
                            self.frame_buffer.put(jpeg_data)
                            self.frames_processed += 1
                            last_successful_frame = time.time()
                            self.last_frame_time = current_time
                        else:
                            self.frame_buffer.put(placeholder)
                    except Exception as e:
                        logger.warning(f"Frame processing timeout/error: {e}")
                        self.frame_buffer.put(placeholder)
                    
                    # Performance logging (every 30 seconds)
                    if current_time - self.last_perf_log > 30:
                        logger.info(f"Camera worker stats - FPS: {self.frames_processed/30:.1f}, "
                                  f"Frames: {self.frames_processed}")
                        self.frames_processed = 0
                        self.last_perf_log = current_time
                
            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                self.connection_errors += 1
                self.frame_buffer.put(placeholder)
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, 10.0)
            
            finally:
                self._cleanup()
                if self.running:  # Only sleep if we're planning to reconnect
                    time.sleep(reconnect_delay)

    def _cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")


# ======================== Worker Management ========================
_workers = {}
_workers_lock = threading.Lock()
_worker_refs = weakref.WeakValueDictionary()

def get_worker_for(camera_id):
    """Get or create worker for camera with connection pooling"""
    with _workers_lock:
        if camera_id in _workers:
            worker = _workers[camera_id]
            if worker.thread and worker.thread.is_alive():
                return worker
            else:
                # Worker died, create new one
                del _workers[camera_id]
        
        try:
            cam = Camera.objects.get(id=camera_id)
            url = cam.get_stream_url()
            
            # Adaptive settings based on camera type
            if cam.stream_type == 'local':
                # Local cameras can handle higher FPS
                worker = CameraWorker(url=url, width=640, height=480, target_fps=25, jpeg_quality=80, buffer_size=3)
            else:
                # Network cameras - more conservative settings
                worker = CameraWorker(url=url, width=640, height=480, target_fps=15, jpeg_quality=70, buffer_size=5)
            
            worker.start()
            _workers[camera_id] = worker
            _worker_refs[camera_id] = worker  # Keep weak reference
            
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

def cleanup_idle_workers():
    """Clean up workers for deleted or inactive cameras"""
    with _workers_lock:
        current_camera_ids = set(Camera.objects.filter(is_active=True).values_list('id', flat=True))
        dead_workers = []
        
        for camera_id, worker in _workers.items():
            if camera_id not in current_camera_ids or not worker.thread or not worker.thread.is_alive():
                dead_workers.append(camera_id)
        
        for camera_id in dead_workers:
            worker = _workers.pop(camera_id, None)
            if worker:
                try:
                    worker.stop()
                except Exception:
                    pass
        
        if dead_workers:
            logger.info(f"Cleaned up {len(dead_workers)} idle workers")


# ======================== RTSP RESTREAM API ========================
@require_http_methods(["POST"])
@csrf_exempt
def start_stream(request, camera_id):
    """Start RTSP restream from specific camera"""
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


# ======================== High-Performance MJPEG Stream ========================
@gzip.gzip_page
@require_http_methods(["GET"])
def mjpeg_stream(request, camera_id):
    """
    High-performance MJPEG stream endpoint with:
    - Connection pooling
    - Frame buffering
    - Adaptive quality
    - Efficient memory usage
    """
    worker = get_worker_for(camera_id)
    if not worker:
        return _error_stream("Camera not available")
    
    def generate_stream():
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_frame = None
        
        while True:
            try:
                frame_data = worker.get_latest_frame()
                
                if frame_data:
                    consecutive_errors = 0
                    last_frame = frame_data
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                else:
                    consecutive_errors += 1
                    if last_frame:
                        # Reuse last good frame to prevent freezing
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
                    else:
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + 
                              worker._create_placeholder("No Signal") + b'\r\n')
                    
                    if consecutive_errors > max_consecutive_errors:
                        logger.warning(f"Too many consecutive errors for camera {camera_id}")
                        break
                
                # Adaptive sleep based on frame rate
                time.sleep(1.0 / worker.target_fps)
                
            except Exception as e:
                logger.error(f"MJPEG stream error: {e}")
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + 
                      worker._create_placeholder("Stream Error") + b'\r\n')
                time.sleep(0.1)
    
    try:
        response = StreamingHttpResponse(generate_stream(), 
                                       content_type='multipart/x-mixed-replace; boundary=frame')
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response
    except Exception as e:
        logger.error(f"MJPEG stream setup error: {e}")
        return _error_stream("Stream setup failed")

def _error_stream(message):
    """Generate error stream"""
    def error_generator():
        placeholder = CameraWorker._create_placeholder(message)
        while True:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(1.0)
    
    return StreamingHttpResponse(error_generator(), 
                               content_type='multipart/x-mixed-replace; boundary=frame')