# Django Core Imports
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators import gzip
from django.contrib import messages
from django.urls import reverse_lazy
from django.views.generic import CreateView, UpdateView, DeleteView, ListView

# Model and Form Imports
from .models import Camera
from .forms import CameraForm

# Python Standard Library Imports
import threading
import time
import logging
import os

# Computer Vision Imports
import cv2
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)

# Import your custom RTSP restreamer
from .rtsp_streamer import DynamicRTSPStreamer

# ------------------------ Global RTSP streamer ------------------------
streamer = DynamicRTSPStreamer()


# ======================== UI Views (unchanged) ========================
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
        # Stop MJPEG worker for this camera (so handles are released)
        _stop_worker(camera.id)
        messages.success(request, f'Camera "{camera.name}" deleted successfully!')
        return super().delete(request, *args, **kwargs)


# ======================== MJPEG Background Worker ========================
class CameraWorker:
    """
    Per-camera background reader that:
      - opens the camera via FFmpeg backend (better for RTSP),
      - keeps the latest JPEG in memory (shared by all viewers),
      - auto-reconnects with backoff,
      - controls FPS and JPEG quality to reduce CPU.
    """
    def __init__(self, url, width=640, height=480, fps=15, jpeg_quality=70):
        self.url = url
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.jpeg_q = int(jpeg_quality)
        self.cap = None
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, name=f"mjpeg_worker", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        # joining is optional (daemon thread), but try to be clean
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._release()

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)  # FFmpeg backend helps RTSP
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # tiny buffer → low latency
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _release(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    def _loop(self):
        backoff = 0.3
        target_delay = 1.0 / self.fps
        # Precompute a "connecting..." placeholder JPEG
        placeholder = self._make_placeholder("Connecting...")
        while self.running:
            try:
                self.cap = self._open()
                if not self.cap.isOpened():
                    logger.warning("MJPEG worker: failed to open %s", self.url)
                    self._release()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)
                    with self.lock:
                        self.latest_jpeg = placeholder
                    continue
                backoff = 0.3

                while self.running and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        logger.warning("MJPEG worker: no frame from %s; reconnecting...", self.url)
                        break

                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                    ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
                    if ok2:
                        with self.lock:
                            self.latest_jpeg = buf.tobytes()
                    else:
                        with self.lock:
                            self.latest_jpeg = placeholder

                    # small sleep to regulate CPU (don’t fully block real-time)
                    time.sleep(target_delay)

                # reconnect path
                self._release()
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

            except Exception as e:
                logger.exception("MJPEG worker error (%s): %s", self.url, e)
                self._release()
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                with self.lock:
                    self.latest_jpeg = placeholder

    @staticmethod
    def _make_placeholder(text="No Signal"):
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, text, (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            return buf.tobytes() if ok else b""
        except Exception:
            return b""


# --------------- Worker registry (one per camera id) ---------------
_workers = {}  # camera_id -> CameraWorker
_workers_lock = threading.Lock()

def get_worker_for(camera_id):
    with _workers_lock:
        if camera_id in _workers:
            return _workers[camera_id]
        cam = Camera.objects.get(id=camera_id)
        url = cam.get_stream_url()
        # You can tune per-camera settings here if needed
        w = CameraWorker(url=url, width=640, height=480, fps=15, jpeg_quality=70)
        w.start()
        _workers[camera_id] = w
        return w

def _stop_worker(camera_id):
    with _workers_lock:
        w = _workers.pop(camera_id, None)
    if w:
        try:
            w.stop()
        except Exception as e:
            logger.debug("Error stopping worker for camera %s: %s", camera_id, e)


# ======================== RTSP RESTREAM API ========================
@require_http_methods(["POST"])
@csrf_exempt
def start_stream(request, camera_id):
    """Start RTSP restream from specific camera (via ffmpeg)."""
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
    """Stop current RTSP restream."""
    try:
        streamer.stop_stream()
        return JsonResponse({'status': 'success', 'message': 'Stream stopped'})
    except Exception as e:
        logger.error(f"Error stopping RTSP stream: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@require_http_methods(["GET"])
def stream_status(request):
    """Get current RTSP restream status."""
    status = streamer.get_status()
    if status['is_streaming'] and streamer.current_camera:
        status['current_camera_id'] = streamer.current_camera.id
    return JsonResponse(status)


# ======================== MJPEG Web Stream ========================
@gzip.gzip_page
@require_http_methods(["GET"])
def mjpeg_stream(request, camera_id):
    """
    MJPEG stream endpoint.
    Uses a shared CameraWorker per camera (low CPU, no freezing).
    """
    worker = get_worker_for(camera_id)
    boundary = b"--frame"

    def gen():
        # Tiny fallback JPEG in case worker hasn't produced a frame yet
        fallback = CameraWorker._make_placeholder("Connecting...")
        while True:
            chunk = worker.get_jpeg()
            if not chunk:
                chunk = fallback
                time.sleep(0.1)
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"

    try:
        resp = StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')
        resp['Cache-Control'] = 'no-cache'
        resp['Pragma'] = 'no-cache'
        return resp
    except Exception as e:
        logger.error(f"MJPEG stream error: {e}")
        # fallback generator with placeholder
        def fallback_gen():
            while True:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + CameraWorker._make_placeholder("Stream Error") + b'\r\n')
        return StreamingHttpResponse(fallback_gen(), content_type='multipart/x-mixed-replace; boundary=frame')
