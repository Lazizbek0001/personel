import cv2
import threading
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators import gzip

from .models import *

import rtsp_streamer as streamer
# Add this to your existing views.py

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
                return None
        return self.cameras[camera_id]
    
    def generate_frames(self, camera_id):
        """Generate MJPEG frames"""
        camera_stream = self.get_camera_stream(camera_id)
        if camera_stream is None:
            return
        
        while True:
            with self.locks[camera_id]:
                success, frame = camera_stream.read()
            
            if not success:
                break
            
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Global MJPEG streamer instance
mjpeg_streamer = MJPEGStreamer()

@gzip.gzip_page
@require_http_methods(["GET"])
def mjpeg_stream(request, camera_id):
    """MJPEG stream endpoint"""
    try:
        return StreamingHttpResponse(
            mjpeg_streamer.generate_frames(camera_id),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        # Return a blank image or error image
        return StreamingHttpResponse(
            generate_blank_frame(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )

def generate_blank_frame():
    """Generate a blank frame when camera is unavailable"""
    import numpy as np
    # Create a blank black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add error text
    cv2.putText(frame, "Camera Unavailable", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, jpeg = cv2.imencode('.jpg', frame)
    if ret:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Update your stream status to include camera ID
@require_http_methods(["GET"])
def stream_status(request):
    """Get current stream status"""
    status = streamer.get_status()
    if status['is_streaming'] and streamer.current_camera:
        status['current_camera_id'] = streamer.current_camera.id
    return JsonResponse(status)