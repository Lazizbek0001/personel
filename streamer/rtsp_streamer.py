import cv2
import threading
import time
import logging
import subprocess
import os
from django.conf import settings

logger = logging.getLogger(__name__)

class DynamicRTSPStreamer:
    def __init__(self):
        self.current_camera = None
        self.cap = None
        self.is_streaming = False
        self.thread = None
        self.ffmpeg_process = None
        
        self.rtsp_port = getattr(settings, 'RTSP_PORT', 8554)
    
    def start_stream(self, camera):
        """Start streaming from a specific camera"""
        if self.is_streaming:
            self.stop_stream()
        
        self.current_camera = camera
        self.is_streaming = True
        
        self.thread = threading.Thread(target=self._stream_worker)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def _stream_worker(self):
        """Main streaming worker using OpenCV and FFmpeg"""
        camera_url = self.current_camera.get_stream_url()
        
        while self.is_streaming and self.current_camera:
            try:
                # Initialize camera with OpenCV
                self.cap = cv2.VideoCapture(camera_url)
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 25)
                
                if not self.cap.isOpened():
                    logger.error(f"Failed to open camera: {camera_url}")
                    time.sleep(2)
                    continue
                
                # FFmpeg command for RTSP streaming
                stream_name = f"camera_{self.current_camera.id}"
                rtsp_url = f'rtsp://localhost:{self.rtsp_port}/{stream_name}'
                
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', '640x480',
                    '-r', '25',
                    '-i', '-',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-tune', 'zerolatency',
                    '-f', 'rtsp',
                    '-rtsp_transport', 'tcp',
                    rtsp_url
                ]
                
                # Start FFmpeg process
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                logger.info(f"Started streaming from {self.current_camera.name} at {rtsp_url}")
                
                # Stream frames
                while self.is_streaming and self.current_camera:
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("No frame received from camera")
                        break
                    
                    # Resize frame if needed
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Write frame to FFmpeg
                    try:
                        self.ffmpeg_process.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        logger.error("FFmpeg pipe broken")
                        break
                    
            except Exception as e:
                logger.error(f"Streaming error for {self.current_camera.name}: {e}")
                self._cleanup()
                time.sleep(2)
    
    def stop_stream(self):
        """Stop current stream"""
        self.is_streaming = False
        self.current_camera = None
        self._cleanup()
    
    def _cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
            if self.ffmpeg_process:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        finally:
            self.cap = None
            self.ffmpeg_process = None
    
    def get_current_stream_url(self):
        """Get RTSP URL for current camera"""
        if self.current_camera and self.is_streaming:
            return f'rtsp://localhost:{self.rtsp_port}/camera_{self.current_camera.id}'
        return None
    
    def get_status(self):
        """Get current streaming status"""
        return {
            'is_streaming': self.is_streaming,
            'current_camera': self.current_camera.name if self.current_camera else None,
            'stream_url': self.get_current_stream_url(),
        }