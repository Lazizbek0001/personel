import cv2
import threading
import time
import logging
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
from django.conf import settings

logger = logging.getLogger(__name__)

class DynamicRTSPStreamer:
    def __init__(self):
        self.current_camera = None
        self.stream = None
        self.writer = None
        self.is_streaming = False
        self.thread = None
        
        # RTSP server configuration
        self.rtsp_port = getattr(settings, 'RTSP_PORT', 8554)
        self.output_params = {
            "-f": "rtsp",
            "-rtsp_transport": "tcp",
            "-vcodec": "libx264", 
            "-preset": "medium",
            "-tune": "zerolatency",
            "-pix_fmt": "yuv420p",
            "-r": "25",
        }
    
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
        """Main streaming worker"""
        camera_url = self.current_camera.get_stream_url()
        
        while self.is_streaming and self.current_camera:
            try:
                # Camera options based on stream type
                camera_options = {}
                if self.current_camera.stream_type == 'local':
                    camera_options = {
                        "CAP_PROP_FRAME_WIDTH": 1280,
                        "CAP_PROP_FRAME_HEIGHT": 720, 
                        "CAP_PROP_FPS": 25,
                        "CAP_PROP_BUFFERSIZE": 1,
                    }
                
                # Initialize camera stream
                self.stream = CamGear(
                    source=camera_url,
                    logging=True,
                    **camera_options
                ).start()
                
                # Initialize RTSP writer with dynamic stream name
                stream_name = f"camera_{self.current_camera.id}"
                rtsp_url = f'rtsp://0.0.0.0:{self.rtsp_port}/{stream_name}'
                
                self.writer = WriteGear(
                    output_filename=rtsp_url,
                    logging=True,
                    **self.output_params
                )
                
                logger.info(f"Started streaming from {self.current_camera.name} at {rtsp_url}")
                
                # Stream frames
                while self.is_streaming and self.current_camera:
                    frame = self.stream.read()
                    if frame is None:
                        logger.warning("No frame received from camera")
                        break
                    
                    self.writer.write(frame)
                    
            except Exception as e:
                logger.error(f"Streaming error for {self.current_camera.name}: {e}")
                self._cleanup()
                time.sleep(2)  # Wait before reconnecting
    
    def stop_stream(self):
        """Stop current stream"""
        self.is_streaming = False
        self.current_camera = None
        self._cleanup()
    
    def _cleanup(self):
        """Clean up resources"""
        try:
            if self.stream:
                self.stream.stop()
            if self.writer:
                self.writer.close()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        finally:
            self.stream = None
            self.writer = None
    
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