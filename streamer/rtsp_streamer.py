import cv2
import threading
import time
import logging
import subprocess
import os
import queue
import psutil
from django.conf import settings

logger = logging.getLogger(__name__)


class DynamicRTSPStreamer:
    """
    High-performance RTSP streamer with:
    - Connection pooling and reuse
    - Adaptive bitrate control
    - System resource monitoring
    - Frame skipping for overload protection
    - Multiple encoding profiles
    """

    def __init__(self):
        self.current_camera = None
        self.cap = None
        self.is_streaming = False
        self.thread = None
        self.ffmpeg_process = None

        # Enhanced configuration
        self.rtsp_port = int(getattr(settings, "RTSP_PORT", 8554))
        self.out_width = int(getattr(settings, "STREAM_WIDTH", 640))
        self.out_height = int(getattr(settings, "STREAM_HEIGHT", 480))
        self.out_fps = int(getattr(settings, "STREAM_FPS", 25))
        self.max_bitrate = getattr(settings, "STREAM_MAX_BITRATE", "1000k")
        self.use_hw_acceleration = getattr(settings, "STREAM_HW_ACCEL", False)
        self.use_stderr_logs = getattr(settings, "STREAM_LOG_FFMPEG", False)
        
        # Performance tuning
        self.frame_skip_threshold = 0.8  # Skip frames if CPU > 80%
        self.target_frame_time = 1.0 / self.out_fps
        self.last_perf_check = time.time()
        self.frames_processed = 0
        self.frames_skipped = 0
        
        # Thread control
        self._stop_ev = threading.Event()
        self._frame_queue = queue.Queue(maxsize=10)  # Prevent memory buildup

    # ------------------------ Public API ------------------------
    def start_stream(self, camera):
        """Start streaming from a specific camera"""
        if self.is_streaming:
            self.stop_stream()

        self.current_camera = camera
        self.is_streaming = True
        self._stop_ev.clear()
        self.frames_processed = 0
        self.frames_skipped = 0

        self.thread = threading.Thread(target=self._stream_worker, name="rtsp_stream_worker", daemon=True)
        self.thread.start()
        
        # Start frame producer in separate thread
        self.producer_thread = threading.Thread(target=self._frame_producer, name="frame_producer", daemon=True)
        self.producer_thread.start()
        
        return True

    def stop_stream(self):
        """Stop current stream"""
        self.is_streaming = False
        self._stop_ev.set()
        self._cleanup()
        self.current_camera = None

    def get_current_stream_url(self):
        """Get RTSP URL for current camera"""
        if self.current_camera and self.is_streaming:
            return f"rtsp://localhost:{self.rtsp_port}/camera_{self.current_camera.id}"
        return None

    def get_status(self):
        """Get current streaming status with performance metrics"""
        return {
            "is_streaming": self.is_streaming,
            "current_camera": self.current_camera.name if self.current_camera else None,
            "stream_url": self.get_current_stream_url(),
            "performance": {
                "frames_processed": self.frames_processed,
                "frames_skipped": self.frames_skipped,
                "queue_size": self._frame_queue.qsize()
            }
        }

    # ------------------------ Frame Production ------------------------
    def _frame_producer(self):
        """Produce frames and put them in queue"""
        camera_url = None
        try:
            camera_url = self.current_camera.get_stream_url()
        except Exception as e:
            logger.error("Camera has no get_stream_url(): %s", e)
            return

        backoff = 0.5
        last_frame_time = 0
        
        while self.is_streaming and not self._stop_ev.is_set() and self.current_camera:
            try:
                # Open capture with optimized settings
                self.cap = self._open_capture(camera_url)
                if not self.cap or not self.cap.isOpened():
                    logger.error("Failed to open camera: %s", camera_url)
                    self._cleanup_capture_only()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)
                    continue
                
                backoff = 0.5  # Reset on success
                logger.info("Frame producer started for %s", camera_url)

                # Frame production loop
                while self.is_streaming and not self._stop_ev.is_set() and self.cap.isOpened():
                    current_time = time.time()
                    
                    # Rate limiting
                    if current_time - last_frame_time < self.target_frame_time:
                        time.sleep(0.001)
                        continue
                    
                    # System load check - skip frames if system is overloaded
                    if self._is_system_overloaded():
                        self.frames_skipped += 1
                        time.sleep(0.01)
                        continue
                    
                    # Read frame
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        logger.warning("No frame received; reconnecting...")
                        break
                    
                    # Resize if necessary
                    if frame.shape[1] != self.out_width or frame.shape[0] != self.out_height:
                        frame = cv2.resize(frame, (self.out_width, self.out_height), 
                                         interpolation=cv2.INTER_LINEAR)
                    
                    # Put frame in queue (non-blocking)
                    try:
                        self._frame_queue.put(frame, timeout=0.1)
                        self.frames_processed += 1
                        last_frame_time = current_time
                    except queue.Full:
                        self.frames_skipped += 1
                        # Queue full, skip this frame
                        continue
                
                # Reconnection logic
                self._cleanup_capture_only()
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

            except Exception as e:
                logger.exception("Frame producer error: %s", e)
                self._cleanup_capture_only()
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    # ------------------------ Stream Encoding ------------------------
    def _stream_worker(self):
        """Consume frames from queue and encode to RTSP"""
        if not self.current_camera:
            return

        stream_name = f"camera_{self.current_camera.id}"
        backoff = 0.5
        last_log_time = time.time()

        while self.is_streaming and not self._stop_ev.is_set() and self.current_camera:
            try:
                # Start FFmpeg process
                self.ffmpeg_process = self._start_ffmpeg(stream_name)
                if not self.ffmpeg_process:
                    logger.error("Failed to start FFmpeg process")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)
                    continue

                logger.info("RTSP encoder started for %s", stream_name)
                backoff = 0.5

                # Encoding loop
                while self.is_streaming and not self._stop_ev.is_set() and self.ffmpeg_process.poll() is None:
                    try:
                        # Get frame from queue (with timeout)
                        frame = self._frame_queue.get(timeout=1.0)
                        
                        # Write to FFmpeg
                        try:
                            self.ffmpeg_process.stdin.write(frame.tobytes())
                        except (BrokenPipeError, ValueError) as ex:
                            logger.error("FFmpeg pipe error: %s", ex)
                            break
                        
                        # Performance logging
                        current_time = time.time()
                        if current_time - last_log_time > 10:
                            logger.debug("RTSP streaming OK: %s (queue: %d)", 
                                       stream_name, self._frame_queue.qsize())
                            last_log_time = current_time

                    except queue.Empty:
                        # No frames in queue, continue
                        continue

                # FFmpeg process ended or error occurred
                self._cleanup_ffmpeg_only()
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

            except Exception as e:
                logger.exception("Stream worker error: %s", e)
                self._cleanup_ffmpeg_only()
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    # ------------------------ Optimized Helpers ------------------------
    def _open_capture(self, camera_url: str):
        """Open camera capture with multiple backend fallbacks"""
        backends = [
            cv2.CAP_FFMPEG,  # Best for RTSP
            cv2.CAP_GSTREAMER,  # Good alternative
            cv2.CAP_ANY  # Fallback
        ]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_url, backend)
                if cap.isOpened():
                    # Optimize capture settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.out_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.out_height)
                    cap.set(cv2.CAP_PROP_FPS, self.out_fps)
                    
                    # Test read
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logger.info(f"Camera opened with backend {backend}")
                        return cap
                    else:
                        cap.release()
            except Exception as e:
                logger.debug(f"Backend {backend} failed: {e}")
                continue
        
        logger.error("All camera backends failed")
        return None

    def _make_ffmpeg_cmd(self, stream_name: str) -> list:
        """Build optimized FFmpeg command"""
        rtsp_url = f"rtsp://localhost:{self.rtsp_port}/{stream_name}"
        
        cmd = [
            "ffmpeg",
            "-y",
            "-fflags", "nobuffer",  # Reduce buffering
            "-flags", "low_delay",  # Low latency
            "-strict", "experimental",
            
            # Input from stdin (raw video)
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.out_width}x{self.out_height}",
            "-r", str(self.out_fps),
            "-i", "-",
            
            # Video encoding settings
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-crf", "23",  # Quality factor
            "-maxrate", self.max_bitrate,
            "-bufsize", "2000k",
            "-g", "10",  # Short GOP
            "-keyint_min", "10",
            "-x264-params", "scenecut=0:open_gop=0:min-keyint=10:keyint=10",
            
            # Output format
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "-muxdelay", "0.1",
            rtsp_url
        ]
        
        # Hardware acceleration (if available)
        if self.use_hw_acceleration and self._check_hw_acceleration():
            cmd = self._add_hw_acceleration(cmd)
        
        return cmd

    def _add_hw_acceleration(self, cmd: list) -> list:
        """Add hardware acceleration to FFmpeg command"""
        # Try different hardware accelerators
        hw_accels = [
            ("h264_nvenc", "cuda"),  # NVIDIA
            ("h264_vaapi", "vaapi"),  # Intel
            ("h264_v4l2m2m", "v4l2m2m"),  # Raspberry Pi
        ]
        
        for encoder, _ in hw_accels:
            if self._check_encoder_available(encoder):
                # Replace software encoder with hardware encoder
                for i, item in enumerate(cmd):
                    if item == "libx264":
                        cmd[i] = encoder
                        logger.info(f"Using hardware encoder: {encoder}")
                        return cmd
        return cmd

    def _check_encoder_available(self, encoder: str) -> bool:
        """Check if encoder is available"""
        try:
            result = subprocess.run([
                "ffmpeg", "-hide_banner", "-encoders"
            ], capture_output=True, text=True, timeout=5)
            return encoder in result.stdout
        except Exception:
            return False

    def _check_hw_acceleration(self) -> bool:
        """Check if hardware acceleration is available"""
        return any([
            self._check_encoder_available("h264_nvenc"),
            self._check_encoder_available("h264_vaapi"),
            self._check_encoder_available("h264_v4l2m2m"),
        ])

    def _is_system_overloaded(self) -> bool:
        """Check if system is overloaded and we should skip frames"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Skip frames if system is under heavy load
            if cpu_percent > 80 or memory_percent > 85:
                return True
                
            # Performance logging
            current_time = time.time()
            if current_time - self.last_perf_check > 30:
                logger.info(f"System load - CPU: {cpu_percent}%, Memory: {memory_percent}%")
                self.last_perf_check = current_time
                
            return False
            
        except Exception:
            return False

    def _start_ffmpeg(self, stream_name: str):
        """Start FFmpeg process with error handling"""
        cmd = self._make_ffmpeg_cmd(stream_name)
        logger.info("Starting FFmpeg: %s", " ".join(cmd))

        try:
            stderr_target = subprocess.PIPE if self.use_stderr_logs else subprocess.DEVNULL
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_target,
                bufsize=0,  # Unbuffered
            )
            
            # Monitor FFmpeg stderr in separate thread if logging enabled
            if self.use_stderr_logs:
                threading.Thread(
                    target=self._monitor_ffmpeg_logs,
                    args=(proc,),
                    daemon=True
                ).start()
                
            return proc
            
        except Exception as e:
            logger.error("Failed to start FFmpeg: %s", e)
            return None

    def _monitor_ffmpeg_logs(self, proc):
        """Monitor FFmpeg logs in separate thread"""
        try:
            while proc.poll() is None and self.is_streaming:
                line = proc.stderr.readline()
                if line:
                    logger.debug("FFmpeg: %s", line.decode().strip())
                time.sleep(0.1)
        except Exception:
            pass

    # ------------------------ Cleanup ------------------------
    def _cleanup_capture_only(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception as e:
            logger.debug("Capture release error: %s", e)
        finally:
            self.cap = None

    def _cleanup_ffmpeg_only(self):
        try:
            if self.ffmpeg_process is not None:
                # Close stdin
                if self.ffmpeg_process.stdin:
                    try:
                        self.ffmpeg_process.stdin.close()
                    except Exception:
                        pass
                
                # Terminate process
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait()
                    
        except Exception as e:
            logger.debug("FFmpeg cleanup error: %s", e)
        finally:
            self.ffmpeg_process = None

    def _cleanup(self):
        self._cleanup_capture_only()
        self._cleanup_ffmpeg_only()
        # Clear frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break