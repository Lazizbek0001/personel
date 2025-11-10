import cv2
import threading
import time
import logging
import subprocess
import os
from django.conf import settings

logger = logging.getLogger(__name__)


class DynamicRTSPStreamer:
    """
    Reads a camera with OpenCV (FFmpeg backend), pushes raw frames to FFmpeg,
    and publishes a local RTSP at: rtsp://localhost:<RTSP_PORT>/camera_<id>.

    Key improvements:
      - Uses cv2.CAP_FFMPEG + small CAP_PROP_BUFFERSIZE for lower latency.
      - Reconnects with exponential backoff.
      - Low-latency RTSP output (ultrafast, zerolatency, short GOP).
      - Clean shutdown for both OpenCV and FFmpeg subprocess.
    """

    def __init__(self):
        self.current_camera = None
        self.cap = None
        self.is_streaming = False
        self.thread = None
        self.ffmpeg_process = None

        # Configurable via settings.py (with sensible defaults)
        self.rtsp_port = int(getattr(settings, "RTSP_PORT", 8554))
        self.out_width = int(getattr(settings, "STREAM_WIDTH", 640))
        self.out_height = int(getattr(settings, "STREAM_HEIGHT", 480))
        self.out_fps = int(getattr(settings, "STREAM_FPS", 25))
        self.jpeg_quality = int(getattr(settings, "STREAM_JPEG_QUALITY", 80))  # if you ever switch to MJPEG
        self.use_stderr_logs = bool(getattr(settings, "STREAM_LOG_FFMPEG", False))  # pipe ffmpeg logs if True

        # Internal stop flag for worker loop
        self._stop_ev = threading.Event()

    # ------------------------ Public API ------------------------

    def start_stream(self, camera):
        """Start streaming from a specific camera."""
        # If an existing stream is running, stop it first
        if self.is_streaming:
            self.stop_stream()

        self.current_camera = camera
        self.is_streaming = True
        self._stop_ev.clear()

        self.thread = threading.Thread(target=self._stream_worker, name="rtsp_stream_worker", daemon=True)
        self.thread.start()
        return True

    def stop_stream(self):
        """Stop current stream."""
        self.is_streaming = False
        self._stop_ev.set()
        self._cleanup()
        self.current_camera = None

    def get_current_stream_url(self):
        """Get RTSP URL for current camera."""
        if self.current_camera and self.is_streaming:
            return f"rtsp://localhost:{self.rtsp_port}/camera_{self.current_camera.id}"
        return None

    def get_status(self):
        """Get current streaming status."""
        return {
            "is_streaming": self.is_streaming,
            "current_camera": self.current_camera.name if self.current_camera else None,
            "stream_url": self.get_current_stream_url(),
        }

    # ------------------------ Worker & helpers ------------------------

    def _open_capture(self, camera_url: str):
        """
        Open the capture using FFmpeg backend for robust RTSP handling.
        Use very small buffer to reduce latency.
        """
        cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        # low-latency buffer; some builds honor CAP_PROP_BUFFERSIZE
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass  # not all OpenCV builds expose this property

        # You can set desired size/FPS; some RTSP servers ignore these,
        # but it helps when pulling from webcams / files.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.out_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.out_height)
        cap.set(cv2.CAP_PROP_FPS, self.out_fps)
        return cap

    def _make_ffmpeg_cmd(self, stream_name: str) -> list:
        """
        Build an ffmpeg command that accepts raw BGR on stdin and publishes RTSP
        with very low latency.
        """
        rtsp_url = f"rtsp://localhost:{self.rtsp_port}/{stream_name}"

        cmd = [
            "ffmpeg",
            "-y",
            # raw BGR24 frames from stdin:
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.out_width}x{self.out_height}",
            "-r", str(self.out_fps),
            "-i", "-",  # stdin

            # Encode to H.264 fast & low-latency:
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", "12",  # short GOP for quick recovery

            # Output RTSP (TCP)
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_url,
        ]
        return cmd

    def _start_ffmpeg(self, stream_name: str):
        """Spawn FFmpeg as a subprocess that reads raw frames from stdin."""
        cmd = self._make_ffmpeg_cmd(stream_name)
        logger.info("Starting FFmpeg: %s", " ".join(cmd))

        stderr_target = subprocess.PIPE if self.use_stderr_logs else subprocess.DEVNULL
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_target,
            bufsize=0,  # unbuffered stdin
        )
        return proc

    def _stream_worker(self):
        """Main streaming loop: read frames and push into FFmpeg stdin."""
        camera_url = None
        try:
            camera_url = self.current_camera.get_stream_url()
        except Exception as e:
            logger.error("Camera has no get_stream_url(): %s", e)

        if not camera_url:
            logger.error("No camera URL; aborting stream worker.")
            self._cleanup()
            return

        stream_name = f"camera_{self.current_camera.id}"
        target_delay = 1.0 / max(1, self.out_fps)

        backoff = 0.5
        while self.is_streaming and not self._stop_ev.is_set() and self.current_camera:
            try:
                # Open capture
                self.cap = self._open_capture(camera_url)
                if not self.cap or not self.cap.isOpened():
                    logger.error("Failed to open camera: %s", camera_url)
                    self._cleanup_capture_only()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)
                    continue
                backoff = 0.5  # reset backoff on success

                # Spawn ffmpeg
                self.ffmpeg_process = self._start_ffmpeg(stream_name)
                if not self.ffmpeg_process or self.ffmpeg_process.stdin is None:
                    logger.error("Failed to start FFmpeg process.")
                    self._cleanup()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)
                    continue

                logger.info("Started streaming from %s at rtsp://localhost:%d/%s",
                            self.current_camera.name, self.rtsp_port, stream_name)

                last_log_ts = time.time()

                # Pump frames
                while (self.is_streaming and not self._stop_ev.is_set()
                       and self.current_camera and self.cap.isOpened()):
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        # lost feed → reconnect
                        logger.warning("No frame received; reconnecting...")
                        break

                    # Normalize to desired size (defensive; many cameras already match)
                    if frame.shape[1] != self.out_width or frame.shape[0] != self.out_height:
                        frame = cv2.resize(frame, (self.out_width, self.out_height), interpolation=cv2.INTER_AREA)

                    # Write raw BGR bytes to ffmpeg stdin
                    try:
                        self.ffmpeg_process.stdin.write(frame.tobytes())
                    except (BrokenPipeError, ValueError) as ex:
                        logger.error("FFmpeg pipe error: %s", ex)
                        break

                    # (optional) log every ~10s to avoid noisy logs
                    now = time.time()
                    if now - last_log_ts > 10:
                        last_log_ts = now
                        logger.debug("Streaming OK: %s", stream_name)

                    # regulate loop (avoid pegging CPU)
                    # note: don't sleep too much; you want real-time; a tiny nap is enough
                    time.sleep(target_delay * 0.2)

                # if we exit inner loop, cleanup and try reconnect
                self._cleanup_ffmpeg_only()
                self._cleanup_capture_only()
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

            except Exception as e:
                logger.exception("Streaming error for %s: %s",
                                 getattr(self.current_camera, "name", "(unknown)"), e)
                self._cleanup()
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

        # final cleanup on exit
        self._cleanup()

    # ------------------------ Cleanup helpers ------------------------

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
                try:
                    if self.ffmpeg_process.stdin:
                        try:
                            self.ffmpeg_process.stdin.flush()
                        except Exception:
                            pass
                        self.ffmpeg_process.stdin.close()
                except Exception:
                    pass
                # terminate and wait shortly; kill if needed
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_process.kill()
        except Exception as e:
            logger.debug("FFmpeg cleanup error: %s", e)
        finally:
            self.ffmpeg_process = None

    def _cleanup(self):
        self._cleanup_capture_only()
        self._cleanup_ffmpeg_only()
