"""Gate Vision - local number-plate and face logging for CCTV gates.

Everything runs offline on your own PC. No subscriptions, no cloud AI.
"""
import os

# Must be set before OpenCV is imported anywhere: force RTSP over TCP (no
# corrupted frames on a busy LAN) and time out dead connections after 5 s.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000"
)

__version__ = "0.1.0"
