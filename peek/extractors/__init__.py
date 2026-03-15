"""
Extractor subpackage.
"""

from .torch import extract_torch_latents
from .ultralytics import extract_ultralytics_latents
from .yolov5 import extract_yolov5_latents

__all__ = ["extract_torch_latents", "extract_ultralytics_latents", "extract_yolov5_latents"]
