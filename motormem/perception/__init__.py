"""Stream B: text-prompted segmentation -> numbered ObjectEstimates and overlays for the brain."""
from .core import DEFAULT_PROMPTS, Perception
from .oracle import OraclePerception
from .overlay import describe, render_overlay
from .segmenters import (GroundedSam2Segmenter, MaskCandidate, Sam3Segmenter, Segmenter, make_segmenter)

__all__ = ["DEFAULT_PROMPTS", "Perception", "OraclePerception", "describe", "render_overlay",
           "GroundedSam2Segmenter", "MaskCandidate", "Sam3Segmenter", "Segmenter", "make_segmenter"]
