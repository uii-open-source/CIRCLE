"""
Prompt-based segmentation subpackage.

ReXGroundingCT — text-guided 3D CT lesion segmentation.
Organised into functional modules:
    - text_encoder:   Qwen3-based prompt feature extractor
    - seg_head:       Text-guided 3D decoder 
    - rex_segmentation: Full pipeline 
"""

from .text_encoder import Qwen3TextEncoder
from .seg_head import (
    TextGuidedFiLM,
    TextCrossAttention3D,
    TextGuidedSegmentationHead,
)
from .rex_segmentation import RexGroundingCTSeg

__all__ = [
    "Qwen3TextEncoder",
    "TextGuidedFiLM",
    "TextCrossAttention3D",
    "TextGuidedSegmentationHead",
    "RexGroundingCTSeg",
]
