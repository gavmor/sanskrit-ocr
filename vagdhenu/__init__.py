"""
Vāgdhenu — Sanskrit Chanting, Metrical Scansion & Text Normalization Engine
Based on: https://github.com/prathoshap/vagdhenu
Developed by Prof. Prathosh, Indian Institute of Science (IISc), Bengaluru.
License: Apache-2.0
"""

from .chandas_labeler import (
    to_slp1,
    scan as scan_meter,
    identify as identify_meter,
    syllabify_slp1,
    label_verse,
    SIGNATURES as METER_SIGNATURES,
)
from .tts_meter import (
    METERS,
    ANUSHTUBH_PADA,
    detect_meter,
    analyze as analyze_meter,
)
from .tts_syllabify import syllabify
from .tts_weight import tag_weights
from .tts_normalize import normalize as normalize_sanskrit
from .prep_text import (
    fix_colon,
    strip_punct,
    to_deva,
)

__all__ = [
    "to_slp1",
    "scan_meter",
    "identify_meter",
    "syllabify_slp1",
    "label_verse",
    "METER_SIGNATURES",
    "METERS",
    "ANUSHTUBH_PADA",
    "detect_meter",
    "analyze_meter",
    "syllabify",
    "tag_weights",
    "normalize_sanskrit",
    "fix_colon",
    "strip_punct",
    "to_deva",
]
