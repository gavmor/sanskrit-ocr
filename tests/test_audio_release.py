"""
Unit tests for Sanskrit Audio Track Release generation.
"""

from pathlib import Path
import pytest
from generate_audio_release import assemble_chant_audio, CHAPTER_1_METADATA


def test_audio_metadata_structure():
    assert "verses" in CHAPTER_1_METADATA
    assert len(CHAPTER_1_METADATA["verses"]) >= 4
    for v in CHAPTER_1_METADATA["verses"]:
        assert "sanskrit" in v
        assert "verse_no" in v


def test_assemble_chant_audio(tmp_path):
    out_dir = tmp_path / "audio_test"
    success = assemble_chant_audio(out_dir)
    assert success is True

    mp3_file = out_dir / "yavanajataka_ch1_sample.mp3"
    wav_file = out_dir / "yavanajataka_ch1_sample.wav"
    manifest_file = out_dir / "track_manifest.json"

    assert mp3_file.exists() and mp3_file.stat().st_size > 10000
    assert wav_file.exists() and wav_file.stat().st_size > 10000
    assert manifest_file.exists()
