"""
Unit tests for pipeline_qc.py (Metrical Quality Control & Audit Tool).
"""

from pathlib import Path
import pytest
from pipeline_qc import audit_transcription


def test_qc_audit_runs_cleanly(tmp_path):
    # Create sample test markdown
    test_md = tmp_path / "test_sanskrit.md"
    content = (
        "## पत्रम् ५७ (Page 57)\n\n"
        "> आद्यः स्मृतो मेषवपुस्ततोऽथ  \n"
        "> कालस्य मूर्द्धा कथितः पुराणैः ।\n\n"
        "> स्थानान्यजाविकगोधराद्रि-  \n"
        "> स्तेनाग्निरत्नाकरलोहभाञ्जि ॥ १४ ॥\n\n"
        "---\n"
    )
    test_md.write_text(content, encoding="utf-8")

    report_path = audit_transcription(test_md, expected_syllables=11)
    assert report_path.exists()
    assert report_path.stat().st_size > 100
