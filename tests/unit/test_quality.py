from transcript_quality import assess_transcript

def test_assess_basic():
    payload = {
        "audio_ms": 60000,
        "transcript_ms": 60000,
        "tokens": [{"text":"Hello","confidence":0.95,"start_ms":100,"end_ms":200}],
        "alignment": {"alignment_score": 0.95, "unaligned_segments": [], "total_ms": 60000},
        "diarization": [{"speaker":"S1","start_ms":0,"end_ms":60000}],
        "metadata_speakers": [{"post_asr_label":"S1","full_name":"Speaker One"}],
        "text": "Hello world.",
        "audio_stats": {"snr_db": 20.0, "clipping_ratio": 0.0, "silence_ratio": 0.05},
        "expected_language": "en"
    }
    res = assess_transcript(payload)
    assert "score" in res
    assert isinstance(res["score"], float)
