# transcript_quality.py
"""
Transcript quality scoring and rule-based detectors.

Input model (expected):
{
  "audio_ms": int,
  "transcript_ms": int,
  "tokens": [ {"text": str, "confidence": float, "start_ms": int, "end_ms": int}, ... ],
  "alignment": { "unaligned_segments": [ {"start_ms":int,"end_ms":int} ], "alignment_score": float },
  "diarization": [ {"speaker":"S1","start_ms":int,"end_ms":int}, ... ],
  "metadata_speakers": [ {"post_asr_label": "S1", "full_name": "..."} ],
  "text": "full transcript text",
  "audio_stats": { "snr_db": float, "clipping_ratio": float, "silence_ratio": float },
  "expected_language": "en"
}
"""

from typing import List, Dict, Any, Tuple
import math
import re
from collections import Counter
import statistics

# --- Configuration (tunable thresholds and weights) ------------------------

DEFAULT_WEIGHTS = {
    "asr_conf": 0.30,
    "alignment": 0.20,
    "timing": 0.10,
    "speaker": 0.10,
    "readability": 0.10,
    "audio": 0.10,
    "semantic": 0.10
}

THRESHOLDS = {
    "low_asr_conf": 0.70,
    "duration_abs_ms": 10000,
    "duration_pct": 0.05,
    "alignment_gap_ms": 3000,
    "overlap_ratio": 0.15,
    "snr_low_db": 10.0,
    "clipping_high": 0.01,
    "silence_high": 0.40,
    "punctuation_density_min": 0.02,  # punctuation chars per word
    "filler_rate_per_min": 5.0
}

FILLER_TOKENS = {"um", "uh", "like", "you know", "i mean", "so", "well"}


# --- Utility helpers ------------------------------------------------------

def safe_mean(values: List[float], default: float = 0.0) -> float:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --- Metric calculators ---------------------------------------------------

def mean_token_confidence(tokens: List[Dict[str, Any]]) -> float:
    if not tokens:
        return 0.0
    confs = [t.get("confidence", 0.0) for t in tokens]
    return clamp01(safe_mean(confs))


def alignment_quality_score(alignment: Dict[str, Any]) -> float:
    # alignment_score expected 0..1 if provided; otherwise derive from unaligned segments
    if not alignment:
        return 0.0
    if "alignment_score" in alignment:
        return clamp01(float(alignment["alignment_score"]))
    unaligned = alignment.get("unaligned_segments", [])
    total_unaligned = sum((s["end_ms"] - s["start_ms"]) for s in unaligned) if unaligned else 0
    # assume transcript length available via alignment or caller; fallback to 1 minute
    total_ms = alignment.get("total_ms", max(60000, total_unaligned))
    ratio = total_unaligned / total_ms if total_ms > 0 else 1.0
    return clamp01(1.0 - ratio)


def normalized_duration_score(audio_ms: int, transcript_ms: int) -> float:
    if audio_ms is None or transcript_ms is None or audio_ms <= 0:
        return 0.0
    diff = abs(audio_ms - transcript_ms)
    if diff <= THRESHOLDS["duration_abs_ms"] or (diff / audio_ms) <= THRESHOLDS["duration_pct"]:
        return 1.0
    # degrade linearly up to 100% mismatch
    score = 1.0 - clamp01((diff - THRESHOLDS["duration_abs_ms"]) / max(audio_ms, 1))
    return clamp01(score)


def speaker_match_score(diarization: List[Dict[str, Any]], metadata_speakers: List[Dict[str, Any]]) -> float:
    if not diarization:
        return 0.0
    meta_labels = {s.get("post_asr_label", "").lower() for s in (metadata_speakers or [])}
    # count unique diarization labels if present
    diar_labels = {d.get("speaker", "").lower() for d in diarization if d.get("speaker")}
    if not meta_labels:
        # no metadata to compare; score based on diarization stability (fewer flips)
        durations = [d["end_ms"] - d["start_ms"] for d in diarization if d.get("end_ms") and d.get("start_ms")]
        avg_dur = safe_mean(durations, default=0)
        # longer average speaker segments -> better
        return clamp01(min(1.0, avg_dur / 30000.0))
    # fraction of diar labels that match metadata labels
    if not diar_labels:
        return 0.0
    match_frac = len(diar_labels.intersection(meta_labels)) / max(1, len(diar_labels))
    return clamp01(match_frac)


def punctuation_score(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r"\w+", text)
    if not words:
        return 0.0
    punct = re.findall(r"[.!?,;:]", text)
    density = len(punct) / max(1, len(words))
    # map density to 0..1 with expected min threshold
    score = clamp01(density / max(THRESHOLDS["punctuation_density_min"], 1e-6))
    return score


def audio_quality_score(audio_stats: Dict[str, Any]) -> float:
    if not audio_stats:
        return 0.0
    snr = audio_stats.get("snr_db")
    clipping = audio_stats.get("clipping_ratio", 0.0)
    silence = audio_stats.get("silence_ratio", 0.0)
    # SNR mapping: >=20 excellent, 10 borderline
    snr_score = 0.0
    if snr is None:
        snr_score = 0.5
    else:
        snr_score = clamp01((snr - THRESHOLDS["snr_low_db"]) / max(1.0, 20.0 - THRESHOLDS["snr_low_db"]))
    clip_penalty = clamp01(1.0 - clipping / max(THRESHOLDS["clipping_high"], 1e-6))
    silence_penalty = clamp01(1.0 - silence / max(THRESHOLDS["silence_high"], 1e-6))
    # combine multiplicatively to penalize any single bad factor
    combined = snr_score * clip_penalty * silence_penalty
    return clamp01(combined)


def semantic_coherence_score(text: str, expected_language: str = None) -> float:
    if not text:
        return 0.0
    # simple heuristics: language detection placeholder and OOV rate via non-word tokens
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    # filler rate reduces score
    filler_count = sum(1 for w in words if w in FILLER_TOKENS)
    filler_rate = filler_count / max(1, len(words)) * 60.0  # per 100 words scaled
    filler_penalty = clamp01(1.0 - min(1.0, filler_rate / THRESHOLDS["filler_rate_per_min"]))
    # crude named-entity plausibility: presence of capitalized tokens (approx)
    caps = sum(1 for w in re.findall(r"\b[A-Z][a-z]+\b", text))
    ne_score = clamp01(min(1.0, caps / max(1, len(words) * 0.02)))
    return clamp01(0.6 * ne_score + 0.4 * filler_penalty)


# --- Rule detectors -------------------------------------------------------

def detect_duration_mismatch(audio_ms: int, transcript_ms: int) -> List[Dict[str, Any]]:
    issues = []
    if audio_ms is None or transcript_ms is None:
        return issues
    diff = abs(audio_ms - transcript_ms)
    if diff > max(THRESHOLDS["duration_abs_ms"], THRESHOLDS["duration_pct"] * max(1, audio_ms)):
        issues.append({
            "code": "DURATION_MISMATCH",
            "severity": "high",
            "message": "Audio and transcript durations differ significantly",
            "evidence": {"audio_ms": audio_ms, "transcript_ms": transcript_ms, "diff_ms": diff}
        })
    return issues


def detect_non_monotonic_timestamps(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues = []
    last = -1
    for i, t in enumerate(tokens):
        s = t.get("start_ms")
        if s is None:
            issues.append({"code": "MISSING_TOKEN_START", "severity": "medium", "message": f"Token {i} missing start_ms"})
            continue
        if s < last:
            issues.append({"code": "NON_MONOTONIC_TIMESTAMPS", "severity": "high",
                           "message": f"Token {i} start {s} < previous {last}", "index": i})
        last = s
    return issues


def detect_low_asr_confidence(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues = []
    mean_conf = mean_token_confidence(tokens)
    if mean_conf < THRESHOLDS["low_asr_conf"]:
        issues.append({
            "code": "LOW_ASR_CONFIDENCE",
            "severity": "high",
            "message": f"Mean token confidence low: {mean_conf:.2f}",
            "evidence": {"mean_confidence": mean_conf}
        })
    # detect clusters of low confidence
    window = 10
    confs = [t.get("confidence", 0.0) for t in tokens]
    for i in range(0, max(0, len(confs) - window + 1)):
        w = confs[i:i+window]
        if safe_mean(w) < 0.5:
            issues.append({"code": "CONFIDENCE_SPIKE", "severity": "medium",
                           "message": f"Low-confidence cluster at token index {i}", "index": i})
            break
    return issues


def detect_alignment_gaps(alignment: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = []
    if not alignment:
        return issues
    for seg in alignment.get("unaligned_segments", []):
        dur = seg["end_ms"] - seg["start_ms"]
        if dur >= THRESHOLDS["alignment_gap_ms"]:
            issues.append({
                "code": "ALIGNMENT_GAP",
                "severity": "medium",
                "message": f"Unaligned segment {dur} ms",
                "evidence": seg
            })
    return issues


def detect_speaker_mismatch(diarization: List[Dict[str, Any]], metadata_speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues = []
    if not diarization or not metadata_speakers:
        return issues
    meta_labels = {s.get("post_asr_label", "").lower() for s in metadata_speakers}
    diar_labels = {d.get("speaker", "").lower() for d in diarization if d.get("speaker")}
    missing = diar_labels - meta_labels
    if missing:
        issues.append({
            "code": "SPEAKER_MISMATCH",
            "severity": "medium",
            "message": "Diarization contains labels not present in metadata",
            "evidence": {"diar_labels": list(diar_labels), "meta_labels": list(meta_labels)}
        })
    # overlap detection
    total = 0
    overlap = 0
    # compute overlap ratio roughly
    sorted_seg = sorted(diarization, key=lambda x: x["start_ms"])
    for i in range(len(sorted_seg)-1):
        a = sorted_seg[i]
        b = sorted_seg[i+1]
        total += (a["end_ms"] - a["start_ms"])
        if b["start_ms"] < a["end_ms"]:
            overlap += (a["end_ms"] - b["start_ms"])
    if total > 0 and (overlap / total) > THRESHOLDS["overlap_ratio"]:
        issues.append({
            "code": "HIGH_OVERLAP",
            "severity": "medium",
            "message": "High speaker overlap ratio",
            "evidence": {"overlap_ratio": overlap / total}
        })
    return issues


def detect_audio_issues(audio_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = []
    if not audio_stats:
        return issues
    snr = audio_stats.get("snr_db")
    clipping = audio_stats.get("clipping_ratio", 0.0)
    silence = audio_stats.get("silence_ratio", 0.0)
    if snr is not None and snr < THRESHOLDS["snr_low_db"]:
        issues.append({"code": "LOW_SNR", "severity": "high", "message": f"SNR low: {snr} dB"})
    if clipping > THRESHOLDS["clipping_high"]:
        issues.append({"code": "CLIPPING", "severity": "high", "message": f"Clipping ratio high: {clipping:.3f}"})
    if silence > THRESHOLDS["silence_high"]:
        issues.append({"code": "EXCESSIVE_SILENCE", "severity": "medium", "message": f"Silence ratio high: {silence:.2f}"})
    return issues


def detect_punctuation_and_readability(text: str) -> List[Dict[str, Any]]:
    issues = []
    if not text:
        return issues
    words = re.findall(r"\w+", text)
    punct = re.findall(r"[.!?,;:]", text)
    density = len(punct) / max(1, len(words))
    if density < THRESHOLDS["punctuation_density_min"]:
        issues.append({"code": "POOR_PUNCTUATION", "severity": "medium", "message": f"Punctuation density low: {density:.3f}"})
    # filler detection
    filler_count = sum(1 for w in (w.lower() for w in words) if w in FILLER_TOKENS)
    # estimate per-minute filler rate assuming 150 wpm
    minutes = max(1.0, len(words) / 150.0)
    filler_per_min = filler_count / minutes
    if filler_per_min > THRESHOLDS["filler_rate_per_min"]:
        issues.append({"code": "FILLER_RATE_HIGH", "severity": "low", "message": f"Filler rate high: {filler_per_min:.1f}/min"})
    return issues


# --- Main scoring and detection API --------------------------------------

def compute_metrics(payload: Dict[str, Any], weights: Dict[str, float] = None) -> Dict[str, Any]:
    w = weights or DEFAULT_WEIGHTS
    tokens = payload.get("tokens", [])
    alignment = payload.get("alignment", {})
    diarization = payload.get("diarization", [])
    metadata_speakers = payload.get("metadata_speakers", [])
    audio_ms = payload.get("audio_ms")
    transcript_ms = payload.get("transcript_ms")
    text = payload.get("text", "")
    audio_stats = payload.get("audio_stats", {})

    metrics = {}
    metrics["asr_conf"] = mean_token_confidence(tokens)
    metrics["alignment"] = alignment_quality_score(alignment)
    metrics["timing"] = normalized_duration_score(audio_ms, transcript_ms)
    metrics["speaker"] = speaker_match_score(diarization, metadata_speakers)
    metrics["readability"] = punctuation_score(text)
    metrics["audio"] = audio_quality_score(audio_stats)
    metrics["semantic"] = semantic_coherence_score(text, payload.get("expected_language"))

    # combine into final score 0..100
    total = 0.0
    for k, weight in w.items():
        total += weight * metrics.get(k, 0.0)
    score = clamp01(total) * 100.0

    return {"score": round(score, 2), "metrics": metrics}


def run_detectors(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    tokens = payload.get("tokens", [])
    alignment = payload.get("alignment", {})
    diarization = payload.get("diarization", [])
    metadata_speakers = payload.get("metadata_speakers", [])
    audio_ms = payload.get("audio_ms")
    transcript_ms = payload.get("transcript_ms")
    text = payload.get("text", "")
    audio_stats = payload.get("audio_stats", {})

    issues = []
    issues.extend(detect_duration_mismatch(audio_ms, transcript_ms))
    issues.extend(detect_non_monotonic_timestamps(tokens))
    issues.extend(detect_low_asr_confidence(tokens))
    issues.extend(detect_alignment_gaps(alignment))
    issues.extend(detect_speaker_mismatch(diarization, metadata_speakers))
    issues.extend(detect_audio_issues(audio_stats))
    issues.extend(detect_punctuation_and_readability(text))

    # deduplicate by code + message
    seen = set()
    dedup = []
    for it in issues:
        key = (it.get("code"), it.get("message"))
        if key not in seen:
            seen.add(key)
            dedup.append(it)
    return dedup


def assess_transcript(payload: Dict[str, Any], weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Returns:
    {
      "score": float,
      "metrics": {...},
      "issues": [ {code, severity, message, evidence?}, ... ],
      "quality_band": "Excellent|Good|Fair|Poor"
    }
    """
    result = compute_metrics(payload, weights=weights)
    issues = run_detectors(payload)
    score = result["score"]
    if score >= 90:
        band = "Excellent"
    elif score >= 75:
        band = "Good"
    elif score >= 50:
        band = "Fair"
    else:
        band = "Poor"
    result.update({"issues": issues, "quality_band": band})
    return result


# --- CLI-friendly example (not executed on import) ------------------------

if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcript_quality.py <payload.json>")
        sys.exit(2)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)
    out = assess_transcript(payload)
    print(json.dumps(out, indent=2))
