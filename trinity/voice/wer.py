"""
Word Error Rate grader + STT command-set harness (Story 1.2 eval half; Voice I/O §7).

WER = (substitutions + deletions + insertions) / reference_words, computed by
word-level edit distance after light normalization (lowercase, strip punctuation,
collapse whitespace). This is the deterministic gate the voice eval suite (Story
1.6 / 3.1) keys on: no STT swap ships unless WER holds-or-improves.

The harness runs a provider over a fixed command set (command-set.yaml) and reports
per-item plus a micro-averaged aggregate. Point it at a folder of recorded WAVs to
get a real number — no live mic needed.

    python -m trinity.voice.wer --audio-dir C:\\path\\to\\command_wavs

Target repo path: trinity/voice/wer.py
Reads:           trinity/voice/command-set.yaml
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

COMMAND_SET_PATH = Path(__file__).parent / "command-set.yaml"
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> list[str]:
    """Lowercase, fold contractions (drop apostrophes), drop other punctuation,
    collapse whitespace -> word list. (Apostrophes are removed rather than split
    on, so "what's" -> "whats" instead of two tokens.)"""
    text = text.lower().replace("'", "").replace("\u2019", "")
    return _PUNCT.sub(" ", text).split()


@dataclass(frozen=True)
class WERResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int
    hyp_words: int


def _levenshtein_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Word-level edit distance with backtrace -> (subs, dels, ins)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    i, j, s, d, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s, i, j = s + 1, i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d, i = d + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return s, d, ins


def word_error_rate(reference: str, hypothesis: str) -> WERResult:
    ref, hyp = normalize(reference), normalize(hypothesis)
    s, d, ins = _levenshtein_counts(ref, hyp)
    n = len(ref)
    if n == 0:
        wer = 0.0 if not hyp else 1.0
    else:
        wer = (s + d + ins) / n
    return WERResult(wer, s, d, ins, n, len(hyp))


def aggregate_wer(pairs: list[tuple[str, str]]) -> dict:
    """Micro-averaged WER over (reference, hypothesis) pairs: total errors / total
    reference words (the standard way to aggregate, not a mean of per-item rates)."""
    per_item, tot_err, tot_ref = [], 0, 0
    for ref, hyp in pairs:
        r = word_error_rate(ref, hyp)
        per_item.append(r)
        tot_err += r.substitutions + r.deletions + r.insertions
        tot_ref += r.ref_words
    overall = tot_err / tot_ref if tot_ref else (0.0 if tot_err == 0 else 1.0)
    return {"wer": overall, "total_errors": tot_err, "total_ref_words": tot_ref,
            "items": per_item}


def load_command_set(path: Path = COMMAND_SET_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("commands", [])


def run_command_set(stt, *, audio_dir: str | None = None,
                    path: Path = COMMAND_SET_PATH) -> dict:
    """Transcribe each command's audio with `stt` and score WER against its reference.
    `stt` must expose transcribe_file(path) -> str. Items with no resolvable audio are
    skipped (reported under `skipped`)."""
    commands = load_command_set(path)
    pairs, scored, skipped = [], [], []
    for c in commands:
        ref = c["reference"]
        audio = c.get("audio")
        if audio and audio_dir:
            audio = str(Path(audio_dir) / audio)
        if not audio:
            skipped.append(c["id"])
            continue
        hyp = stt.transcribe_file(audio)
        pairs.append((ref, hyp))
        scored.append({"id": c["id"], "reference": ref, "hypothesis": hyp})
    report = aggregate_wer(pairs) if pairs else {"wer": None, "total_errors": 0,
                                                 "total_ref_words": 0, "items": []}
    report["scored"] = scored
    report["skipped"] = skipped
    return report


if __name__ == "__main__":
    import argparse

    from .config import ProviderConfig
    from .stt import make_stt

    ap = argparse.ArgumentParser(description="Score an STT provider's WER on the command set")
    ap.add_argument("--audio-dir", help="folder of recorded WAVs named per command-set 'audio'")
    ap.add_argument("--provider", default="faster-whisper", help="stt provider (faster-whisper|stub)")
    ap.add_argument("--model", default="small")
    args = ap.parse_args()

    stt = make_stt(ProviderConfig(args.provider, {"model": args.model}))
    rep = run_command_set(stt, audio_dir=args.audio_dir)
    print(f"Aggregate WER: {rep['wer']}  ({rep['total_errors']} errors / "
          f"{rep['total_ref_words']} ref words)")
    for s in rep["scored"]:
        print(f"  [{s['id']}] ref={s['reference']!r}  hyp={s['hypothesis']!r}")
    if rep["skipped"]:
        print(f"  skipped (no audio): {', '.join(rep['skipped'])}")
