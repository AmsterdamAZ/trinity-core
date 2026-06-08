"""
Speech-to-text providers behind the STTProvider interface (Story 1.2).

FasterWhisperSTT — the real engine, in-process (no HTTP/WS server). faster-whisper
is a CTranslate2 reimplementation of Whisper (~4x faster, minimal accuracy loss;
Voice I/O §2). Accepts both live PCM (raw s16le mono from capture) via transcribe()
and audio files via transcribe_file() — the latter is what the WER harness uses, so
you can score a recorded command set with no live mic.

StubSTT — fixed transcript so the pipeline + WER harness run headless. Selected via
`stt.provider: stub` in the voice registry; swap to `faster-whisper` is config-only.

Note on "partials + one final" (AC): faster-whisper is chunk-based, not truly
streaming — fine for the wake-word-burst interaction (Voice I/O §2 caveat). The
in-process contract returns the final transcript; partial-emission belongs to the
optional HTTP/WS streaming shape, deferred with that endpoint.

Target repo path: trinity/voice/stt.py
"""
from __future__ import annotations

from .config import ProviderConfig
from .interfaces import STTProvider


class StubSTT(STTProvider):
    """Returns a fixed transcript so the loop/WER harness run without models."""

    def __init__(self, transcript: str = "what's my briefing"):
        self.transcript = transcript

    def transcribe(self, audio: bytes) -> str:
        return self.transcript

    def transcribe_file(self, path: str) -> str:
        return self.transcript


class FasterWhisperSTT(STTProvider):
    def __init__(self, model: str = "small", device: str = "cpu",
                 compute_type: str = "int8", sample_rate: int = 16000,
                 beam_size: int = 5, vad_filter: bool = True):
        self.model_size = model
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # heavy; real use only
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "faster-whisper not installed. `uv pip install faster-whisper`, or set "
                "stt.provider: stub in voice-registry.yaml to run headless."
            ) from exc
        # CPU/int8 is the robust default (no CUDA deps). For the 4070 SUPER, set
        # device: cuda, compute_type: float16 in the registry once CUDA libs are present.
        self._model = WhisperModel(self.model_size, device=self.device,
                                   compute_type=self.compute_type)

    def _decode(self, audio_or_path) -> str:
        self._ensure_model()
        segments, _info = self._model.transcribe(
            audio_or_path, beam_size=self.beam_size, vad_filter=self.vad_filter)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe(self, audio: bytes) -> str:
        """Transcribe raw PCM (s16le, mono, self.sample_rate) from live capture."""
        import numpy as np
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        return self._decode(samples)

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file (WAV/etc); faster-whisper decodes via ffmpeg."""
        return self._decode(path)


def make_stt(cfg: ProviderConfig) -> STTProvider:
    s = cfg.settings
    if cfg.provider == "stub":
        return StubSTT(s.get("transcript", "what's my briefing"))
    if cfg.provider == "faster-whisper":
        return FasterWhisperSTT(
            model=s.get("model", "small"),
            device=s.get("device", "cpu"),
            compute_type=s.get("compute_type", "int8"),
            sample_rate=int(s.get("sample_rate", 16000)),
        )
    raise ValueError(f"Unknown stt provider {cfg.provider!r}")
