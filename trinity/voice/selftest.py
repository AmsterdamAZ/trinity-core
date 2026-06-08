"""
Headless self-test for the Story 1.1 voice scaffold. No mic, no models, no model
calls — pure logic. Run: python -m trinity.voice.selftest

Covers:
  1. StateMachine: legal/illegal transitions and listener notification order.
  2. capture_utterance: closes after exactly the tuned end-of-speech window.
  3. run_once (stub providers + fake agent): full IDLE->LISTENING->THINKING->IDLE.

Target repo path: trinity/voice/selftest.py
"""
from __future__ import annotations

from .agent_runner import FakeAgentRunner
from .app import capture_utterance, run_once
from .capture import StubAudioSource
from .config import ProviderConfig, registry
from .segmenter import StreamingSentenceSegmenter, collecting_sink
from .state import StateMachine, VoiceState
from .vad import StubVAD
from .wer import aggregate_wer, run_command_set, word_error_rate


def test_state_machine() -> None:
    seen: list[tuple[str, str]] = []
    sm = StateMachine()
    sm.on_change(lambda o, n: seen.append((o.value, n.value)))
    sm.transition(VoiceState.LISTENING)
    sm.transition(VoiceState.THINKING)
    sm.transition(VoiceState.IDLE)
    assert seen == [("IDLE", "LISTENING"), ("LISTENING", "THINKING"), ("THINKING", "IDLE")], seen

    # illegal jump IDLE -> THINKING must raise
    sm2 = StateMachine()
    try:
        sm2.transition(VoiceState.THINKING)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on IDLE->THINKING")
    print("PASS  state machine: legal order recorded, illegal transition rejected")


def test_capture_window() -> None:
    frame_ms = 30
    # 5 speech frames, then long silence; end-of-speech at 700ms => 24 silence frames kept
    src = StubAudioSource(frame_ms=frame_ms, lead=2, speech=5, trail=50)
    audio = capture_utterance(src, StubVAD(), end_of_speech_ms=700, frame_ms=frame_ms)
    frame_bytes = int(16000 * frame_ms / 1000) * 2
    silence_frames = -(-700 // frame_ms)        # ceil(700/30) = 24
    expected = (5 + silence_frames) * frame_bytes
    assert len(audio) == expected, (len(audio), expected)
    print(f"PASS  capture: closed at end-of-speech ({len(audio)} bytes = 5 speech + "
          f"{silence_frames} trailing-silence frames)")


def test_full_loop() -> None:
    reg = registry()
    reg.capture = ProviderConfig("stub", {"sample_rate": 16000, "frame_ms": 30})
    reg.stt = ProviderConfig("stub", {"transcript": "what's my briefing"})
    object.__setattr__(reg.wake_word, "provider", "stub")
    object.__setattr__(reg.vad, "provider", "stub")

    order: list[str] = []
    sm = StateMachine()
    sm.on_change(lambda o, n: order.append(n.value))
    resp = run_once(reg, FakeAgentRunner(), sm)

    # starts in IDLE (no-op, not emitted); recorded changes are the real transitions
    assert order == ["LISTENING", "THINKING", "IDLE"], order
    assert sm.state == VoiceState.IDLE, sm.state
    assert "what's my briefing" in resp, resp
    print("PASS  full loop: IDLE->LISTENING->THINKING->IDLE, agent reached with transcript")


# --- Story 1.3: segmenter -----------------------------------------------------

_FMT = "pcm_s16le_24k"


def test_segmenter_hold_one_ahead() -> None:
    out, sink = collecting_sink()
    s = StreamingSentenceSegmenter(sink, "t")
    s.feed("One. Two. Three.")
    s.finalize()
    assert [e["segment_id"] for e in out] == [0, 1, 2], out
    assert [e["is_final"] for e in out] == [False, False, True], out
    assert [e["text"] for e in out] == ["One.", "Two.", "Three."], out
    print("PASS  segmenter: 3 sentences, monotonic ids, exactly the last is_final")


def test_segmenter_guards() -> None:
    out, sink = collecting_sink()
    s = StreamingSentenceSegmenter(sink, "t")
    s.feed("MRR hit $2.3M today. Dr. Lee agreed. Done.")
    s.finalize()
    assert [e["text"] for e in out] == [
        "MRR hit $2.3M today.", "Dr. Lee agreed.", "Done."], out
    print("PASS  segmenter: decimals ($2.3M) and abbreviations (Dr.) not split")


def test_segmenter_streaming_equivalence() -> None:
    text = "Alpha beta. Gamma delta? Epsilon!"
    whole, s1 = collecting_sink()
    seg1 = StreamingSentenceSegmenter(s1, "t"); seg1.feed(text); seg1.finalize()
    charwise, s2 = collecting_sink()
    seg2 = StreamingSentenceSegmenter(s2, "t")
    for c in text:
        seg2.feed(c)
    seg2.finalize()
    assert whole == charwise, (whole, charwise)
    print("PASS  segmenter: char-by-char stream == whole-string (streaming invariant)")


def test_segmenter_release_on_next_start() -> None:
    out, sink = collecting_sink()
    s = StreamingSentenceSegmenter(sink, "t")
    s.feed("First sentence. ")            # S1 complete; held, nothing emitted yet
    assert out == [], "held segment must not emit before the next sentence begins"
    s.feed("S")                           # first char of S2 -> S1 releases now
    assert len(out) == 1 and out[0]["text"] == "First sentence." and not out[0]["is_final"], out
    print("PASS  segmenter: held released on next-sentence START (first-segment latency unchanged)")


def test_segmenter_maxchars() -> None:
    out, sink = collecting_sink()
    s = StreamingSentenceSegmenter(sink, "t", max_chars=20)
    s.feed("this is a very long run on clause with no terminal punctuation at all ")
    s.finalize()
    assert len(out) >= 2, out
    assert out[-1]["is_final"] is True and all(not e["is_final"] for e in out[:-1]), out
    assert [e["segment_id"] for e in out] == list(range(len(out))), out
    print(f"PASS  segmenter: max-char safety break produced {len(out)} segments, last is_final")


def test_segmenter_edges() -> None:
    out, sink = collecting_sink()
    s = StreamingSentenceSegmenter(sink, "t"); s.feed("just one line"); s.finalize()
    assert out == [{"session_id": "t", "segment_id": 0, "text": "just one line",
                    "is_final": True, "audio_format": _FMT}], out
    out2, sink2 = collecting_sink()
    StreamingSentenceSegmenter(sink2, "t").finalize()
    assert out2 == [{"session_id": "t", "segment_id": 0, "text": "",
                     "is_final": True, "audio_format": _FMT}], out2
    print("PASS  segmenter: single unterminated sentence + empty stream edges")


# --- Story 1.2: WER grader + command-set harness ------------------------------

def test_wer_grader() -> None:
    assert word_error_rate("how are my apps doing", "how are my apps doing").wer == 0.0
    sub = word_error_rate("how are my apps doing", "how are my apps going")
    assert (sub.substitutions, sub.deletions, sub.insertions) == (1, 0, 0) and sub.wer == 0.2, sub
    dele = word_error_rate("deploy the latest build", "deploy latest build")
    assert (dele.substitutions, dele.deletions, dele.insertions) == (0, 1, 0) and dele.wer == 0.25, dele
    ins = word_error_rate("any errors", "any new errors")
    assert (ins.substitutions, ins.deletions, ins.insertions) == (0, 0, 1) and ins.wer == 0.5, ins
    empty = word_error_rate("hello world", "")
    assert empty.deletions == 2 and empty.wer == 1.0, empty
    print("PASS  WER: substitution/deletion/insertion counts and rates correct")


def test_wer_normalization() -> None:
    # case + punctuation + contraction apostrophe must not register as errors
    r = word_error_rate("What's my briefing?", "whats my briefing")
    assert r.wer == 0.0, r
    print("PASS  WER: case, punctuation, and contractions normalized away")


def test_aggregate_wer() -> None:
    rep = aggregate_wer([("a b c", "a b c"), ("d e", "d x")])
    assert rep["total_errors"] == 1 and rep["total_ref_words"] == 5, rep
    assert rep["wer"] == 0.2, rep   # micro-average: 1 error / 5 ref words
    print("PASS  WER: micro-averaged aggregate (total errors / total ref words)")


def test_command_set_harness() -> None:
    import os
    import tempfile
    import yaml as _yaml
    from pathlib import Path as _Path

    fixture = {"commands": [
        {"id": "a", "reference": "how are my apps doing", "audio": "a.wav"},
        {"id": "b", "reference": "any errors", "audio": "b.wav"},
    ]}
    tmp = _Path(tempfile.mkdtemp())
    fp = tmp / "cs.yaml"
    fp.write_text(_yaml.safe_dump(fixture), encoding="utf-8")

    class FakeFileSTT:
        def transcribe_file(self, path: str) -> str:
            return {"a.wav": "how are my apps going", "b.wav": "any errors"}[os.path.basename(path)]

    rep = run_command_set(FakeFileSTT(), audio_dir=str(tmp), path=fp)
    assert rep["total_errors"] == 1 and rep["total_ref_words"] == 7, rep   # a:1 sub/5, b:0/2
    assert abs(rep["wer"] - 1 / 7) < 1e-9, rep
    assert rep["skipped"] == [] and len(rep["scored"]) == 2, rep
    print("PASS  WER harness: scores a provider over the command set, micro-averages")


if __name__ == "__main__":
    test_state_machine()
    test_capture_window()
    test_full_loop()
    test_segmenter_hold_one_ahead()
    test_segmenter_guards()
    test_segmenter_streaming_equivalence()
    test_segmenter_release_on_next_start()
    test_segmenter_maxchars()
    test_segmenter_edges()
    test_wer_grader()
    test_wer_normalization()
    test_aggregate_wer()
    test_command_set_harness()
    print("\nAll Story 1.1 + 1.3 + 1.2 self-tests passed.")
