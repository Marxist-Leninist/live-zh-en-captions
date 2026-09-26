#!/usr/bin/env python3
"""Live Chinese / Hindi / Russian system-audio -> English captions for Windows.

Pipeline: Windows speaker loopback (SoundCard/WASAPI) -> rolling VAD segmenter ->
faster-whisper multilingual ASR/translation -> always-on-top Tk overlay.

Default mode uses Whisper's native task='translate', avoiding a second text MT pass.
A two-stage mode uses Chinese OPUS-MT only when Chinese is detected; other languages fall back to Whisper translation.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
import traceback
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel
import tkinter as tk

APP_DIR = Path(__file__).resolve().parent
LOG_PATH = APP_DIR / "live_captions.log"
TARGET_SR = 16000
CAPTURE_SR = 48000
CHUNK_MS = 80
DEFAULT_RMS_THRESHOLD = 0.0022
DEFAULT_STREAM_WINDOW_S = 1.6
DEFAULT_STREAM_EMIT_S = 0.35
FINAL_SILENCE_S = 0.42
MAX_UTTERANCE_S = 5.0
MIN_SPEECH_S = 0.28

# SoundCard sometimes reports recoverable Media Foundation discontinuities when a
# device starts or changes format. They are noisy but not fatal, so log real errors
# ourselves and suppress only warnings from this package.
warnings.filterwarnings("ignore", module="soundcard")

_logfh = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
print(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} ===", file=_logfh)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=_logfh, flush=True)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    n = max(1, int(round(audio.size * dst_sr / src_sr)))
    xp = np.arange(audio.size, dtype=np.float64)
    x = np.linspace(0, audio.size, n, endpoint=False, dtype=np.float64)
    return np.interp(x, xp, audio).astype(np.float32)


def put_latest(q: queue.Queue, item) -> None:
    try:
        while q.qsize() > 0:
            q.get_nowait()
    except queue.Empty:
        pass
    q.put(item)


def resolve_whisper_model(model_name: str) -> str:
    """Use a complete local faster-whisper snapshot when available.

    Passing a model alias makes huggingface_hub perform a metadata request on every
    launch. The default small model is already cached, so resolving its snapshot
    directly makes startup offline and deterministic while preserving downloads for
    any explicitly requested uncached model.
    """
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate)
    repo = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{model_name}"
    ref = repo / "refs" / "main"
    try:
        revision = ref.read_text(encoding="utf-8").strip()
        snapshot = repo / "snapshots" / revision
        required = ("config.json", "model.bin", "tokenizer.json")
        if revision and all((snapshot / name).exists() for name in required):
            return str(snapshot)
    except OSError:
        pass
    return model_name


class CaptureThread(threading.Thread):
    daemon = True

    def __init__(self, out_q: queue.Queue, status_cb):
        super().__init__(name="wasapi-loopback")
        self.out_q = out_q
        self.status_cb = status_cb
        self.running = True
        self.device_name = ""

    def run(self) -> None:
        frames = int(CAPTURE_SR * CHUNK_MS / 1000)
        while self.running:
            try:
                speaker = sc.default_speaker()
                if speaker is None:
                    raise RuntimeError("Windows has no default playback device")
                loopback = sc.get_microphone(id=str(speaker.id), include_loopback=True)
                self.device_name = speaker.name
                self.status_cb(f"Listening to {speaker.name}")
                log(f"capture open: {speaker.name} id={speaker.id} rate={CAPTURE_SR}")
                with loopback.recorder(samplerate=CAPTURE_SR, blocksize=frames) as rec:
                    while self.running:
                        block = rec.record(numframes=frames)
                        if block is None or len(block) == 0:
                            continue
                        mono = resample(block, CAPTURE_SR)
                        # Audio chunks are disposable; keep the newest if UI/model stalls.
                        if self.out_q.qsize() > 40:
                            try:
                                self.out_q.get_nowait()
                            except queue.Empty:
                                pass
                        self.out_q.put(mono)
            except Exception as exc:
                log(f"capture error: {exc}\n{traceback.format_exc()}")
                self.status_cb(f"Audio reconnecting: {exc}")
                time.sleep(1.0)


class SegmenterThread(threading.Thread):
    daemon = True

    def __init__(self, in_q: queue.Queue, out_q: queue.Queue, rms_threshold: float,
                 stream_window_s: float, stream_emit_s: float, streaming: bool = True):
        super().__init__(name="segmenter")
        self.in_q = in_q
        self.out_q = out_q
        self.rms_threshold = rms_threshold
        self.stream_window_s = stream_window_s
        self.stream_emit_s = stream_emit_s
        self.streaming = streaming
        self.running = True

    def run(self) -> None:
        rolling: deque[np.ndarray] = deque()
        rolling_samples = 0
        active: list[np.ndarray] = []
        active_samples = 0
        speech_samples = 0
        silent_samples = 0
        last_emit = 0.0
        # Recomputed every chunk so the audio-window control takes effect live.
        window_samples = int(TARGET_SR * self.stream_window_s)
        min_speech = int(TARGET_SR * MIN_SPEECH_S)
        final_silence = int(TARGET_SR * FINAL_SILENCE_S)
        max_utterance = int(TARGET_SR * MAX_UTTERANCE_S)

        while self.running:
            try:
                chunk = self.in_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if chunk is None or len(chunk) == 0:
                continue
            window_samples = int(TARGET_SR * self.stream_window_s)
            rms = float(np.sqrt(np.mean(chunk * chunk, dtype=np.float64)))
            speech = rms >= self.rms_threshold

            rolling.append(chunk)
            rolling_samples += len(chunk)
            while rolling and rolling_samples - len(rolling[0]) >= window_samples:
                old = rolling.popleft()
                rolling_samples -= len(old)

            if speech:
                if not active:
                    # Include a little leading context already in the rolling buffer.
                    pre = list(rolling)[:-1][-2:]
                    active = pre.copy()
                    active_samples = sum(len(x) for x in active)
                active.append(chunk)
                active_samples += len(chunk)
                speech_samples += len(chunk)
                silent_samples = 0
            elif active:
                active.append(chunk)
                active_samples += len(chunk)
                silent_samples += len(chunk)

            now = time.perf_counter()
            if self.streaming and speech_samples >= min_speech and now - last_emit >= self.stream_emit_s:
                audio = np.concatenate(list(rolling)) if rolling else np.empty(0, np.float32)
                if audio.size >= min_speech:
                    put_latest(self.out_q, (audio.astype(np.float32, copy=False), False, now))
                    last_emit = now

            if active and speech_samples >= min_speech and (
                silent_samples >= final_silence or active_samples >= max_utterance
            ):
                audio = np.concatenate(active).astype(np.float32, copy=False)
                put_latest(self.out_q, (audio, True, now))
                active = []
                active_samples = 0
                speech_samples = 0
                silent_samples = 0
                rolling.clear()
                rolling_samples = 0
                last_emit = now
            elif active and speech_samples < min_speech and silent_samples >= final_silence:
                active = []
                active_samples = 0
                speech_samples = 0
                silent_samples = 0


SETTINGS_PATH = Path(__file__).with_name("settings.json")

DEFAULT_SETTINGS = {
    "width_frac": 0.90,
    "height_px": 190,
    "font_size": 31,
    "alpha": 0.78,
    "stream_window": 1.6,
    "model": "",
    "engine": "direct",
}


def load_settings() -> dict:
    """User-adjustable overlay + decoding settings, persisted between runs."""
    data = dict(DEFAULT_SETTINGS)
    try:
        if SETTINGS_PATH.exists():
            import json as _json
            stored = _json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                for k in DEFAULT_SETTINGS:
                    if k in stored:
                        data[k] = stored[k]
    except Exception as exc:
        log(f"settings load failed, using defaults: {exc}")
    return data


def save_settings(data: dict) -> None:
    try:
        import json as _json
        SETTINGS_PATH.write_text(
            _json.dumps({k: data.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        log(f"settings save failed: {exc}")


def discover_models() -> list:
    """Locally available Whisper models, best-first.

    Anything under ./models is offered by name, plus the standard cached sizes.
    """
    found = []
    root = Path(__file__).with_name("models")
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "model.bin").exists():
                found.append((d.name, str(d)))
    for name in ("medium", "small", "base"):
        if not any(n == name for n, _ in found):
            cached = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{name}"
            if cached.is_dir():
                found.append((name, name))
    order = {"medium": 0, "small": 1, "base": 2}
    found.sort(key=lambda t: (order.get(t[0], 99), t[0]))
    return found


class MeshDirectTranslator:
    """Optional LLM translation via the MeshDirect auto-router.

    Far better English than a small ASR model, but measured at roughly 8 s per
    call against this deployment - too slow for partial captions, so it is only
    ever applied to finals, behind a hard timeout, and falls back to the Whisper
    text if it does not answer in time.
    """

    def __init__(self, endpoint: str, token: str, timeout_s: float = 6.0):
        self.endpoint = endpoint
        self.token = token
        self.timeout_s = timeout_s
        self.ok = bool(endpoint and token)
        self.last_error = ""

    def translate(self, text: str) -> str:
        if not self.ok or not text.strip():
            return text
        import json as _json
        import urllib.request
        body = _json.dumps({
            "model": "auto",
            "messages": [
                {"role": "system", "content":
                 "Turn speech transcripts (Mandarin, Hindi or Russian source, possibly "
                 "already machine-translated) into natural English subtitles. "
                 "Output ONLY the English translation: no notes, no pinyin or "
                 "transliteration, no quotes."},
                {"role": "user", "content": text},
            ],
            "max_tokens": 120,
            "temperature": 0.2,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                out = _json.loads(resp.read().decode("utf-8"))
            got = out["choices"][0]["message"]["content"].strip()
            return got or text
        except Exception as exc:
            self.last_error = str(exc)
            log(f"meshdirect translate failed ({exc}); keeping Whisper text")
            return text


class OpusTranslator:
    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.model_id = "Helsinki-NLP/opus-mt-zh-en"
        self.tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, local_files_only=True)
        self.model.eval()
        if device == "cuda":
            self.model.to("cuda")

    def translate(self, text: str) -> str:
        x = self.tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        if self.device == "cuda":
            x = {k: v.to("cuda") for k, v in x.items()}
            self.torch.cuda.synchronize()
        with self.torch.inference_mode():
            y = self.model.generate(**x, max_new_tokens=128, num_beams=1, do_sample=False)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        return self.tok.batch_decode(y.cpu(), skip_special_tokens=True)[0].strip()


class TranslateThread(threading.Thread):
    daemon = True

    def __init__(self, in_q: queue.Queue, ui_cb, status_cb, model_name: str,
                 device: str, compute_type: str, mode: str,
                 final_model_name: str | None = None,
                 final_compute_type: str | None = None):
        super().__init__(name="whisper-translate")
        self.in_q = in_q
        self.ui_cb = ui_cb
        self.status_cb = status_cb
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.mode = mode
        self.final_model_name = final_model_name
        self.final_compute_type = final_compute_type or compute_type
        self.running = True
        self.model = None
        self.final_model = None
        self.opus = None
        self.mesh = None
        self._pending_model = None
        self._model_lock = threading.Lock()

    def _warm_final_model(self) -> None:
        if not self.final_model_name or self.device != "cuda" or self.mode != "direct":
            return
        try:
            source = resolve_whisper_model(self.final_model_name)
            self.status_cb(f"Live captions ready; loading final Whisper {self.final_model_name}...")
            log(f"Final Whisper model source: {source}")
            t0 = time.perf_counter()
            candidate = WhisperModel(
                source,
                device=self.device,
                compute_type=self.final_compute_type,
                download_root=str(Path.home() / ".cache" / "huggingface" / "hub"),
            )
            self.final_model = candidate
            load_s = time.perf_counter() - t0
            log(
                f"final model ready: whisper={self.final_model_name} "
                f"device={self.device} compute={self.final_compute_type} load_s={load_s:.3f}"
            )
            self.status_cb("Ready: fast Small partials + accurate Medium final captions")
        except Exception as exc:
            self.final_model = None
            log(f"final model load error; continuing with primary: {exc}\n{traceback.format_exc()}")
            self.status_cb("Ready: fast captions; final model unavailable, using primary")

    def request_model(self, name: str) -> None:
        """Ask the decode thread to swap models at the next safe point."""
        with self._model_lock:
            self._pending_model = name

    def _maybe_swap_model(self) -> None:
        with self._model_lock:
            pending = self._pending_model
            self._pending_model = None
        if not pending or pending == self.model_name:
            return
        try:
            self.status_cb(f"Switching to Whisper {Path(pending).name}...")
            source = resolve_whisper_model(pending)
            t0 = time.perf_counter()
            replacement = WhisperModel(
                source,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(Path.home() / ".cache" / "huggingface" / "hub"),
            )
            old = self.model
            self.model = replacement
            self.model_name = pending
            del old
            load_s = time.perf_counter() - t0
            log(f"model switched: whisper={pending} device={self.device} "
                f"compute={self.compute_type} load_s={load_s:.3f}")
            self.status_cb(f"Now using {Path(pending).name} ({self.device}/{self.compute_type})")
        except Exception as exc:
            log(f"model switch to {pending} failed, keeping current: {exc}")
            self.status_cb(f"Could not load {Path(pending).name}; kept previous model")

    def run(self) -> None:
        try:
            self.status_cb(f"Loading Whisper {self.model_name} on {self.device}/{self.compute_type}...")
            t0 = time.perf_counter()
            model_source = resolve_whisper_model(self.model_name)
            log(f"Whisper model source: {model_source}")
            try:
                self.model = WhisperModel(
                    model_source,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(Path.home() / ".cache" / "huggingface" / "hub"),
                )
            except Exception as primary_exc:
                if self.device != "cuda":
                    raise
                log(f"CUDA model load failed, falling back to CPU/int8: {primary_exc}")
                self.status_cb(f"GPU unavailable ({primary_exc}); falling back to CPU/int8 direct mode...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.mode = "direct"
                self.model = WhisperModel(
                    model_source,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(Path.home() / ".cache" / "huggingface" / "hub"),
                )
            load_s = time.perf_counter() - t0
            if self.mode == "two-stage":
                self.status_cb("Loading Chinese->English OPUS translator...")
                self.opus = OpusTranslator("cpu")
            elif self.mode == "hybrid":
                def warm_opus():
                    try:
                        self.status_cb("Live captions ready; warming accurate Chinese translator...")
                        self.opus = OpusTranslator("cpu")
                        self.status_cb("Ready: fast live captions + accurate Chinese final captions")
                        log("OPUS accuracy model ready")
                    except Exception as exc:
                        log(f"OPUS warm error: {exc}\\n{traceback.format_exc()}")
                        self.status_cb("Live captions ready; accurate final translator unavailable")
                threading.Thread(target=warm_opus, name="opus-warm", daemon=True).start()
            self.status_cb(f"Ready: auto-detect source -> English ({self.device}/{self.compute_type}, load {load_s:.1f}s)")
            log(f"model ready: whisper={self.model_name} device={self.device} compute={self.compute_type} mode={self.mode} load_s={load_s:.3f}")
            if self.final_model_name and self.device == "cuda" and self.mode == "direct":
                threading.Thread(
                    target=self._warm_final_model,
                    name="whisper-final-warm",
                    daemon=True,
                ).start()
        except Exception as exc:
            log(f"model load error: {exc}\n{traceback.format_exc()}")
            self.status_cb(f"Model load failed: {exc}")
            return

        while self.running:
            self._maybe_swap_model()
            try:
                audio, final, queued_at = self.in_q.get(timeout=0.25)
            except queue.Empty:
                continue
            # Never let captions drift behind live speech. If a newer window arrived,
            # consume it and discard the stale one before decoding.
            dropped = 0
            while self.in_q.qsize() > 0:
                try:
                    audio, final, queued_at = self.in_q.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            if dropped:
                log(f"dropped {dropped} stale caption windows")

            try:
                t0 = time.perf_counter()
                use_two_stage = self.mode == "two-stage" or (self.mode == "hybrid" and final and self.opus is not None)
                use_final_model = final and not use_two_stage and self.final_model is not None
                decode_model = self.final_model if use_final_model else self.model
                task = "transcribe" if use_two_stage else "translate"
                segments, info = decode_model.transcribe(
                    audio,
                    language=None,
                    task=task,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=3,
                    max_new_tokens=64,
                    vad_filter=True,
                    vad_parameters={"threshold": 0.35, "min_silence_duration_ms": 160},
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    no_speech_threshold=0.72,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.2,
                )
                text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
                detected = getattr(info, "language", None) or "unknown"
                if use_two_stage and text and self.opus is not None:
                    if detected.startswith("zh"):
                        text = self.opus.translate(text)
                    elif detected == "en":
                        pass
                    else:
                        # Hybrid final on a non-Chinese/non-English segment: ask Whisper
                        # for English directly rather than sending the wrong language to OPUS.
                        seg2, _ = decode_model.transcribe(
                            audio, language=(detected if detected != "unknown" else None), task="translate", beam_size=1, best_of=1,
                            temperature=0.0, repetition_penalty=1.12,
                            no_repeat_ngram_size=3, max_new_tokens=64,
                            condition_on_previous_text=False,
                            without_timestamps=True, vad_filter=True,
                        )
                        text = " ".join(s.text.strip() for s in seg2 if s.text.strip()).strip()
                infer_s = time.perf_counter() - t0
                end_to_end_s = time.perf_counter() - queued_at
                if use_final_model:
                    log(f"used final model infer={infer_s:.3f}s")
                if text and final and self.mesh is not None and self.mesh.ok:
                    text = self.mesh.translate(text)
                if text:
                    self.ui_cb(text, final, infer_s, end_to_end_s, detected)
                    log(f"caption final={final} lang={detected} infer={infer_s:.3f}s e2e={end_to_end_s:.3f}s text={text!r}")
            except Exception as exc:
                log(f"translate error: {exc}\n{traceback.format_exc()}")
                self.status_cb(f"Translate error: {exc}")


def make_mesh_translator():
    """Build the optional LLM translator from environment configuration.

    Set MESHDIRECT_URL and MESHDIRECT_TOKEN to enable. Left unset, the LLM
    option stays visible in the menu but reports itself unavailable rather
    than silently doing nothing.
    """
    url = os.environ.get("MESHDIRECT_URL", "").strip()
    token = os.environ.get("MESHDIRECT_TOKEN", "").strip()
    if not url or not token:
        return None
    return MeshDirectTranslator(url, token)


LANGUAGE_NAMES = {
    "zh": "Chinese",
    "hi": "Hindi",
    "ru": "Russian",
    "uk": "Ukrainian",  # short Russian clips are sometimes detected as Ukrainian; avoid a bare "UK" label
    "en": "English",
}

def language_display_name(code: str | None) -> str:
    key = (code or "unknown").lower().split("-")[0]
    return LANGUAGE_NAMES.get(key, key.upper() if key != "unknown" else "Unknown")

class CaptionUI:
    def __init__(self, args):
        self.root = tk.Tk()
        self.root.title("Chinese / Hindi / Russian -> English Live Captions")
        self.root.configure(bg="#050505")
        self.root.attributes("-topmost", True)
        self.settings = load_settings()
        # An explicitly passed --font-size / --alpha wins over the stored value.
        if args.font_size != 31:
            self.settings["font_size"] = args.font_size
        if abs(args.alpha - 0.78) > 1e-9:
            self.settings["alpha"] = args.alpha
        self.root.attributes("-alpha", self.settings["alpha"])
        self.root.overrideredirect(True)
        self.visible = True
        self.events: queue.Queue = queue.Queue()
        self.final_lines = deque(maxlen=2)
        self.current = ""
        self.segmenter = None      # wired up in main()
        self.transcriber = None    # wired up in main()

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        width = int(sw * self.settings["width_frac"])
        height = int(self.settings["height_px"])
        x = int((sw - width) / 2)
        y = sh - height - 55
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        self.topbar = tk.Frame(self.root, bg="#050505")
        self.topbar.pack(fill="x")
        self.status = tk.Label(
            self.topbar, text="Starting...", fg="#B7B7B7", bg="#050505",
            font=("Segoe UI", 10), anchor="w", padx=16, pady=5,
        )
        self.status.pack(side="left", fill="x", expand=True)
        self.close_button = tk.Button(
            self.topbar, text="Close", command=self.close,
            fg="#FFFFFF", bg="#2A2A2A",
            activeforeground="#FFFFFF", activebackground="#3A3A3A",
            relief="flat", bd=0, highlightthickness=0,
            font=("Segoe UI Semibold", 10), padx=12, pady=3,
            cursor="hand2",
        )
        self.close_button.pack(side="right", padx=(4, 6), pady=3)
        self.label = tk.Label(
            self.root, text="", fg="#FFFFFF", bg="#050505",
            font=("Segoe UI Semibold", self.settings["font_size"]), justify="center",
            wraplength=width - 40, padx=20, pady=4,
        )
        self.label.pack(fill="both", expand=True)

        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.bind("<F9>", lambda _e: self.toggle())
        self.root.bind("<ButtonPress-1>", self._drag_start)
        self.root.bind("<B1-Motion>", self._drag_move)
        self.root.bind("<Button-3>", self._popup)
        self._drag_xy = (0, 0)
        self.root.bind("<Control-Right>", lambda _e: self.nudge("width_frac", 0.05))
        self.root.bind("<Control-Left>", lambda _e: self.nudge("width_frac", -0.05))
        self.root.bind("<Control-Down>", lambda _e: self.nudge("height_px", 20))
        self.root.bind("<Control-Up>", lambda _e: self.nudge("height_px", -20))
        self.root.bind("<Control-plus>", lambda _e: self.nudge("font_size", 2))
        self.root.bind("<Control-equal>", lambda _e: self.nudge("font_size", 2))
        self.root.bind("<Control-minus>", lambda _e: self.nudge("font_size", -2))
        self._build_menu()
        self.root.after(40, self.poll)

    # ------------------------------------------------------------------ menu
    def _build_menu(self):
        self.menu = tk.Menu(self.root, tearoff=0)

        size_menu = tk.Menu(self.menu, tearoff=0)
        size_menu.add_command(label="Wider            Ctrl+Right",
                              command=lambda: self.nudge("width_frac", 0.05))
        size_menu.add_command(label="Narrower         Ctrl+Left",
                              command=lambda: self.nudge("width_frac", -0.05))
        size_menu.add_separator()
        size_menu.add_command(label="Taller           Ctrl+Down",
                              command=lambda: self.nudge("height_px", 20))
        size_menu.add_command(label="Shorter          Ctrl+Up",
                              command=lambda: self.nudge("height_px", -20))
        size_menu.add_separator()
        size_menu.add_command(label="Bigger text      Ctrl++",
                              command=lambda: self.nudge("font_size", 2))
        size_menu.add_command(label="Smaller text     Ctrl+-",
                              command=lambda: self.nudge("font_size", -2))
        size_menu.add_separator()
        size_menu.add_command(label="More solid",
                              command=lambda: self.nudge("alpha", 0.06))
        size_menu.add_command(label="More see-through",
                              command=lambda: self.nudge("alpha", -0.06))
        size_menu.add_separator()
        size_menu.add_command(label="Reset to defaults", command=self.reset_layout)
        self.menu.add_cascade(label="Caption window size", menu=size_menu)

        self.window_var = tk.DoubleVar(value=float(self.settings["stream_window"]))
        win_menu = tk.Menu(self.menu, tearoff=0)
        for secs, note in ((0.8, "snappiest, least context"),
                           (1.2, ""),
                           (1.6, "recommended"),
                           (2.0, ""),
                           (2.4, ""),
                           (3.0, "most context, slowest")):
            label = f"{secs:.1f} s" + (f"   ({note})" if note else "")
            win_menu.add_radiobutton(label=label, value=secs, variable=self.window_var,
                                     command=lambda v=secs: self.set_audio_window(v))
        self.menu.add_cascade(label="Audio window (responsiveness)", menu=win_menu)

        self.model_var = tk.StringVar(value=str(self.settings.get("model") or ""))
        model_menu = tk.Menu(self.menu, tearoff=0)
        found = discover_models()
        if found:
            for i, (name, path) in enumerate(found):
                if name == "medium":
                    label = "Medium  (best quality - default)"
                elif name == "small":
                    label = "Small   (fastest)"
                else:
                    label = name
                model_menu.add_radiobutton(label=label, value=path, variable=self.model_var,
                                           command=lambda v=path: self.set_model(v))
        else:
            model_menu.add_command(label="(no local models found)", state="disabled")
        self.menu.add_cascade(label="Speech model", menu=model_menu)

        self.engine_var = tk.StringVar(value=str(self.settings.get("engine") or "direct"))
        eng_menu = tk.Menu(self.menu, tearoff=0)
        eng_menu.add_radiobutton(label="Whisper direct  (fast - default)", value="direct",
                                 variable=self.engine_var,
                                 command=lambda: self.set_engine("direct"))
        eng_menu.add_radiobutton(label="LLM polish on finals  (best English, ~8 s lag)",
                                 value="llm", variable=self.engine_var,
                                 command=lambda: self.set_engine("llm"))
        self.menu.add_cascade(label="Translation engine", menu=eng_menu)

        self.menu.add_separator()
        self.menu.add_command(label="Hide / show (F9)", command=self.toggle)
        self.menu.add_separator()
        self.menu.add_command(label="Exit", command=self.close)

    # ------------------------------------------------------- layout controls
    LIMITS = {"width_frac": (0.25, 1.0), "height_px": (90, 700),
              "font_size": (10, 96), "alpha": (0.20, 1.0)}

    def nudge(self, key, delta):
        lo, hi = self.LIMITS[key]
        value = max(lo, min(hi, self.settings[key] + delta))
        if key in ("height_px", "font_size"):
            value = int(round(value))
        self.settings[key] = value
        self.apply_layout()
        save_settings(self.settings)
        pretty = {"width_frac": "Width", "height_px": "Height",
                  "font_size": "Text size", "alpha": "Opacity"}[key]
        shown = f"{value:.0%}" if key == "width_frac" else (
                f"{value:.2f}" if key == "alpha" else f"{value}")
        self.post_status(f"{pretty}: {shown}   (right-click for more)")

    def reset_layout(self):
        for key in ("width_frac", "height_px", "font_size", "alpha"):
            self.settings[key] = DEFAULT_SETTINGS[key]
        self.apply_layout()
        save_settings(self.settings)
        self.post_status("Caption window reset to defaults")

    def apply_layout(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        width = max(200, int(sw * self.settings["width_frac"]))
        height = int(self.settings["height_px"])
        x = int((sw - width) / 2)
        y = sh - height - 55
        if self.visible:
            self.root.geometry(f"{width}x{height}+{x}+{y}")
        self._normal_geometry = f"{width}x{height}+{x}+{y}"
        self.label.configure(font=("Segoe UI Semibold", int(self.settings["font_size"])),
                             wraplength=max(120, width - 40))
        try:
            self.root.attributes("-alpha", self.settings["alpha"])
        except tk.TclError:
            pass

    # ------------------------------------------------------ decoding controls
    def set_audio_window(self, seconds: float):
        self.settings["stream_window"] = float(seconds)
        save_settings(self.settings)
        if self.segmenter is not None:
            self.segmenter.stream_window_s = float(seconds)
            self.post_status(f"Audio window: {seconds:.1f} s - takes effect on the next phrase")
        else:
            self.post_status(f"Audio window: {seconds:.1f} s (applies on restart)")

    def set_model(self, model_path: str):
        self.settings["model"] = model_path
        save_settings(self.settings)
        if self.transcriber is not None:
            self.transcriber.request_model(model_path)
        else:
            self.post_status("Model saved; applies on restart")

    def set_engine(self, engine: str):
        self.settings["engine"] = engine
        save_settings(self.settings)
        if self.transcriber is None:
            self.post_status("Engine saved; applies on restart")
            return
        if engine == "llm":
            mesh = make_mesh_translator()
            if mesh is None or not mesh.ok:
                self.engine_var.set("direct")
                self.settings["engine"] = "direct"
                save_settings(self.settings)
                self.post_status("LLM polish unavailable - no MESHDIRECT_URL/TOKEN configured")
                return
            self.transcriber.mesh = mesh
            self.post_status("LLM polish ON for final captions (adds several seconds)")
        else:
            self.transcriber.mesh = None
            self.post_status("Whisper direct translation (fast)")

    def _drag_start(self, e):
        self._drag_xy = (e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y())

    def _drag_move(self, e):
        dx, dy = self._drag_xy
        self.root.geometry(f"+{e.x_root-dx}+{e.y_root-dy}")

    def _popup(self, e):
        self.menu.tk_popup(e.x_root, e.y_root)

    def toggle(self):
        # A withdrawn Tk window cannot receive F9 to restore itself. Keep a tiny,
        # clickable tab on screen instead, so hiding never strands the app.
        if self.visible:
            self._normal_geometry = self.root.geometry()
            self.label.pack_forget()
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            self.root.geometry(f"340x38+{max(0, sw-360)}+{max(0, sh-95)}")
            self.status.configure(
                text="Captions hidden - double-click to restore",
                anchor="center",
                cursor="hand2",
            )
            self.status.bind("<Double-Button-1>", lambda _e: self.toggle())
        else:
            self.status.unbind("<Double-Button-1>")
            self.status.configure(anchor="w", cursor="")
            self.label.pack(fill="both", expand=True)
            self.root.geometry(getattr(self, "_normal_geometry", self.root.geometry()))
            self.root.attributes("-topmost", True)
        self.visible = not self.visible

    def close(self):
        self.root.destroy()

    def post_status(self, text: str):
        self.events.put(("status", text))

    def post_caption(self, text: str, final: bool, infer_s: float, e2e_s: float, detected: str):
        self.events.put(("caption", text, final, infer_s, e2e_s, detected))

    def poll(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == "status":
                    self.status.configure(text=ev[1])
                elif ev[0] == "caption":
                    _, text, final, infer_s, e2e_s, detected = ev
                    if final:
                        if not self.final_lines or self.final_lines[-1] != text:
                            self.final_lines.append(text)
                        self.current = ""
                    else:
                        self.current = text
                    lines = list(self.final_lines)[-1:]
                    if self.current and (not lines or lines[-1] != self.current):
                        lines.append(self.current)
                    self.label.configure(text="\n".join(lines[-2:]))
                    if self.visible:
                        source = language_display_name(detected)
                        self.status.configure(text=f"{source} -> English | model {infer_s:.2f}s | queue+model {e2e_s:.2f}s | F9 compact")
        except queue.Empty:
            pass
        try:
            self.root.after(40, self.poll)
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Chinese / Hindi / Russian system audio to English captions")
    ap.add_argument("--model", default="small", help="faster-whisper multilingual model (default: small)")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--compute-type", default=None, help="CTranslate2 compute type")
    ap.add_argument("--final-model", default=None,
                    help="optional second faster-whisper model used only for final utterances")
    ap.add_argument("--final-compute-type", default=None,
                    help="CTranslate2 compute type for --final-model (defaults to primary type)")
    ap.add_argument("--mode", choices=["direct", "two-stage", "hybrid"], default="direct",
                    help="direct = single-pass Whisper translation (default); hybrid/two-stage are optional comparison modes")
    ap.add_argument("--rms-threshold", type=float, default=DEFAULT_RMS_THRESHOLD)
    ap.add_argument("--stream-window", type=float, default=DEFAULT_STREAM_WINDOW_S)
    ap.add_argument("--stream-every", type=float, default=DEFAULT_STREAM_EMIT_S)
    ap.add_argument("--no-streaming", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.78)
    ap.add_argument("--font-size", type=int, default=31)
    args = ap.parse_args()

    # Saved UI settings supply anything the command line did not pin explicitly,
    # so the menu choices survive a restart.
    stored = load_settings()
    if "--stream-window" not in sys.argv:
        args.stream_window = float(stored.get("stream_window", args.stream_window))
    if "--model" not in sys.argv and stored.get("model"):
        args.model = stored["model"]

    device = "cpu" if args.cpu else "cuda"
    compute = args.compute_type or ("int8" if device == "cpu" else "int8_float32")

    audio_q: queue.Queue = queue.Queue()
    segment_q: queue.Queue = queue.Queue()
    ui = CaptionUI(args)
    capture = CaptureThread(audio_q, ui.post_status)
    segmenter = SegmenterThread(
        audio_q, segment_q, args.rms_threshold, args.stream_window,
        args.stream_every, streaming=not args.no_streaming,
    )
    transcriber = TranslateThread(
        segment_q, ui.post_caption, ui.post_status,
        args.model, device, compute, args.mode,
        args.final_model, args.final_compute_type,
    )
    ui.segmenter = segmenter
    ui.transcriber = transcriber
    if stored.get("engine") == "llm":
        mesh = make_mesh_translator()
        if mesh is not None and mesh.ok:
            transcriber.mesh = mesh
    capture.start()
    segmenter.start()
    transcriber.start()
    try:
        ui.run()
    finally:
        capture.running = False
        segmenter.running = False
        transcriber.running = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())