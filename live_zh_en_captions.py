#!/usr/bin/env python3
"""Live Chinese system-audio -> English captions for Windows.

Pipeline: Windows speaker loopback (SoundCard/WASAPI) -> rolling VAD segmenter ->
faster-whisper multilingual ASR/translation -> always-on-top Tk overlay.

Default mode uses Whisper's native task='translate', avoiding a second text MT pass.
A two-stage mode (Chinese ASR -> Helsinki OPUS-MT zh-en) is available for comparison.
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
            self.status_cb(f"Ready: system audio -> English ({self.device}/{self.compute_type}, load {load_s:.1f}s)")
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
                    language="zh",
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
                            audio, language="zh", task="translate", beam_size=1, best_of=1,
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
                if text:
                    self.ui_cb(text, final, infer_s, end_to_end_s)
                    log(f"caption final={final} lang={detected} infer={infer_s:.3f}s e2e={end_to_end_s:.3f}s text={text!r}")
            except Exception as exc:
                log(f"translate error: {exc}\n{traceback.format_exc()}")
                self.status_cb(f"Translate error: {exc}")


class CaptionUI:
    def __init__(self, args):
        self.root = tk.Tk()
        self.root.title("Chinese -> English Live Captions")
        self.root.configure(bg="#050505")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", args.alpha)
        self.root.overrideredirect(True)
        self.visible = True
        self.events: queue.Queue = queue.Queue()
        self.final_lines = deque(maxlen=2)
        self.current = ""

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        width = int(sw * 0.90)
        height = 190
        x = int((sw - width) / 2)
        y = sh - height - 55
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        self.status = tk.Label(
            self.root, text="Starting...", fg="#B7B7B7", bg="#050505",
            font=("Segoe UI", 10), anchor="w", padx=16, pady=5,
        )
        self.status.pack(fill="x")
        self.label = tk.Label(
            self.root, text="", fg="#FFFFFF", bg="#050505",
            font=("Segoe UI Semibold", args.font_size), justify="center",
            wraplength=width - 40, padx=20, pady=4,
        )
        self.label.pack(fill="both", expand=True)

        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.bind("<F9>", lambda _e: self.toggle())
        self.root.bind("<ButtonPress-1>", self._drag_start)
        self.root.bind("<B1-Motion>", self._drag_move)
        self.root.bind("<Button-3>", self._popup)
        self._drag_xy = (0, 0)
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="Hide / show (F9)", command=self.toggle)
        self.menu.add_separator()
        self.menu.add_command(label="Exit", command=self.close)
        self.root.after(40, self.poll)

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

    def post_caption(self, text: str, final: bool, infer_s: float, e2e_s: float):
        self.events.put(("caption", text, final, infer_s, e2e_s))

    def poll(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == "status":
                    self.status.configure(text=ev[1])
                elif ev[0] == "caption":
                    _, text, final, infer_s, e2e_s = ev
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
                        self.status.configure(text=f"Chinese -> English | model {infer_s:.2f}s | queue+model {e2e_s:.2f}s | F9 compact")
        except queue.Empty:
            pass
        try:
            self.root.after(40, self.poll)
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Chinese system audio to English captions")
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