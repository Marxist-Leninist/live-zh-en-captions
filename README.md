# Live Chinese → English Captions

Real-time English subtitles for whatever is playing on your Windows speakers.
Captures system audio via WASAPI loopback, segments it with a rolling VAD, and
runs `faster-whisper` on the GPU to produce English text in an always-on-top
overlay. No microphone, no cloud round-trip, no browser extension — it captions
anything that makes sound, including DRM-protected video.

## Pipeline

```
Windows speaker loopback (SoundCard/WASAPI, 48 kHz)
  → resample to 16 kHz
  → rolling RMS + VAD segmenter (partial + final windows)
  → faster-whisper, task="translate"  (CTranslate2, CUDA, int8)
  → always-on-top Tk overlay
```

`--mode direct` (default) uses Whisper's own `translate` task, producing English
in a single model pass. `--mode two-stage` transcribes Chinese first and then
runs Helsinki OPUS-MT zh→en; `--mode hybrid` does that only on final segments.
Direct is the fastest and is what the measurements below use.

## Measured performance

Quadro P3200 (Pascal, 4 GB, SM 6.1), Whisper medium, `int8`, 1.6 s window:

| configuration | infer avg | infer max | end-to-end max |
|---|---|---|---|
| small, hybrid, int8_float32, 2.4 s window | 0.609 s | 1.671 s | 1.949 s |
| medium, direct, int8, 1.6 s window | 0.644 s | **1.162 s** | **1.257 s** |

Worst-case latency — the part a viewer actually perceives as stutter — improved
by about 35%, and translation quality improved substantially, for 35 ms of extra
average inference.

### Why `int8` and not `int8_float32`

The P3200 is Pascal (SM 6.1), which supports dp4a but has FP16 at 1/64 rate.
Pure `int8` beats `int8_float32` here and `float16` would be far slower. On a
Turing or newer card, re-measure before assuming the same ordering.

## Requirements

- Windows 10/11 with a default playback device
- Python 3.12
- `faster-whisper`, `soundcard`, `numpy`, `av`
- Optional, for `two-stage`/`hybrid`: `transformers`, `torch`,
  and a local copy of `Helsinki-NLP/opus-mt-zh-en`
- A CUDA GPU is optional; the app falls back to CPU `int8` automatically

## Models

Point `--model` at a plain directory containing `config.json`, `model.bin` and
`tokenizer.json` (plus `vocabulary.txt`), or pass a bare name like `small` /
`medium` to use the Hugging Face cache.

```
curl -L -o model.bin https://huggingface.co/Systran/faster-whisper-medium/resolve/main/model.bin
```

Note: `preprocessor_config.json` does not exist in the `Systran/faster-whisper-*`
repos and is not needed.

## Running

```
python live_zh_en_captions.py ^
  --model  C:\path\to\models\medium ^
  --mode   direct ^
  --stream-window 1.6 ^
  --stream-every  0.35 ^
  --compute-type  int8
```

`Live Chinese-English Captions.cmd` launches it minimised with `pythonw.exe`.

To have it start with Windows, register a scheduled task with an
`InteractiveToken` principal and a logon trigger. It **must** run in the
interactive session — a task or launcher running as `SYSTEM` will start the
process inside a job object that kills it on teardown.

## Operational notes

- **The app runs as a pair of processes** (launcher + worker). They are not
  duplicates. Killing one kills the captions.
- **One instance only.** A second instance cannot get the GPU and silently falls
  back to `device=cpu`. If you are benchmarking, grep `live_captions.log` for
  `device=cpu` before trusting any timing.
- Exclude the app directory from Defender; it does continuous audio and model
  I/O and real-time scanning shows up directly in inference latency.

## Options

| flag | default | meaning |
|---|---|---|
| `--model` | `small` | model name or path to a model directory |
| `--mode` | `hybrid` | `direct`, `two-stage`, or `hybrid` |
| `--stream-window` | 2.4 | seconds of audio per partial inference |
| `--stream-every` | — | seconds between partial emissions |
| `--compute-type` | `int8_float32` (CUDA) | CTranslate2 compute type |
| `--cpu` | off | force CPU |
| `--no-streaming` | off | finals only, no partials |
| `--rms-threshold` | — | silence gate |
| `--font-size` | 31 | overlay font size |
| `--alpha` | 0.78 | overlay opacity |

## Licence

MIT — see `LICENSE`.
