Chinese -> English Live Captions
================================

What it does
------------
This listens to the Windows default SPEAKER OUTPUT and converts Mandarin Chinese
speech into English captions. It works with browser video, YouTube, Discord audio,
media players, and other sound played through the laptop. It uses WASAPI loopback,
not the microphone, so it does not take over or reconfigure Discord's recording device.

Start and persistence
---------------------
The Windows scheduled task LiveCaptionsZH starts the overlay at Scott's logon. It is
allowed on battery, starts when available, ignores duplicate launches, and retries a
real failure every minute (up to 999 retries). The desktop shortcut can also bring the
existing overlay forward.

IMPORTANT: the running app normally has TWO pythonw.exe processes. The first is the
virtual-environment launcher and the second is the Python worker/UI. They are one app
unit. Do not kill either as a supposed duplicate.

Current pipeline
----------------
1. SoundCard captures the default Windows speaker through WASAPI loopback at 48 kHz.
2. Rolling voice-activity detection forms low-latency speech windows.
3. faster-whisper Medium runs on the Quadro P3200 using CUDA int8_float32 and produces
   the rolling English partial captions. Medium is the default speech model from
   3 September 2026; choosing Small in the right-click menu switches the partial
   model live for the fastest response.
4. A local faster-whisper Medium model runs only at phrase boundaries to produce the
   more accurate final English revision.
5. Mandarin is forced as the source language. Repetition protection is enabled:
   repetition_penalty=1.12, no_repeat_ngram_size=3, max_new_tokens=64.
6. Medium loads in the background. If it cannot load, finals automatically use Small.
   If CUDA primary loading fails, the app falls back to CPU int8 direct translation.

No cloud service is required for live audio or caption text. MeshDirect Auto was tested
as a grammar editor but took 8.081 seconds for one sentence, so it is deliberately not
in the live path.

Controls
--------
F9          Toggle between the full overlay and a small bottom-right tab.
Double-click the small tab to restore the full overlay.
Escape      Exit.
Left drag   Move the overlay.
Right click Open the settings menu:
              Caption window size  - wider/narrower, taller/shorter, bigger/smaller
                                     text, more/less see-through, reset to defaults
              Audio window         - 0.8 to 3.0 s responsiveness vs context; applies
                                     from the next phrase, no restart needed
              Speech model         - Medium (best quality, default) or Small (fastest);
                                     swaps live at the next phrase boundary
              Translation engine   - Whisper direct (default) or LLM polish on finals
Ctrl+Left / Ctrl+Right   Narrower / wider caption window.
Ctrl+Up / Ctrl+Down      Shorter / taller caption window.
Ctrl+plus / Ctrl+minus   Bigger / smaller caption text.
All menu and keyboard changes are saved to settings.json and survive a restart.

Measured performance
--------------------
Same 8.424-second Chinese clip, faster-whisper Small:
  CPU int8 median:             4.281 seconds
  Quadro CUDA int8_float32:    1.309 seconds
  GPU speed-up over CPU:       3.27x

Same clip, current guarded models (measured while the production model was also live):
  Small partial model median:  0.592 seconds
  Medium final model median:   1.261 seconds
Medium preserved the phrase 'Artificial intelligence'; Small mistranslated it as
'Humanity', which is why the split pipeline uses Medium for final revisions.

Verified post-deployment live sample (68 captions):
  Partial median / p90 / max:  0.239 / 0.331 / 0.984 seconds
  Final median / p90 / max:    0.71 / 0.96 / 1.048 seconds
  Languages:                   zh
  Over 2 seconds / errors / repetition hits: 0 / 0 / 0

Diagnostics
-----------
Main program:       C:\Users\Scott\Apps\LiveChineseEnglishCaptions\live_zh_en_captions.py
Live log:           C:\Users\Scott\Apps\LiveChineseEnglishCaptions\live_captions.log
Benchmark data:     C:\Users\Scott\Apps\LiveChineseEnglishCaptions\benchmark_results.json
Deployment receipt: C:\Users\Scott\Apps\LiveChineseEnglishCaptions\diagnostics\20260903\dual_model_deployment_receipt.json
Current source SHA-256: 736C497711390425F8178DEE9EF279BC7D51185954197555BBFB71FC83FFABA7
Runtime settings recovered 3 September 2026
-------------------------------------------
The scheduled task no longer pins --model or --stream-window, so the right-click menu
choices are what actually load at logon. Medium is the default speech model and Small
is the selectable fast option.

Compute-type measurements, 3 September 2026 17:03 UK. Same 8.424 s clip and identical
decode settings to diagnostics\bench_dual_models_20260903.py, measured while the live
launcher/worker pair kept running:
  Medium CUDA int8_float32 median:  1.067 s   (primary now runs this)
  Medium CUDA int8 median:          1.102 s
  Small  CUDA int8 median:          0.492 s
Full data: diagnostics\bench_compute_types_20260903.json
