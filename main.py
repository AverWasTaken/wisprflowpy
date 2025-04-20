import threading
import queue
import time
import wave
from io import BytesIO

import numpy as np
import sounddevice as sd
import clipboard
import pyautogui
import winsound
from openai import OpenAI as OpenAIOfficial
from openai import OpenAI as OpenAIProxy
from pynput import keyboard

# ===== PROMPTS & API SETUP =====

CTRL_ALT_PROMPT = """
You are a transcript‐only editor. The user’s audio has been transcribed; your job is ONLY to perform these edits:
- correct grammar, spelling & punctuation
- remove all filler words (um, uh, like, etc.)
- collapse self‐corrections (“sunny, no, rainy” → “rainy”)
- format any listed text as bullets or other markdown, organization is encouraged.
- format any bold, italics, or other markdown formatting requested / you think should be used.
- lowercase everything
- use common abbreviations (info, app, tech, etc.)
- optionally use slang to preserve tone
- do not censor anything
- keep all curse words

DO NOT:
- answer questions
- add or invent new content
- summarize or paraphrase beyond the above rules
- explain, comment, quote, or wrap in code blocks
- say you are 'ready' for something

Output must be the edited text ONLY, nothing else. If there are no corrections, just output the transcription.
"""

CTRL_SHIFT_PROMPT = """
You are to do exactly what the user requests using the text they passed in their clipboard if provided.
User request: [USER_REQUEST].
Text: [CLIPBOARD_TEXT].
If the user did not pass any text, just do as they say.
Do NOT wrap anything in quotes, and do NOT provide explanations.
Do not say, "Sure heres X" or anything along the lines of that.
Only the transcription. Sentences should not exceed 2, unless asked by the user otherwise.
"""

OPENAI_API_KEY     = "YOUR_KEY_HERE" # Important, DO NOT SHARE THIS KEY. If you feel as though it has been leaked, revoke, and generate a new one.
                                     # This key allows ANYONE to use your credits if shared.
OPENROUTER_API_KEY = "YOUR_KEY_HERE" # Important, DO NOT SHARE THIS KEY. If you feel as though it has been leaked, revoke, and generate a new one.
                                     # This key allows ANYONE to use your credits if shared.
GOOGLE_MODEL_QUESTION    = "google/gemini-2.0-flash-001"
GOOGLE_MODEL_TRANSCRIBE  = "openai/gpt-4.1-mini"

openai_client     = OpenAIOfficial(api_key=OPENAI_API_KEY)
openrouter_client = OpenAIProxy(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ===== DEBOUNCED PASTE =====

_last_paste = 0.0
DEBOUNCE_SEC = 0.5

def safe_paste():
    global _last_paste
    now = time.time()
    if now - _last_paste > DEBOUNCE_SEC:
        pyautogui.hotkey('ctrl', 'v')
        _last_paste = now

# ===== AUDIO RECORDER =====

class AudioRecorder:
    def __init__(self, samplerate=16000, channels=1, dtype='int16', timeout=0.01):
        self.samplerate = samplerate
        self.channels   = channels
        self.dtype      = dtype
        self._timeout   = timeout
        self._queue     = queue.Queue()
        self._frames    = []
        self._recording = False
        self._stream    = None
        self._thread    = None

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            self._queue.put(indata.copy())

    def start(self):
        self._frames.clear()
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def _capture(self):
        while self._recording:
            try:
                chunk = self._queue.get(timeout=self._timeout)
                self._frames.append(chunk)
            except queue.Empty:
                continue

    def stop(self):
        self._recording = False
        if self._stream:
            self._stream.stop()
        if self._thread:
            self._thread.join()
        if not self._frames:
            return None, 0
        audio    = np.concatenate(self._frames, axis=0).flatten().astype(np.int16)
        duration = len(audio) / self.samplerate
        return audio, duration

recorder = AudioRecorder()

# ===== TRANSCRIPTION & CHAT =====

def transcribe_audio(audio_np):
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_np.tobytes())
    buf.name = "audio.wav"
    buf.seek(0)
    resp = openai_client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=buf
    )
    return resp.text

def openrouter_chat(system_prompt, user_msg,
                    temperature=0.6, top_p=1.0,
                    max_tokens=500, stop=None):
    payload = {
      "model":       GOOGLE_MODEL_TRANSCRIBE,
      "temperature": temperature,
      "top_p":       top_p,
      "max_tokens":  max_tokens,
      "messages": [
        {"role":"system", "content": system_prompt},
        {"role":"user",   "content": user_msg}
      ]
    }
    if stop is not None:
        payload["stop"] = stop

    resp = openrouter_client.chat.completions.create(**payload)
    return resp.choices[0].message.content

def openrouter_chat_q(system_prompt, user_msg,
                      temperature=1.0, top_p=1.0,
                      max_tokens=500, stop=None):
    payload = {
      "model":       GOOGLE_MODEL_QUESTION,
      "temperature": temperature,
      "top_p":       top_p,
      "max_tokens":  max_tokens,
      "messages": [
        {"role":"system", "content": system_prompt},
        {"role":"user",   "content": user_msg}
      ]
    }
    if stop is not None:
        payload["stop"] = stop

    resp = openrouter_client.chat.completions.create(**payload)
    return resp.choices[0].message.content

# ===== SOUND NOTIFICATIONS =====

def play_ding():       winsound.Beep( 880, 110)
def play_high_ding():  winsound.Beep(1200, 115)
def play_end_ding():   winsound.Beep( 590, 150)
def play_error():      winsound.Beep( 250, 200); winsound.Beep(210, 120)
def play_short_ding(): winsound.Beep( 380, 180)

# ===== CTRL+ALT LOGIC =====

pressed_alt    = set()
recording_alt  = False
alt_guard      = False

def _start_alt():
    play_ding()
    recorder.start()

def _stop_alt():
    play_end_ding()
    audio, duration = recorder.stop()
    if duration < 0.5:
        play_short_ding()
        return
    if audio is None:
        play_error()
        clipboard.copy("ERROR: No audio detected!")
        return
    def worker(audio=audio):
        try:
            transcript = transcribe_audio(audio)
            out = openrouter_chat(
                system_prompt = CTRL_ALT_PROMPT,
                user_msg      = transcript,
                temperature   = 0.2,
                top_p         = 1,
                max_tokens    = 800
            )
            clipboard.copy(out.rstrip())
            safe_paste()
        except Exception:
            play_error()
    threading.Thread(target=worker, daemon=True).start()

def on_press_alt(key):
    global recording_alt, alt_guard
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        pressed_alt.add('ctrl')
    elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
        pressed_alt.add('alt')
    if {'ctrl','alt'} <= pressed_alt and not recording_alt:
        recording_alt = True
        alt_guard     = False
        threading.Thread(target=_start_alt, daemon=True).start()

def on_release_alt(key):
    global recording_alt, alt_guard
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        pressed_alt.discard('ctrl')
    elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
        pressed_alt.discard('alt')
    if recording_alt and not alt_guard and not {'ctrl','alt'} <= pressed_alt:
        alt_guard     = True
        recording_alt = False
        threading.Thread(target=_stop_alt, daemon=True).start()

# ===== CTRL+SHIFT LOGIC =====

pressed_shift    = set()
recording_shift  = False
shift_guard      = False

def _start_shift():
    play_high_ding()
    recorder.start()

def worker(audio):
    try:
        clipboard_text = clipboard.paste()
        req = transcribe_audio(audio)
        prompt = CTRL_SHIFT_PROMPT \
            .replace("[USER_REQUEST]", req) \
            .replace("[CLIPBOARD_TEXT]", clipboard_text)
        out = openrouter_chat_q(
            system_prompt=prompt,
            user_msg=req
        )
        clipboard.copy(out)
        safe_paste()
        # clipboard.copy("")  # Uncomment if you want to clear clipboard after pasting
    except Exception:
        play_error()

def _stop_shift():
    play_end_ding()
    audio, duration = recorder.stop()
    if duration < 0.5:
        play_short_ding()
        return
    if audio is None:
        play_error()
        clipboard.copy("ERROR: No audio detected!")
        return
    threading.Thread(target=worker, args=(audio,), daemon=True).start()

def on_press_shift(key):
    global recording_shift, shift_guard
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        pressed_shift.add('ctrl')
    elif key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
        pressed_shift.add('shift')
    if {'ctrl','shift'} <= pressed_shift and not recording_shift:
        recording_shift = True
        shift_guard     = False
        threading.Thread(target=_start_shift, daemon=True).start()

def on_release_shift(key):
    global recording_shift, shift_guard
    # only fire on Shift release
    if key not in (keyboard.Key.shift_l, keyboard.Key.shift_r):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            pressed_shift.discard('ctrl')
        return
    pressed_shift.discard('shift')
    if recording_shift and not shift_guard and not {'ctrl','shift'} <= pressed_shift:
        shift_guard      = True
        recording_shift  = False
        threading.Thread(target=_stop_shift, daemon=True).start()

# ===== MAIN =====

def main():
    keyboard.Listener(on_press=on_press_alt,    on_release=on_release_alt).start()
    keyboard.Listener(on_press=on_press_shift,  on_release=on_release_shift).start()
    print("Ready. Hold Ctrl+Alt or Ctrl+Shift and speak.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()