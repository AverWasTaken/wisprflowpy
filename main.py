# -*- coding: utf-8 -*-
import ctypes, json, queue, threading, time, traceback, wave, io
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import clipboard
import sounddevice as sd
import winsound
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from openai import OpenAI as OpenAIProxy 
from pynput import keyboard
import tkinter as tk

# ───────────────────── CONFIG ────────────────────── #

DG_API_KEY         = "change_me" # WARNING - DO NOT SHARE THESE KEYS PUBLICLY!
OPENROUTER_API_KEY = "change_me" # WARNING - DO NOT SHARE THESE KEYS PUBLICLY!


MODEL_FORMATTER  = "google/gemini-flash-1.5-8b" # Model for Ctrl+Alt
MODEL_ASSISTANT  = "google/gemini-2.0-flash-001" # Model for Ctrl+Shift assistant

print("--- Initializing Clients ---")
try:
    dg_client   = DeepgramClient(DG_API_KEY)
    print("Deepgram client initialized.")
    openrouter  = OpenAIProxy(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    print("OpenRouter client initialized.")
except Exception as e:
    print(f"!!! Error initializing clients: {e}")
    traceback.print_exc()
    exit(1)

EXECUTOR    = ThreadPoolExecutor(max_workers=6)
DEBOUNCE_MS = 500

# ───────────────────── PROMPTS ───────────────────── #

# Prompt for Ctrl+Alt (Formatting)
CTRL_ALT_PROMPT = """
role: transcript-only editor.
task: edit the provided transcript using only the rules below. output only the edited text.
rules:

organize and format things nicely as you see fit.
correct grammar, spelling, punctuation.
remove filler words (um, uh, like, etc.).
collapse self-corrections (e.g., "red, no, blue car" -> "blue car").
make everything lowercase.
use common abbreviations (idk, lol, tech, asap, etc.).
keep slang and curse words.
output:

if edits are made, output only the edited lowercase text.
if no edits are needed, output the original transcript unchanged.
""".strip()

# Prompt for Ctrl+Shift (Assistant) - Added clipboard placeholder
CTRL_SHIFT_PROMPT = """
You are a concise AI assistant. Follow the user's request based on their transcribed audio.
Use the provided clipboard text if the user's request refers to it.

Clipboard Text:
{clipboard_content}

User Request (from audio):
{user_request}

Instructions:
- If the clipboard text is "(empty)" or the user request doesn't seem to relate to it, just fulfill the user's audio request directly.
- Be direct and answer concisely.
- Do not add explanations unless asked.
- Keep output brief, ideally one or two sentences, unless the request requires more detail.
- Do not wrap your response in quotes or code blocks, unless outputting code.
- You may ignore the concise instruction if instructed to output long text.
""".strip()


# ───────────────────── UI BADGE ───────────────────── #

class StatusBadge:
    _COL = {"ready": "#222", "recording": "#b40000", "processing": "#444"}

    def __init__(self):
        self.tk  = tk.Tk()
        self.tk.overrideredirect(True)
        self.tk.attributes("-topmost", True)
        self.lbl = tk.Label(self.tk, text="ready", fg="white",
                            font=("Segoe UI", 10, "bold"),
                            bg=self._COL["ready"], padx=10, pady=2)
        self.lbl.pack()
        self._place()
    def _place(self):
        self.tk.update_idletasks()
        w, h = self.tk.winfo_width(), self.tk.winfo_height()
        sw, sh = self.tk.winfo_screenwidth(), self.tk.winfo_screenheight()
        self.tk.geometry(f"+{(sw-w)//2}+{sh-h-40}")
    def set(self, txt):
        if txt == "ready" or self._COL.get(txt) == self._COL["processing"]:
             print(f"UI: Status -> {txt}")
        def _u():
            self.lbl.config(text=txt, bg=self._COL.get(txt, "#222")); self._place()
        self.tk.after(1, _u)

badge = StatusBadge()

# ───────────────────── SOUND FX ───────────────────── #

beep = winsound.Beep
play_ding   = lambda: beep( 880,110) # Sound for Ctrl+Alt start
play_hi     = lambda: beep(1200,115) # Sound for Ctrl+Shift start
play_end    = lambda: beep( 590,150)
play_err    = lambda: (print("SFX: Error sound played"), beep(250,200), beep(210,120))
play_short  = lambda: (print("SFX: Short recording sound played"), beep( 380,180))

# ───────────────────── SMART PASTE (Win) ──────────── #

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def _get_extra_info():
    try:
        return ctypes.cast(ctypes.windll.user32.GetMessageExtraInfo(), PUL)
    except AttributeError:
        print("!!! PASTE Warning: Could not call GetMessageExtraInfo. Using default.")
        return PUL()

def send_ctrl_v():
    try:
        user32 = ctypes.windll.user32
        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002
        VK_CONTROL = 0x11
        VK_V = 0x56
        extra = _get_extra_info()
        ctrl_down = Input(type=INPUT_KEYBOARD, ii=Input_I(ki=KeyBdInput(wVk=VK_CONTROL, wScan=0, dwFlags=0, time=0, dwExtraInfo=extra)))
        v_down = Input(type=INPUT_KEYBOARD, ii=Input_I(ki=KeyBdInput(wVk=VK_V, wScan=0, dwFlags=0, time=0, dwExtraInfo=extra)))
        v_up = Input(type=INPUT_KEYBOARD, ii=Input_I(ki=KeyBdInput(wVk=VK_V, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=extra)))
        ctrl_up = Input(type=INPUT_KEYBOARD, ii=Input_I(ki=KeyBdInput(wVk=VK_CONTROL, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=extra)))
        inputs = (Input * 4)(ctrl_down, v_down, v_up, ctrl_up)
        user32.SendInput(4, ctypes.byref(inputs), ctypes.sizeof(Input))
    except Exception as e:
        print(f"!!! PASTE Error: {e}")
        traceback.print_exc()

_last_paste = 0.0
def safe_paste():
    global _last_paste
    now = time.time()
    if now - _last_paste > DEBOUNCE_MS / 1000:
        print("PASTE: Pasting content.")
        send_ctrl_v()
        _last_paste = now
    else:
        print("PASTE: Debounced.")

# ───────────────────── AUDIO RECORDER ─────────────── #

class Recorder:
    def __init__(self, rate=16000):
        self.rate = rate; self.q = queue.Queue(); self.frames=[]
        self.rec = threading.Event()
        self.stream = None
        print(f"Recorder initialized with rate {rate}Hz.")
    def _cb(self, data, frame_count, time_info, status):
        if status:
            print(f"REC: Status warning in callback: {status}")
        if self.rec.is_set():
            self.q.put_nowait(data.copy())
    def start(self):
        print("REC: Start recording...")
        self.frames.clear(); self.q = queue.Queue(); self.rec.set()
        try:
            self.stream = sd.InputStream(samplerate=self.rate, channels=1,
                                         dtype='int16', callback=self._cb)
            self.stream.start()
            threading.Thread(target=self._pump, daemon=True).start()
        except Exception as e:
            print(f"!!! REC Error starting stream: {e}")
            traceback.print_exc()
            badge.set("ready")
            play_err()
            self.rec.clear()
    def _pump(self):
        while self.rec.is_set():
            try:
                frame = self.q.get(timeout=0.1)
                self.frames.append(frame)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"!!! REC Error in pump: {e}")
    def stop(self)->Tuple[Optional[np.ndarray],float]:
        print("REC: Stop recording...")
        self.rec.clear()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"!!! REC Error stopping stream: {e}")
                traceback.print_exc()
            self.stream = None
        else:
            print("REC: Stop called but stream was not active.")

        if not self.frames:
            print("REC: No frames recorded.")
            return None, 0.0

        try:
            audio = np.concatenate(self.frames, axis=0).astype(np.int16)
            duration = len(audio) / self.rate
            print(f"REC: Recording stopped. Duration: {duration:.2f}s")
            return audio, duration
        except ValueError as e:
             print(f"!!! REC Error concatenating frames: {e}")
             print(f"!!! REC Frame shapes: {[f.shape for f in self.frames]}")
             traceback.print_exc()
             return None, 0.0
        except Exception as e:
            print(f"!!! REC Error processing frames: {e}")
            traceback.print_exc()
            return None, 0.0

rec = Recorder()

# ───────────────────── DEEPGRAM STT ───────────────────── #

def transcribe(audio: np.ndarray)->str:
    print(f"STT: Transcribing audio (size: {audio.size} bytes)...")
    if audio.size == 0:
        print("STT: Audio data is empty, cannot transcribe.")
        return ""

    pay: FileSource = {"buffer": audio.tobytes()}
    # Options suitable for getting a raw transcript for further processing
    opt = PrerecordedOptions(
        model="nova-3",
        language="en",
        smart_format=True, # Keep smart format for basic structure
        punctuate=True,    # Keep punctuation
        paragraphs=False,  # Let LLM handle paragraphs
        filler_words=True, # Keep filler words for LLM context if needed
        channels=1,
        sample_rate=rec.rate,
        encoding="linear16"
    )

    transcript = ""
    for attempt in range(2):
        try:
            print(f"STT: Sending request to Deepgram (attempt {attempt+1})...")
            start_time = time.time()
            rsp = dg_client.listen.rest.v("1").transcribe_file(pay, opt, timeout=20.0)
            end_time = time.time()
            print(f"STT: Deepgram response received in {end_time - start_time:.2f}s.")

            if rsp and rsp.results and rsp.results.channels and rsp.results.channels[0].alternatives:
                transcript = rsp.results.channels[0].alternatives[0].transcript
                print(f"STT: Transcription successful: '{transcript[:80]}...'")
                return transcript
            else:
                if rsp and rsp.results and rsp.results.channels and not rsp.results.channels[0].alternatives:
                     print("STT: Transcription result is empty (no alternatives found).")
                     return ""
                print("!!! STT Error: Deepgram response structure unexpected or empty.")
                raise ValueError("Invalid response structure from Deepgram")

        except Exception as e:
            print(f"!!! STT Error (Attempt {attempt+1}): {e}")
            if attempt == 1:
                 traceback.print_exc()
            if attempt == 0:
                print("STT: Retrying after delay...")
                time.sleep(0.5)
            else:
                print("!!! STT Failed after retries.")

    print("STT: Transcription failed after retries, returning empty string.")
    return ""

# ───────────────────── OPENROUTER CHAT ─────────────── #

def chat(model, system, user, temp=1, top_p=1, max_tokens=1000):
    if model == "change_me":
        print(f"!!! LLM Error: Assistant model is set to '{model}'. Please update MODEL_ASSISTANT.")
        raise ValueError(f"Assistant model not configured: {model}")

    print(f"LLM: Calling model '{model}'...")
    try:
        start_time = time.time()
        r = openrouter.chat.completions.create(
            model=model, temperature=temp, top_p=top_p, max_tokens=max_tokens,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        end_time = time.time()
        response_text = r.choices[0].message.content
        print(f"LLM: Response received in {end_time - start_time:.2f}s.")
        return response_text
    except Exception as e:
        print(f"!!! LLM Error calling model {model}: {e}")
        traceback.print_exc()
        raise e

# ───────────────────── WORKERS ─────────────────────── #

# Worker for Ctrl+Alt (Formatting)
def alt_task(audio):
    print("--- Starting Alt Task (Formatter) ---")
    try:
        badge.set("processing")
        transcript = transcribe(audio)
        if not transcript:
            print("ALT_TASK: Transcription failed or empty, aborting.")
            play_err()
            return

        print("ALT_TASK: Calling Formatter LLM...")
        # Use MODEL_FORMATTER and specific parameters for formatting
        cleaned_transcript = chat(MODEL_FORMATTER, CTRL_ALT_PROMPT, transcript,
                                  temp=0.5, top_p=1, max_tokens=1000).rstrip()

        if not cleaned_transcript:
             print("ALT_TASK: Formatting returned empty result, aborting paste.")
             play_err()
             return

        print(f"ALT_TASK: Formatting complete.")
        clipboard.copy(cleaned_transcript)
        print("ALT_TASK: Copied formatted text to clipboard.")
        safe_paste()
    except Exception as e:
        print(f"!!! ALT_TASK Error: {e}")
        play_err()
    finally:
        print("--- Finished Alt Task ---")
        badge.set("ready")

# Worker for Ctrl+Shift (Assistant)
def shift_task(audio):
    print("--- Starting Shift Task (Assistant) ---")
    try:
        badge.set("processing")
        request_transcript = transcribe(audio)
        if not request_transcript:
            print("SHIFT_TASK: Transcription failed or empty, aborting.")
            play_err()
            return

        # Get clipboard content
        print("SHIFT_TASK: Getting clipboard content...")
        try:
            cb_content = clipboard.paste() or "(empty)"
            print(f"SHIFT_TASK: Clipboard content: '{cb_content[:100]}...'")
        except Exception as e:
            print(f"!!! SHIFT_TASK: Error getting clipboard content: {e}")
            cb_content = "(error reading clipboard)"

        # Format the system prompt with clipboard content and user request
        formatted_system_prompt = CTRL_SHIFT_PROMPT.format(
            clipboard_content=cb_content,
            user_request=request_transcript # Include user request here for context within the prompt
        )

        print("SHIFT_TASK: Calling Assistant LLM...")
        # Pass the formatted system prompt. The user message is now the raw request transcript.
        assistant_response = chat(MODEL_ASSISTANT, formatted_system_prompt, request_transcript,
                                  temp=0.7, top_p=1, max_tokens=1000).strip()

        if not assistant_response:
             print("SHIFT_TASK: Assistant returned empty result, aborting paste.")
             play_err()
             return

        print(f"SHIFT_TASK: Assistant response complete.")
        clipboard.copy(assistant_response)
        print("SHIFT_TASK: Copied assistant response to clipboard.")
        safe_paste()
    except Exception as e:
        print(f"!!! SHIFT_TASK Error: {e}")
        if isinstance(e, ValueError) and "Assistant model not configured" in str(e):
            pass
        else:
            play_err()
    finally:
        print("--- Finished Shift Task ---")
        badge.set("ready")

# ───────────────────── HOT‑KEY FSM ─────────────────── #

class HotKeys:
    def __init__(self):
        self.ctrl=self.alt=self.shift=False; self.mode=None
        self.listener = keyboard.Listener(on_press=self.p,on_release=self.r)
        self.listener.start()
        print("HotKey listener started.")

    def p(self,k):
        key_pressed = None
        if k in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):   self.ctrl=True; key_pressed="Ctrl"
        elif k in (keyboard.Key.alt_l, keyboard.Key.alt_r):    self.alt=True; key_pressed="Alt"
        elif k in (keyboard.Key.shift_l, keyboard.Key.shift_r): self.shift=True; key_pressed="Shift"

        if key_pressed:
            self._start()

    def r(self,k):
        key_released = None
        if k in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):   self.ctrl=False; key_released="Ctrl"
        elif k in (keyboard.Key.alt_l, keyboard.Key.alt_r):    self.alt=False; key_released="Alt"
        elif k in (keyboard.Key.shift_l, keyboard.Key.shift_r): self.shift=False; key_released="Shift"

        if key_released:
            self._stop(k)

    def _start(self):
        # Check for Ctrl+Alt (Formatter)
        if self.ctrl and self.alt and not self.shift and self.mode is None:
            self.mode="alt"
            print("HOTKEY: Ctrl+Alt detected, starting recording (Formatter).")
            play_ding() # Specific sound for formatter
            badge.set("recording"); rec.start()
        # Check for Ctrl+Shift (Assistant)
        elif self.ctrl and self.shift and not self.alt and self.mode is None:
            self.mode="shift"
            print("HOTKEY: Ctrl+Shift detected, starting recording (Assistant).")
            play_hi() # Specific sound for assistant
            badge.set("recording"); rec.start()

    def _stop(self, released_key):
        # Trigger alt_task if Ctrl or Alt is released while in 'alt' mode
        if self.mode == "alt" and released_key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.alt_l, keyboard.Key.alt_r):
             if not self.ctrl or not self.alt:
                self._finish(alt_task) # Call formatter task
        # Trigger shift_task if Ctrl or Shift is released while in 'shift' mode
        elif self.mode == "shift" and released_key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.shift_l, keyboard.Key.shift_r):
            if not self.ctrl or not self.shift:
                self._finish(shift_task) # Call assistant task

    def _finish(self, fn):
        active_mode = self.mode
        self.mode = None # Reset mode immediately
        print(f"FINISH: Finishing task for mode '{active_mode}' -> {fn.__name__}")
        play_end()
        audio, dur = rec.stop()

        if audio is None or dur < 0.5:
            print(f"FINISH: Recording too short ({dur:.2f}s) or failed. Aborting task.")
            play_short()
            badge.set("ready")
            return

        print(f"FINISH: Submitting task '{fn.__name__}' to executor.")
        EXECUTOR.submit(fn, audio)

# ───────────────────── MAIN ──────────────────────── #

def main():
    print("--- Starting Main Application ---")
    if MODEL_ASSISTANT == "change_me":
        print("\n" + "*"*60)
        print("!!! WARNING: Assistant model (MODEL_ASSISTANT) is not configured. !!!")
        print("!!!          Ctrl+Shift functionality will raise an error.       !!!")
        print("!!!          Please update the 'change_me' value.                !!!")
        print("*"*60 + "\n")

    hotkey_manager = HotKeys()
    print("\n" + "="*40)
    print("Ready. Hold Ctrl+Alt (Format) or Ctrl+Shift (Assistant) and speak.")
    print("Press Ctrl+C in the console to exit.")
    print("="*40 + "\n")
    try:
        badge.tk.mainloop()
    except KeyboardInterrupt:
        print("\n--- KeyboardInterrupt detected, shutting down. ---")
    finally:
        print("--- Cleaning up ---")
        EXECUTOR.shutdown(wait=False)
        if hotkey_manager and hotkey_manager.listener:
             hotkey_manager.listener.stop()
        print("--- Exited ---")

if __name__ == "__main__":
    main()
