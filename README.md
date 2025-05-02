## Features

*   **Hotkey Activation:** Uses `Ctrl+Alt` for formatting and `Ctrl+Shift` for assistant tasks.
*   **Real-time Audio Recording:** Captures audio from the default input device while hotkeys are held.
*   **Deepgram STT:** Utilizes Deepgram's Nova-3 model for fast and accurate speech-to-text.
*   **LLM Integration via OpenRouter:** Leverages models available on OpenRouter for text processing (formatting or assistant tasks).
*   **Clipboard Awareness (Assistant Mode):** The assistant prompt includes text from the clipboard, allowing requests like "Summarize this:" or "Format this code:".
*   **Automatic Output:** Copies the processed text to the clipboard.
*   **Automatic Paste (Windows Only):** Simulates `Ctrl+V` to paste the result into the focused application.
*   **Status Indicator:** Displays a small, always-on-top badge showing the current status (ready, recording, processing).
*   **Configurable Models:** Easily change the LLM models used for formatting and assistant tasks.

## Requirements

*   **Python:** 3.8+ recommended.
*   **Operating System:** Primarily designed for **Windows** due to `winsound` for audio cues and `ctypes` for the automatic paste functionality. Recording and transcription *might* work on other OSes, but pasting and sound effects will likely fail.
*   **API Keys:**
    *   Deepgram API Key
    *   OpenRouter API Key
*   **Python Packages:** See `requirements.txt`. Install using `pip install -r requirements.txt`.
*   **Audio Input Device:** A working microphone configured as the default input device.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **API Keys:**
    Open the script (`your_script_name.py`) and replace the placeholder values for:
    *   `DG_API_KEY` (Your Deepgram API Key)
    *   `OPENROUTER_API_KEY` (Your OpenRouter API Key)

    You can get API keys respectivley from [Deepgram](https://deepgram.com/) & [OpenRouter](https://openrouter.ai/).

    Deepgram will give you **$200** in free credits upon signup. 

    You are required to pay for OpenRouter. 

    **OR**, you may use any model on the models page marked with the :free suffix (*these have usage limits*)

    <hr>

    **⚠️ Security Warning:** Storing API keys directly in the code is a security risk, especially if you share the code or commit it to public repositories. Consider using environment variables or a separate, untracked configuration file (e.g., `.env`) for better security in production or shared environments.

2.  **LLM Models:**
    Modify these variables in the script to use your desired models available through OpenRouter:
    *   `MODEL_FORMATTER`: The model identifier for the `Ctrl+Alt` formatting task.
    *   `MODEL_ASSISTANT`: The model identifier for the `Ctrl+Shift` assistant task.
    Verify that the chosen model identifiers are correct and accessible via your OpenRouter account.

3.  **Prompts (Optional):**
    You can adjust the system prompts (`CTRL_ALT_PROMPT`, `CTRL_SHIFT_PROMPT`) within the script to fine-tune the behavior of the LLMs.

## Usage

1.  **Run the script:**
    ```bash
    python your_script_name.py
    ```
    You should see "--- Starting Main Application ---" and the "Ready" message in the console. A small status badge will appear near the bottom-center of your screen.

2.  **Use Hotkeys:**
    *   **Format Text (`Ctrl+Alt`):** Hold `Ctrl+Alt`, speak clearly, and release the keys. The script will record, transcribe, send the transcript to the formatter LLM, copy the result, and paste it.
    *   **Assistant Task (`Ctrl+Shift`):**
        *   *(Optional)* Copy text to your clipboard that you want the assistant to reference.
        *   Hold `Ctrl+Shift`, speak your request (e.g., "summarize the text in the clipboard", "write python code to list files", "what's the capital of france?"), and release the keys. The script will record, transcribe, send the request (and clipboard context) to the assistant LLM, copy the response, and paste it.

3.  **Status Badge:**
    *   `ready`: Waiting for hotkey input.
    *   `recording`: Actively recording audio.
    *   `processing`: Sending data to APIs (Deepgram/OpenRouter) and processing the response.

4.  **Stop the script:** Press `Ctrl+C` in the console window where the script is running.

## Troubleshooting

*   **No Transcription/Errors:**
    *   Verify your `DG_API_KEY` and `OPENROUTER_API_KEY` are correct and have credits/access.
    *   Check that the `MODEL_FORMATTER` and `MODEL_ASSISTANT` identifiers are valid on OpenRouter.
    *   Ensure your microphone is working and selected as the default input device.
    *   Check the console output for specific error messages from Deepgram or OpenRouter.
*   **Pasting Issues (Windows):**
    *   The script needs focus on an application window that accepts text input for pasting to work.
    *   Permissions issues might occasionally interfere with `SendInput`.
*   **Sound Issues (Windows):** Ensure your system sounds are enabled if you don't hear the audio cues.
