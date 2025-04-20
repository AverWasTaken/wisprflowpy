# Wispr Flow PY

An open-source, cost-effective alternative to WisprFlow designed for casual transcription needs. This tool uses OpenAI and OpenRouter APIs to provide high-quality transcription with convenient keyboard shortcuts.

## Features

- **Keyboard Shortcuts**: 
  - `Ctrl+Alt` - Record and transcribe speech with grammar correction
  - `Ctrl+Shift` - Record speech and process with user-specified instructions

- **Automatic Clipboard Integration**:
  - Results are automatically copied to clipboard and pasted
  - Works with existing clipboard content for Ctrl+Shift mode

- **Audio Feedback**:
  - Distinctive sounds for starting/stopping recording and errors
  - Clear indication of program status

## Cost Comparison

This tool is significantly more cost-effective than subscription-based alternatives:

- **WisprFlow Pro**: $12/month for unlimited words
- **WisprFlow Free**: Limited to 2,000 words with no AI assistant mode
- **This Tool**: Pay-per-use model with OpenRouter/OpenAI
  - GPT-4.1-mini: $0.40 per million input tokens
  - GPT-4.1-mini: $0.60 per million output tokens

For casual users, this can represent substantial savings, as you only pay for what you actually use. A million tokens is approximately 750,000 words, making the cost per transcribed word extremely low.

## Requirements

```
numpy
sounddevice
clipboard
pyautogui
openai
pynput
```

## Setup

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Obtain API keys:
   - OpenAI API key
   - OpenRouter API key
4. Update the API keys in the main.py file

## Usage

1. Run the script:
   ```
   python main.py
   ```
2. Press and hold `Ctrl+Alt` while speaking for basic transcription
3. Press and hold `Ctrl+Shift` while speaking for custom processing with clipboard content

## Model Selection

You can modify the models used in the script to suit your needs or budget:

- Current default: `gpt-4.1-mini` for transcription and `gemini-2.0-flash-001` for custom processing
- View all available models at [OpenRouter Models](https://openrouter.ai/models)

⚠️ **Note**: Some models may have different pricing. Check the OpenRouter pricing before changing models.

## Beta Status

This project is in beta stage and under active development. Issues and limitations may be present. Contributions are welcome!

- Feel free to make pull requests
- Report bugs or suggest features via issues
- Share your experiences and improvements

## License

[MIT License](LICENSE)

## Acknowledgements

- Inspired by WisprFlow but designed for cost-conscious casual users
- Uses OpenAI's transcription capabilities
- Leverages OpenRouter for model flexibility
