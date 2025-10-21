# F5-TTS-MLX Setup Guide

This project now uses [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) for local text-to-speech generation on Apple Silicon.

## Installation

1. Install the required dependencies:
```bash
cd backend
pip install -r requirements.txt
```

This will install:
- `f5-tts-mlx` - The TTS engine
- `soundfile` - For audio file handling
- All other project dependencies

## Configuration

The TTS settings are configured in `backend/.env`:

```env
# TTS Configuration (F5-TTS-MLX)
TTS_MODEL=F5-TTS  # Options: F5-TTS, E2-TTS
TTS_VOICE_FILE=  # Optional: Path to reference audio file for voice cloning
TTS_VOICE_TEXT=  # Optional: Transcript of the reference audio
TTS_FORMAT=wav  # Output format (wav recommended)
TTS_SAMPLE_RATE=24000  # Sample rate in Hz (24000 recommended)
```

## Models

F5-TTS-MLX supports two models:

- **F5-TTS** (default) - High quality, natural sounding
- **E2-TTS** - Alternative model with different characteristics

## Voice Cloning (Optional)

You can clone any voice by providing a reference audio file:

1. Create a `voices` directory in the backend folder:
```bash
mkdir -p backend/voices
```

2. Add your reference audio file (WAV format recommended):
```bash
# Example: voices/my_voice.wav
```

3. Update your `.env` file:
```env
TTS_VOICE_FILE=voices/my_voice.wav
TTS_VOICE_TEXT=This is the exact text spoken in the reference audio file.
```

**Important**: The `TTS_VOICE_TEXT` must match exactly what is said in the audio file for best results.

## Default Voice

If you don't provide a reference audio file, F5-TTS will use its default voice, which is already quite natural and pleasant.

## Performance

- **First run**: The model will be downloaded automatically (may take a few minutes)
- **Subsequent runs**: Model is cached locally for fast loading
- **Generation speed**: Depends on text length, typically 1-3 seconds for short responses
- **Quality**: High quality, natural-sounding speech at 24kHz

## Requirements

- **Apple Silicon Mac** (M1, M2, M3, etc.)
- **macOS** with MLX support
- **Python 3.9+**

## Troubleshooting

### Model download fails
- Check your internet connection
- The model will be cached in `~/.cache/huggingface/`

### Audio quality issues
- Ensure `TTS_SAMPLE_RATE=24000` (recommended)
- Try the alternative model: `TTS_MODEL=E2-TTS`

### Voice cloning not working well
- Ensure reference audio is clear and high quality
- Make sure `TTS_VOICE_TEXT` exactly matches the audio
- Use 5-10 seconds of clean speech for best results

### Import errors
```bash
pip install --upgrade f5-tts-mlx soundfile
```

## Advantages over Cloud TTS

✅ **Privacy**: All processing happens locally  
✅ **No API costs**: Free to use  
✅ **No rate limits**: Generate as much as you need  
✅ **Voice cloning**: Clone any voice with a short sample  
✅ **Offline**: Works without internet (after initial model download)  
✅ **Fast**: Optimized for Apple Silicon with MLX  

## More Information

- GitHub: https://github.com/lucasnewman/f5-tts-mlx
- Original F5-TTS paper: https://arxiv.org/abs/2410.06885
