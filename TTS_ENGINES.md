# TTS Engine Configuration

Vocalis supports two TTS engines:

## 1. edge-tts (Default) ⚡
- **Type**: Online service (Microsoft Edge TTS)
- **Speed**: Very fast (~1-2 seconds)
- **Quality**: Excellent, natural voices
- **Requirements**: Internet connection
- **Cost**: Free

### Configuration
Set in `.env` file or environment variables:
```bash
TTS_ENGINE=edge-tts
TTS_VOICE=en-US-AriaNeural  # Default voice
```

### Available Voices
Common English voices:
- `en-US-AriaNeural` - Female (default)
- `en-US-GuyNeural` - Male
- `en-US-JennyNeural` - Female
- `en-GB-SoniaNeural` - British Female
- `en-GB-RyanNeural` - British Male
- `en-AU-NatashaNeural` - Australian Female

List all available voices:
```bash
edge-tts --list-voices
```

## 2. F5-TTS-MLX 🎨
- **Type**: Local generation (Apple Silicon MLX)
- **Speed**: Slower (~10-20 seconds)
- **Quality**: High quality, customizable
- **Requirements**: Apple Silicon Mac, no internet needed
- **Cost**: Free (local)

### Configuration
Set in `.env` file or environment variables:
```bash
TTS_ENGINE=f5-tts
TTS_MODEL=F5-TTS  # or E2-TTS
```

### Voice Cloning (Optional)
For custom voice:
```bash
TTS_VOICE_FILE=/path/to/reference_audio.wav
TTS_VOICE_TEXT="Transcript of the reference audio"
```

## Switching Between Engines

### Method 1: Environment Variables
Edit `.env` file in the project root:
```bash
# For edge-tts (fast, online)
TTS_ENGINE=edge-tts
TTS_VOICE=en-US-AriaNeural

# OR for F5-TTS (local, slower)
TTS_ENGINE=f5-tts
TTS_MODEL=F5-TTS
```

Then restart the backend:
```bash
python -m backend.main
```

### Method 2: Direct Code Change
Edit `backend/config.py`, line 22:
```python
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge-tts")  # Change default here
```

## Performance Comparison

| Feature | edge-tts | F5-TTS-MLX |
|---------|----------|------------|
| Speed | ⚡⚡⚡ Very Fast | 🐌 Slow |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Voice Options | 100+ voices | Custom cloning |
| Offline | ❌ Requires internet | ✅ Works offline |
| Setup | Zero config | Requires MLX |
| Memory Usage | Low | High (~2GB) |

## Recommendations

**Use edge-tts if:**
- ✅ You have internet connection
- ✅ You want fast response times
- ✅ You're okay with Microsoft voices

**Use F5-TTS if:**
- ✅ You need offline capability
- ✅ You want custom voice cloning
- ✅ You have Apple Silicon Mac
- ✅ Response time is not critical
