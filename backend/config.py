"""
Vocalis Configuration Module

Loads and provides access to configuration settings from environment variables
and the .env file.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# API Endpoints
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions")

# Whisper Model Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")

# TTS Configuration (F5-TTS-MLX)
TTS_MODEL = os.getenv("TTS_MODEL", "F5-TTS")  # F5-TTS or E2-TTS
TTS_VOICE_FILE = os.getenv("TTS_VOICE_FILE", "")  # Path to reference audio for voice cloning
TTS_VOICE_TEXT = os.getenv("TTS_VOICE_TEXT", "")  # Transcript of reference audio
TTS_FORMAT = os.getenv("TTS_FORMAT", "wav")
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", 24000))

# WebSocket Server Configuration
WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8000))

# Audio Processing
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", 0.5))
VAD_BUFFER_SIZE = int(os.getenv("VAD_BUFFER_SIZE", 30))
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 48000))

def get_config() -> Dict[str, Any]:
    """
    Returns all configuration settings as a dictionary.
    
    Returns:
        Dict[str, Any]: Dictionary containing all configuration settings
    """
    return {
        "llm_api_endpoint": LLM_API_ENDPOINT,
        "whisper_model": WHISPER_MODEL,
        "tts_model": TTS_MODEL,
        "tts_voice_file": TTS_VOICE_FILE,
        "tts_voice_text": TTS_VOICE_TEXT,
        "tts_format": TTS_FORMAT,
        "tts_sample_rate": TTS_SAMPLE_RATE,
        "websocket_host": WEBSOCKET_HOST,
        "websocket_port": WEBSOCKET_PORT,
        "vad_threshold": VAD_THRESHOLD,
        "vad_buffer_size": VAD_BUFFER_SIZE,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
    }
