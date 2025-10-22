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

# TTS Configuration
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge-tts")  # "edge-tts" (default, fast, online), "f5-tts" (local, slower), or "elevenlabs" (online, high quality)
TTS_MODEL = os.getenv("TTS_MODEL", "F5-TTS")  # F5-TTS or E2-TTS (for f5-tts engine)
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")  # Voice for edge-tts
TTS_VOICE_FILE = os.getenv("TTS_VOICE_FILE", "")  # Path to reference audio for F5-TTS voice cloning
TTS_VOICE_TEXT = os.getenv("TTS_VOICE_TEXT", "")  # Transcript of reference audio for F5-TTS
TTS_FORMAT = os.getenv("TTS_FORMAT", "wav")
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", 24000))

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

# WebSocket Server Configuration
WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8000))

# Audio Processing
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", 0.5))
VAD_BUFFER_SIZE = int(os.getenv("VAD_BUFFER_SIZE", 30))
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 48000))

# RAG Configuration
RAG_ENABLED = os.getenv("RAG_ENABLED", "false").lower() == "true"
RAG_CHUNKS_PATH = os.getenv("RAG_CHUNKS_PATH", "")
RAG_EMBEDDINGS_PATH = os.getenv("RAG_EMBEDDINGS_PATH", "")
RAG_DOCS_PATH = os.getenv("RAG_DOCS_PATH", "")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 3))
RAG_DATASET_NAME = os.getenv("RAG_DATASET_NAME", "documents")
RAG_STRICT_CONTEXT = os.getenv("RAG_STRICT_CONTEXT", "true").lower() == "true"
RAG_WEB_FALLBACK = os.getenv("RAG_WEB_FALLBACK", "true").lower() == "true"
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", 0.7))

# Persona Configuration
PERSONA_ENABLED = os.getenv("PERSONA_ENABLED", "false").lower() == "true"
PERSONA_NAME = os.getenv("PERSONA_NAME", "John Doe")
PERSONA_STYLE = os.getenv("PERSONA_STYLE", "confident, direct, and enthusiastic")
PERSONA_ASK_USER_NAME = os.getenv("PERSONA_ASK_USER_NAME", "false").lower() == "true"
WEB_SEARCH_PREFIX = os.getenv("WEB_SEARCH_PREFIX", "John Doe")

# Logging Configuration
LOG_DIR = os.getenv("LOG_DIR", "backend/logs")

def get_config() -> Dict[str, Any]:
    """
    Returns all configuration settings as a dictionary.

    Returns:
        Dict[str, Any]: Dictionary containing all configuration settings
    """
    return {
        "llm_api_endpoint": LLM_API_ENDPOINT,
        "whisper_model": WHISPER_MODEL,
        "tts_engine": TTS_ENGINE,
        "tts_model": TTS_MODEL,
        "tts_voice": TTS_VOICE,
        "tts_voice_file": TTS_VOICE_FILE,
        "tts_voice_text": TTS_VOICE_TEXT,
        "tts_format": TTS_FORMAT,
        "tts_sample_rate": TTS_SAMPLE_RATE,
        "elevenlabs_api_key": ELEVENLABS_API_KEY,
        "elevenlabs_voice_id": ELEVENLABS_VOICE_ID,
        "websocket_host": WEBSOCKET_HOST,
        "websocket_port": WEBSOCKET_PORT,
        "vad_threshold": VAD_THRESHOLD,
        "vad_buffer_size": VAD_BUFFER_SIZE,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "rag_enabled": RAG_ENABLED,
        "rag_chunks_path": RAG_CHUNKS_PATH,
        "rag_embeddings_path": RAG_EMBEDDINGS_PATH,
        "rag_docs_path": RAG_DOCS_PATH,
        "rag_top_k": RAG_TOP_K,
        "rag_dataset_name": RAG_DATASET_NAME,
        "rag_strict_context": RAG_STRICT_CONTEXT,
        "rag_web_fallback": RAG_WEB_FALLBACK,
        "rag_min_score": RAG_MIN_SCORE,
        "persona_enabled": PERSONA_ENABLED,
        "persona_name": PERSONA_NAME,
        "persona_style": PERSONA_STYLE,
        "persona_ask_user_name": PERSONA_ASK_USER_NAME,
        "web_search_prefix": WEB_SEARCH_PREFIX,
        "log_dir": LOG_DIR,
    }
