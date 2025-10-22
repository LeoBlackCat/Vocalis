"""
Vocalis Backend Server

FastAPI application entry point.
"""

# Fix OpenMP conflict before importing ML libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import logging
import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import configuration
from . import config

# Import services
from .services.transcription import WhisperTranscriber
from .services.llm import LLMClient
from .services.tts import TTSClient
from .services.vision import vision_service

# Import routes
from .routes.websocket import websocket_endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global service instances
transcription_service = None
llm_service = None
tts_service = None
rag_service = None
web_search_service = None
# Vision service is a singleton already initialized in its module

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the FastAPI application.
    """
    # Load configuration
    cfg = config.get_config()
    
    # Initialize services on startup
    logger.info("Initializing services...")

    global transcription_service, llm_service, tts_service, rag_service, web_search_service

    # Initialize transcription service
    transcription_service = WhisperTranscriber(
        model_size=cfg["whisper_model"],
        sample_rate=cfg["audio_sample_rate"]
    )

    # Initialize LLM service
    llm_service = LLMClient(
        api_endpoint=cfg["llm_api_endpoint"],
        log_dir=cfg["log_dir"]
    )

    # Initialize TTS service
    tts_service = TTSClient(
        engine=cfg["tts_engine"],
        model=cfg["tts_model"],
        voice=cfg["tts_voice"],
        voice_file=cfg["tts_voice_file"] if cfg["tts_voice_file"] else None,
        voice_text=cfg["tts_voice_text"] if cfg["tts_voice_text"] else None,
        output_format=cfg["tts_format"],
        sample_rate=cfg["tts_sample_rate"],
        elevenlabs_api_key=cfg["elevenlabs_api_key"] if cfg["elevenlabs_api_key"] else None,
        elevenlabs_voice_id=cfg["elevenlabs_voice_id"] if cfg["elevenlabs_voice_id"] else None
    )

    # Initialize RAG service if enabled
    if cfg["rag_enabled"]:
        try:
            logger.info("Initializing RAG service...")
            from .services.rag import RAGService

            rag_service = RAGService(
                chunks_path=cfg["rag_chunks_path"],
                embeddings_path=cfg["rag_embeddings_path"],
                docs_path=cfg["rag_docs_path"],
                top_k=cfg["rag_top_k"],
                dataset_name=cfg["rag_dataset_name"],
                strict_context=cfg["rag_strict_context"],
                min_score=cfg["rag_min_score"],
                log_dir=cfg["log_dir"]
            )
            logger.info(f"RAG service initialized with {cfg['rag_dataset_name']}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            rag_service = None

    # Initialize web search service if RAG enabled and web fallback enabled
    if cfg["rag_enabled"] and cfg["rag_web_fallback"]:
        try:
            logger.info("Initializing web search service...")
            from .services.web_search import WebSearchService

            web_search_service = WebSearchService(
                log_dir=cfg["log_dir"],
                enabled=True,
                search_prefix=cfg["web_search_prefix"]
            )
            logger.info("Web search service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize web search service: {e}")
            web_search_service = None

    # Initialize vision service (will download model if not cached)
    logger.info("Initializing vision service...")
    vision_service.initialize()

    logger.info("All services initialized successfully")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down services...")
    
    # No specific cleanup needed for these services,
    # but we could add resource release code here if needed (maybe in a future release lex 31/03/25)
    
    logger.info("Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Vocalis Backend",
    description="Speech-to-Speech AI Assistant Backend",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service dependency functions
def get_transcription_service():
    return transcription_service

def get_llm_service():
    return llm_service

def get_tts_service():
    return tts_service

# API routes
@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "ok", "message": "Vocalis backend is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "services": {
            "transcription": transcription_service is not None,
            "llm": llm_service is not None,
            "tts": tts_service is not None,
            "vision": vision_service.is_ready()
        },
        "config": {
            "whisper_model": config.WHISPER_MODEL,
            "tts_model": config.TTS_MODEL,
            "websocket_port": config.WEBSOCKET_PORT
        }
    }

@app.get("/config")
async def get_full_config():
    """Get full configuration."""
    if not all([transcription_service, llm_service, tts_service]) or not vision_service.is_ready():
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    return {
        "transcription": transcription_service.get_config(),
        "llm": llm_service.get_config(),
        "tts": tts_service.get_config(),
        "system": config.get_config()
    }

# WebSocket route
@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """WebSocket endpoint for bidirectional audio streaming."""
    # Prepare RAG and persona configurations
    cfg = config.get_config()

    rag_config = {
        "enabled": cfg["rag_enabled"],
        "web_fallback": cfg["rag_web_fallback"],
    }

    persona_config = {
        "enabled": cfg["persona_enabled"],
        "name": cfg["persona_name"],
        "style": cfg["persona_style"],
        "ask_user_name": cfg["persona_ask_user_name"],
    }

    await websocket_endpoint(
        websocket,
        transcription_service,
        llm_service,
        tts_service,
        rag_service=rag_service,
        web_search_service=web_search_service,
        rag_config=rag_config,
        persona_config=persona_config
    )

# Run server directly if executed as script
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=config.WEBSOCKET_HOST,
        port=config.WEBSOCKET_PORT,
        reload=True
    )
