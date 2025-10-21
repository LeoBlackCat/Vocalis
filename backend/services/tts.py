"""
Text-to-Speech Service

Handles text-to-speech generation using F5-TTS-MLX.
"""

import json
import logging
import io
import time
import base64
import asyncio
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Optional, BinaryIO, Generator, AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSClient:
    """
    Client for F5-TTS-MLX text-to-speech generation.
    
    This class handles local TTS generation using F5-TTS with MLX on Apple Silicon.
    """
    
    def __init__(
        self,
        model: str = "F5-TTS",
        voice_file: Optional[str] = None,
        voice_text: Optional[str] = None,
        output_format: str = "wav",
        sample_rate: int = 24000,
        chunk_size: int = 4096
    ):
        """
        Initialize the F5-TTS-MLX client.
        
        Args:
            model: TTS model name to use (F5-TTS or E2-TTS)
            voice_file: Path to reference audio file for voice cloning (optional)
            voice_text: Transcript of the reference audio (optional)
            output_format: Output audio format (wav, mp3)
            sample_rate: Audio sample rate (default: 24000)
            chunk_size: Size of audio chunks to stream in bytes
        """
        self.model = model
        self.voice_file = voice_file
        self.voice_text = voice_text
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # State tracking
        self.is_processing = False
        self.last_processing_time = 0
        self._tts_model = None
        
        logger.info(f"Initialized F5-TTS-MLX Client with model={model}, sample_rate={sample_rate}")
    
    def _load_model(self):
        """Lazy load the F5-TTS model."""
        if self._tts_model is None:
            try:
                from f5_tts_mlx.generate import generate
                self._generate_fn = generate
                logger.info("F5-TTS-MLX model loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import f5_tts_mlx: {e}")
                raise ImportError(
                    "f5-tts-mlx not installed. Install with: pip install f5-tts-mlx"
                )
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech audio using F5-TTS-MLX.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        self.is_processing = True
        start_time = time.time()
        
        try:
            # Load model if not already loaded
            self._load_model()
            
            logger.info(f"Generating TTS for {len(text)} characters of text")
            
            # Map model name to full model path
            model_map = {
                "F5-TTS": "lucasnewman/f5-tts-mlx",
                "E2-TTS": "lucasnewman/e2-tts-mlx"
            }
            model_name = model_map.get(self.model, "lucasnewman/f5-tts-mlx")
            
            # Generate audio using F5-TTS
            # If voice cloning is configured, use reference audio
            # Check for non-empty strings and strip whitespace
            has_voice_file = self.voice_file and self.voice_file.strip()
            has_voice_text = self.voice_text and self.voice_text.strip()
            
            if has_voice_file and has_voice_text:
                logger.info(f"Using voice cloning with reference: {self.voice_file}")
                result = self._generate_fn(
                    generation_text=text,
                    ref_audio_path=self.voice_file.strip(),
                    ref_audio_text=self.voice_text.strip(),
                    model_name=model_name
                )
            else:
                # Use default voice
                logger.info("Using default voice (no voice cloning)")
                result = self._generate_fn(
                    generation_text=text,
                    model_name=model_name
                )
            
            # The generate function returns (audio_array, sample_rate)
            if isinstance(result, tuple):
                audio_array, actual_sample_rate = result
            else:
                audio_array = result
                actual_sample_rate = self.sample_rate
            
            # Convert numpy array to audio bytes
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_array, actual_sample_rate, format='WAV')
            audio_data = audio_buffer.getvalue()
            
            # Calculate processing time
            self.last_processing_time = time.time() - start_time
            
            logger.info(f"Generated TTS audio after {self.last_processing_time:.2f}s, "
                       f"size: {len(audio_data)} bytes")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            raise
        finally:
            self.is_processing = False
    
    def stream_text_to_speech(self, text: str) -> Generator[bytes, None, None]:
        """
        Stream audio data by generating complete audio and chunking it.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Chunks of audio data
        """
        self.is_processing = True
        start_time = time.time()
        
        try:
            logger.info(f"Generating streaming TTS for {len(text)} characters of text")
            
            # Generate complete audio
            audio_data = self.text_to_speech(text)
            
            # Split into chunks and yield
            total_chunks = (len(audio_data) + self.chunk_size - 1) // self.chunk_size
            
            for i in range(total_chunks):
                start_idx = i * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, len(audio_data))
                yield audio_data[start_idx:end_idx]
            
            # Calculate processing time
            self.last_processing_time = time.time() - start_time
            logger.info(f"Completed TTS streaming after {self.last_processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            raise
        finally:
            self.is_processing = False
    
    async def async_text_to_speech(self, text: str) -> bytes:
        """
        Asynchronously generate audio data from the TTS API.
        
        This method provides asynchronous TTS capability by running
        the synchronous method in a thread.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Complete audio data as bytes
        """
        self.is_processing = True
        
        try:
            # Get complete audio data
            audio_data = await asyncio.to_thread(self.text_to_speech, text)
            return audio_data
        except Exception as e:
            logger.error(f"Async TTS error: {e}")
            raise
        finally:
            self.is_processing = False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dict containing the current configuration
        """
        return {
            "model": self.model,
            "voice_file": self.voice_file,
            "voice_text": self.voice_text,
            "output_format": self.output_format,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "is_processing": self.is_processing,
            "last_processing_time": self.last_processing_time
        }
