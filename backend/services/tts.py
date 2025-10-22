"""
Text-to-Speech Service

Handles text-to-speech generation using multiple TTS engines:
- edge-tts (online, fast, default)
- F5-TTS-MLX (local, slower, higher quality)
- ElevenLabs (online, high quality, requires API key)
"""

import json
import logging
import io
import time
import base64
import asyncio
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, BinaryIO, Generator, AsyncGenerator, Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSClient:
    """
    Client for text-to-speech generation supporting multiple engines.

    Supports:
    - edge-tts: Online TTS service (fast, default)
    - f5-tts: Local F5-TTS-MLX (slower, higher quality)
    - elevenlabs: ElevenLabs API (online, high quality)
    """

    def __init__(
        self,
        engine: Literal["edge-tts", "f5-tts", "elevenlabs"] = "edge-tts",
        model: str = "F5-TTS",
        voice: str = "en-US-AriaNeural",  # For edge-tts
        voice_file: Optional[str] = None,  # For F5-TTS voice cloning
        voice_text: Optional[str] = None,  # For F5-TTS voice cloning
        output_format: str = "wav",
        sample_rate: int = 24000,
        chunk_size: int = 4096,
        elevenlabs_api_key: Optional[str] = None,  # For ElevenLabs
        elevenlabs_voice_id: Optional[str] = None  # For ElevenLabs
    ):
        """
        Initialize the TTS client.

        Args:
            engine: TTS engine to use ("edge-tts", "f5-tts", or "elevenlabs")
            model: TTS model name for F5-TTS (F5-TTS or E2-TTS)
            voice: Voice name for edge-tts (e.g., "en-US-AriaNeural")
            voice_file: Path to reference audio file for F5-TTS voice cloning (optional)
            voice_text: Transcript of the reference audio for F5-TTS (optional)
            output_format: Output audio format (wav, mp3)
            sample_rate: Audio sample rate (default: 24000)
            chunk_size: Size of audio chunks to stream in bytes
            elevenlabs_api_key: ElevenLabs API key (required for elevenlabs engine)
            elevenlabs_voice_id: ElevenLabs voice ID (required for elevenlabs engine)
        """
        self.engine = engine
        self.model = model
        self.voice = voice
        self.voice_file = voice_file
        self.voice_text = voice_text
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_voice_id = elevenlabs_voice_id

        # State tracking
        self.is_processing = False
        self.last_processing_time = 0
        self._generate_fn = None  # For F5-TTS lazy loading

        # Validate ElevenLabs configuration
        if engine == "elevenlabs":
            if not elevenlabs_api_key or not elevenlabs_voice_id:
                raise ValueError("ElevenLabs engine requires both api_key and voice_id")

        logger.info(f"Initialized TTS Client with engine={engine}, voice/model={voice if engine == 'edge-tts' else (elevenlabs_voice_id if engine == 'elevenlabs' else model)}, sample_rate={sample_rate}")
    
    def _load_model(self):
        """Lazy load the F5-TTS model."""
        if self._generate_fn is None:
            try:
                from f5_tts_mlx.generate import generate
                self._generate_fn = generate
                logger.info("F5-TTS-MLX model loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import f5_tts_mlx: {e}")
                raise ImportError(
                    "f5-tts-mlx not installed. Install with: pip install f5-tts-mlx"
                )
    
    async def _generate_edge_tts(self, text: str) -> bytes:
        """
        Generate speech using edge-tts (online service).

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes
        """
        import edge_tts

        # Create a temporary file for the audio output
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp3')
        os.close(temp_fd)

        try:
            logger.info(f"Generating edge-tts audio to temporary file: {temp_path}")

            # Create edge-tts Communicate instance
            communicate = edge_tts.Communicate(text, self.voice)

            # Save to file
            await communicate.save(temp_path)

            # Read the generated audio file
            logger.info(f"Reading generated edge-tts audio from: {temp_path}")
            with open(temp_path, 'rb') as f:
                audio_data = f.read()

            logger.info(f"Successfully read edge-tts audio file: {len(audio_data)} bytes")

            return audio_data

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")

    async def _generate_elevenlabs(self, text: str) -> bytes:
        """
        Generate speech using ElevenLabs API.

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes
        """
        import aiohttp

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}"

        headers = {
            "xi-api-key": self.elevenlabs_api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.8,
                "similarity_boost": 0.9
            }
        }

        logger.info(f"Sending request to ElevenLabs API for {len(text)} characters")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"ElevenLabs API error: {response.status} - {error_text}")

                audio_data = await response.read()
                logger.info(f"Received {len(audio_data)} bytes from ElevenLabs")
                return audio_data

    def _generate_f5_tts(self, text: str) -> bytes:
        """
        Generate speech using F5-TTS-MLX (local generation).

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes
        """
        # Load model if not already loaded
        self._load_model()

        logger.info(f"Generating F5-TTS for {len(text)} characters of text")

        # Map model name to full model path
        model_map = {
            "F5-TTS": "lucasnewman/f5-tts-mlx",
            "E2-TTS": "lucasnewman/e2-tts-mlx"
        }
        model_name = model_map.get(self.model, "lucasnewman/f5-tts-mlx")

        # Generate audio using F5-TTS
        # F5-TTS generate() doesn't return audio - it either plays or saves to file
        # We need to specify output_path to prevent auto-playback and get the audio file

        # Create a temporary file for the audio output
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)  # Close the file descriptor, we'll use the path

        try:
            # Check for non-empty strings and strip whitespace
            has_voice_file = self.voice_file and self.voice_file.strip()
            has_voice_text = self.voice_text and self.voice_text.strip()

            logger.info(f"Generating audio to temporary file: {temp_path}")

            if has_voice_file and has_voice_text:
                logger.info(f"Using voice cloning with reference: {self.voice_file}")
                self._generate_fn(
                    generation_text=text,
                    ref_audio_path=self.voice_file.strip(),
                    ref_audio_text=self.voice_text.strip(),
                    model_name=model_name,
                    output_path=temp_path  # This prevents auto-playback!
                )
            else:
                # Use default voice
                logger.info("Using default voice (no voice cloning)")
                self._generate_fn(
                    generation_text=text,
                    model_name=model_name,
                    output_path=temp_path,  # This prevents auto-playback!
                    # quantization_bits=4   # keep it commented
                    steps=8,                # very fast generation, draft quality
                    cfg_strength=1
                )

            # Read the generated audio file
            logger.info(f"Reading generated audio from: {temp_path}")
            with open(temp_path, 'rb') as f:
                audio_data = f.read()

            logger.info(f"Successfully read audio file: {len(audio_data)} bytes")

            return audio_data

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")

    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech audio.

        Routes to the appropriate TTS engine based on configuration.

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes
        """
        self.is_processing = True
        start_time = time.time()

        try:
            logger.info(f"Generating TTS for {len(text)} characters of text using {self.engine}")

            # Route to appropriate engine
            if self.engine == "edge-tts":
                # edge-tts is async, need to run it in event loop
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                audio_data = loop.run_until_complete(self._generate_edge_tts(text))
            elif self.engine == "f5-tts":
                audio_data = self._generate_f5_tts(text)
            elif self.engine == "elevenlabs":
                # elevenlabs is async, need to run it in event loop
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                audio_data = loop.run_until_complete(self._generate_elevenlabs(text))
            else:
                raise ValueError(f"Unknown TTS engine: {self.engine}")

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
        try:
            logger.info(f"Streaming TTS for {len(text)} characters of text")
            
            # Generate complete audio (this already handles is_processing flag)
            audio_data = self.text_to_speech(text)
            
            # Split into chunks and yield
            total_chunks = (len(audio_data) + self.chunk_size - 1) // self.chunk_size
            
            for i in range(total_chunks):
                start_idx = i * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, len(audio_data))
                yield audio_data[start_idx:end_idx]
            
            logger.info(f"Completed TTS streaming")
            
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            raise
    
    async def async_text_to_speech(self, text: str) -> bytes:
        """
        Asynchronously generate audio data from the TTS engine.

        This method provides asynchronous TTS capability.
        For edge-tts, it runs natively async.
        For F5-TTS, it runs the synchronous method in a thread.

        Args:
            text: Text to convert to speech

        Returns:
            Complete audio data as bytes
        """
        self.is_processing = True
        start_time = time.time()

        try:
            logger.info(f"Generating TTS for {len(text)} characters of text using {self.engine}")

            # Route to appropriate engine
            if self.engine == "edge-tts":
                audio_data = await self._generate_edge_tts(text)
            elif self.engine == "f5-tts":
                # Run F5-TTS in a thread since it's synchronous
                audio_data = await asyncio.to_thread(self._generate_f5_tts, text)
            elif self.engine == "elevenlabs":
                audio_data = await self._generate_elevenlabs(text)
            else:
                raise ValueError(f"Unknown TTS engine: {self.engine}")

            # Calculate processing time
            self.last_processing_time = time.time() - start_time

            logger.info(f"Generated TTS audio after {self.last_processing_time:.2f}s, "
                       f"size: {len(audio_data)} bytes")

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
            "engine": self.engine,
            "model": self.model,
            "voice": self.voice,
            "voice_file": self.voice_file,
            "voice_text": self.voice_text,
            "output_format": self.output_format,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "is_processing": self.is_processing,
            "last_processing_time": self.last_processing_time
        }
