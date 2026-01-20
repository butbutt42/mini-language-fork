#!/usr/bin/env python3
"""
WebSocket server for real-time speech transcription.
Compatible with Refuge.Help frontend.
"""

import argparse
import asyncio
import json
import logging
import numpy as np
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Will be set after imports
pipeline = None
SAMPLE_RATE = 16000


async def handle_websocket(websocket, path):
    """Handle WebSocket connection for audio streaming."""
    import torch

    # Parse query params
    from urllib.parse import parse_qs, urlparse
    parsed = urlparse(path)
    params = parse_qs(parsed.query)

    source_lang = params.get("source", [None])[0]
    target_lang = params.get("target", ["nld_Latn"])[0]  # Default Dutch

    # Map short codes to full codes
    LANG_MAP = {
        "uk": "ukr_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn",
        "nl": "nld_Latn", "de": "deu_Latn", "fr": "fra_Latn",
        "ar": "ara_Arab", "fa": "fas_Arab", "ps": "pus_Arab",
        "so": "som_Latn", "ti": "tir_Ethi", "tr": "tur_Latn",
        "pl": "pol_Latn", "it": "ita_Latn", "ku": "kmr_Latn",
    }

    if source_lang and source_lang in LANG_MAP:
        source_lang = LANG_MAP[source_lang]

    logger.info(f"New connection: source={source_lang}, target={target_lang}")

    audio_buffer = []
    silence_count = 0
    SILENCE_THRESHOLD = 0.008
    MIN_AUDIO_LENGTH = SAMPLE_RATE * 0.5  # 0.5 seconds minimum
    MAX_AUDIO_LENGTH = SAMPLE_RATE * 30   # 30 seconds maximum

    try:
        await websocket.send(json.dumps({"status": "connected"}))

        async for message in websocket:
            if isinstance(message, bytes):
                # Convert bytes to float32 audio
                audio_chunk = np.frombuffer(message, dtype=np.float32)

                # Check RMS for silence detection
                rms = np.sqrt(np.mean(audio_chunk ** 2))

                if rms > SILENCE_THRESHOLD:
                    audio_buffer.extend(audio_chunk)
                    silence_count = 0
                    await websocket.send(json.dumps({"status": "listening"}))
                else:
                    silence_count += 1

                    # Process if we have audio and detected silence
                    if len(audio_buffer) > MIN_AUDIO_LENGTH and silence_count > 3:
                        await websocket.send(json.dumps({"status": "processing"}))

                        # Prepare audio
                        audio_np = np.array(audio_buffer, dtype=np.float32)

                        # Truncate if too long
                        if len(audio_np) > MAX_AUDIO_LENGTH:
                            audio_np = audio_np[:MAX_AUDIO_LENGTH]

                        # Transcribe
                        try:
                            audio_dict = {
                                "waveform": audio_np,
                                "sample_rate": SAMPLE_RATE
                            }

                            lang_list = [source_lang] if source_lang else None
                            results = pipeline.transcribe(
                                [audio_dict],
                                lang=lang_list,
                                batch_size=1
                            )

                            transcription = results[0] if results else ""

                            if transcription.strip():
                                # Send result (translation would be separate step)
                                await websocket.send(json.dumps({
                                    "status": "result",
                                    "source": transcription,
                                    "translation": transcription,  # TODO: Add translation
                                    "lang": source_lang or "detected"
                                }))
                                logger.info(f"Transcribed: {transcription[:50]}...")

                        except Exception as e:
                            logger.error(f"Transcription error: {e}")
                            await websocket.send(json.dumps({
                                "status": "error",
                                "message": str(e)
                            }))

                        # Clear buffer
                        audio_buffer = []
                        silence_count = 0

                    elif len(audio_buffer) > MAX_AUDIO_LENGTH:
                        # Force process if buffer is too long
                        audio_buffer = audio_buffer[-int(MAX_AUDIO_LENGTH):]

            elif isinstance(message, str):
                # Handle text commands
                try:
                    data = json.loads(message)
                    if data.get("command") == "ping":
                        await websocket.send(json.dumps({"pong": True}))
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Connection closed")


async def main(args):
    global pipeline

    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    logger.info(f"Loading model: {args.model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    pipeline = ASRInferencePipeline(
        model_card=args.model,
        device=device
    )

    logger.info(f"Model loaded. Starting server on port {args.port}")

    import websockets

    async with websockets.serve(handle_websocket, "0.0.0.0", args.port):
        logger.info(f"WebSocket server running on ws://0.0.0.0:{args.port}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omnilingual ASR WebSocket Server")
    parser.add_argument("--model", "-m", default="omniASR_LLM_3B_v2",
                        help="Model card (default: omniASR_LLM_3B_v2)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="WebSocket port (default: 8000)")

    args = parser.parse_args()
    asyncio.run(main(args))
