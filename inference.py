#!/usr/bin/env python3
"""
Simple inference script for Omnilingual ASR.
Supports 1600+ languages on local GPU.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Omnilingual ASR")
    parser.add_argument("--audio", "-a", required=True, help="Path to audio file")
    parser.add_argument("--lang", "-l", default=None, help="Language code (e.g., ukr_Cyrl, nld_Latn)")
    parser.add_argument("--model", "-m", default="omniASR_LLM_3B_v2",
                        help="Model card name (default: omniASR_LLM_3B_v2)")
    parser.add_argument("--device", "-d", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--list-langs", action="store_true", help="List all supported languages")

    args = parser.parse_args()

    # List languages mode
    if args.list_langs:
        try:
            from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
            print(f"Supported languages ({len(supported_langs)} total):\n")
            for lang in sorted(supported_langs):
                print(f"  {lang}")
        except ImportError:
            print("Error: omnilingual-asr not installed. Run: pip install omnilingual-asr")
            sys.exit(1)
        return

    # Check audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}")

    try:
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        # Check CUDA
        if args.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            args.device = "cpu"

        if args.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Load pipeline
        pipeline = ASRInferencePipeline(
            model_card=args.model,
            device=args.device
        )

        print(f"\nTranscribing: {audio_path}")
        if args.lang:
            print(f"Language: {args.lang}")

        # Transcribe
        lang_list = [args.lang] if args.lang else None
        results = pipeline.transcribe(
            [str(audio_path)],
            lang=lang_list,
            batch_size=1
        )

        print("\n" + "="*50)
        print("TRANSCRIPTION:")
        print("="*50)
        print(results[0])
        print("="*50)

    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("\nInstall with: pip install omnilingual-asr")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
