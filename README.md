# Mini Language Fork

Simplified inference wrapper for [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) - speech recognition for **1600+ languages** running locally on your GPU.

## Why This Fork?

The original repo has everything needed, but setup can be tricky. This fork provides:
- Simple Docker setup for RTX 3090 / RTX 4090
- Ready-to-use inference script
- WebSocket server for real-time transcription
- Integration examples for Refuge.Help translator

## Requirements

- NVIDIA GPU with 6-24GB VRAM (RTX 3090 recommended)
- Docker with NVIDIA runtime
- ~30GB disk space for models

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build
docker build -t omniasr .

# Run interactive
docker run --gpus all -it -v $(pwd)/models:/models omniasr

# Inside container
python inference.py --audio test.wav --lang ukr_Cyrl
```

### Option 2: Manual Install

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install omnilingual-asr

# Run
python inference.py --audio test.wav --lang ukr_Cyrl
```

## Models

| Model | VRAM | Speed | Best For |
|-------|------|-------|----------|
| `omniASR_CTC_3B_v2` | ~8 GB | 32x realtime | Fast transcription |
| `omniASR_LLM_3B_v2` | ~10 GB | ~1x realtime | Better quality |
| `omniASR_LLM_7B_v2` | ~17 GB | ~1x realtime | Best quality |
| `omniASR_LLM_Unlimited_7B_v2` | ~17 GB | ~1x realtime | Long audio (>40s) |

## Usage Examples

### Basic Transcription

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Load model (downloads automatically on first run)
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_3B_v2")

# Transcribe
result = pipeline.transcribe(
    ["audio.wav"],
    lang=["ukr_Cyrl"],  # Ukrainian Cyrillic
    batch_size=1
)
print(result[0])
```

### Supported Languages

```python
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

# 1600+ languages!
print(f"Total: {len(supported_langs)}")

# Common codes:
# ukr_Cyrl - Ukrainian
# nld_Latn - Dutch
# eng_Latn - English
# ara_Arab - Arabic
# fas_Arab - Farsi
# pus_Arab - Pashto
# tir_Ethi - Tigrinya
```

### Real-time WebSocket Server

```python
python server.py --model omniASR_LLM_3B_v2 --port 8000
```

## Troubleshooting

### fairseq2 installation fails

```bash
# Try with specific CUDA version
pip install fairseq2 --extra-index-url https://fair.pkg.meta.com/fairseq2/pt2.5.1/cu124
```

### Out of memory

Use a smaller model:
```python
pipeline = ASRInferencePipeline(model_card="omniASR_CTC_1B_v2")  # Only 3GB VRAM
```

### Model download stuck

Download manually:
```bash
wget https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B-v2.pt -O ~/.cache/fairseq2/models/omniASR-LLM-3B-v2.pt
```

## Integration with Refuge.Help

This can replace Whisper in the Refuge.Help translator for better rare language support.

See `examples/refuge_integration.py` for WebSocket server compatible with the existing frontend.

## Credits

- Original: [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)
- Paper: [Omnilingual ASR](https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/)

## License

BSD License (same as original)
