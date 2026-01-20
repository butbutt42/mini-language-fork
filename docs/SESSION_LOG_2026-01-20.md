# Session Log 2026-01-20 - Mini Language Fork

## Project Goal
Replace Whisper in Refuge.Help translator with Facebook's Omnilingual ASR for:
- 1600+ languages (vs Whisper's ~100)
- Better support for rare languages (Tigrinya, Pashto, etc.)
- Potentially better quality for low-resource languages

## Source Repository
- **Original:** https://github.com/facebookresearch/omnilingual-asr
- **Our Fork:** https://github.com/butbutt42/mini-language-fork

## What Was Done

### 1. Repository Analysis
Analyzed facebookresearch/omnilingual-asr:
- **Has inference code!** - `ASRInferencePipeline` class
- **Models available** - downloaded from `dl.fbaipublicfiles.com`
- **Key dependency:** `fairseq2` (Meta's sequence modeling library)

### 2. Created Fork with Simplified Setup
Files created:
- `README.md` - Setup instructions
- `inference.py` - CLI transcription script
- `server.py` - WebSocket server (Refuge.Help compatible)
- `Dockerfile` - RTX 3090/4090 Docker setup
- `requirements.txt` - Dependencies

### 3. Model Options for RTX 3090 (24GB)

| Model | VRAM | Speed | Notes |
|-------|------|-------|-------|
| `omniASR_CTC_1B_v2` | ~3 GB | 48x RT | Fastest, lower quality |
| `omniASR_CTC_3B_v2` | ~8 GB | 32x RT | Fast, good quality |
| `omniASR_LLM_3B_v2` | ~10 GB | ~1x RT | Good balance |
| `omniASR_LLM_7B_v2` | ~17 GB | ~1x RT | Best quality |
| `omniASR_LLM_Unlimited_7B_v2` | ~17 GB | ~1x RT | Long audio (>40s) |

---

## Next Steps

### Immediate
- [ ] Test installation on GPU server (158.51.110.52:22372)
- [ ] Verify fairseq2 installs correctly
- [ ] Run test transcription

### Integration with Refuge.Help
- [ ] Create `v1.0-whisper` tag in refuge-translator (backup current working version)
- [ ] Create `feature/omnilingual` branch
- [ ] Replace Whisper with Omnilingual ASR in `deploy/app.py`
- [ ] Test rare languages (Tigrinya, Pashto, Kurdish)
- [ ] Compare quality vs Whisper

### Architecture Change

**Current (Whisper):**
```
Audio → Whisper Turbo → Text → MADLAD-400 → Translation
```

**New (Omnilingual):**
```
Audio → Omnilingual ASR → Text → MADLAD-400 → Translation
```

Only ASR component changes. Translation (MADLAD-400) stays the same.

---

## Technical Notes

### Omnilingual ASR Dependencies
```
fairseq2>=0.5.2,<=0.6.0  # Meta's library - may have CUDA issues
torch>=2.5.0
torchaudio>=2.5.0
```

### Language Code Format
Omnilingual uses `{language}_{script}` format:
- `ukr_Cyrl` - Ukrainian Cyrillic
- `nld_Latn` - Dutch Latin
- `ara_Arab` - Arabic
- `tir_Ethi` - Tigrinya Ethiopic
- `pus_Arab` - Pashto Arabic script

### Potential Issues
1. **fairseq2 installation** - requires specific CUDA version matching
2. **Model size** - 7B model is ~30GB download
3. **Audio length limit** - standard models: 40s max, "Unlimited" models: no limit

---

## Files Reference

### mini-language-fork/
```
├── README.md           # Setup instructions
├── Dockerfile          # Docker for RTX 3090
├── inference.py        # CLI transcription
├── server.py           # WebSocket server
├── requirements.txt    # Dependencies
└── docs/
    └── SESSION_LOG_2026-01-20.md  # This file
```

### refuge-translator/ (to be modified)
```
├── deploy/
│   └── app.py          # Will replace Whisper with Omnilingual
└── frontend/
    └── index.html      # No changes needed (same API)
```

---

## Commands Reference

### Test on GPU Server
```bash
ssh -i ~/.ssh/refuge_key -p 22372 root@158.51.110.52

# Install
pip install omnilingual-asr

# Test
python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; print('OK')"
```

### Download Model Manually
```bash
wget https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B-v2.pt -O ~/.cache/fairseq2/models/omniASR-LLM-3B-v2.pt
```

---

*Session by Claude Opus 4.5*
