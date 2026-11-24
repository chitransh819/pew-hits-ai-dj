# Pew Hits AI DJ - Research Report (Phase 1)

## Executive Summary

This report presents comprehensive research on audio analysis technologies and NVIDIA AI models for building an intelligent music recommendation engine for two internet radio stations: Pew Hits English and Pew Hits Desi.

**Target Hardware:** 2-vCPU, ~12GB RAM ARM64 Oracle VPS
**Constraint:** Commercial licensing required, no production system modifications

---

## 1. Audio Analysis Technology Research

### 1.1 Essentia Library & Pretrained Models

Essentia 2.1 (MTG/UPF) provides a comprehensive C++ library with Python bindings and TensorFlow model inference for music analysis. Key findings:

| Model | Task | Accuracy | License | Commercial |
|-------|------|----------|---------|------------|
| discogs-effnet-bsdynamic-1 | Feature extraction/embeddings | SOTA on Discogs 400/519 classes | CC BY-NC-SA 4.0 | ❌ |
| genre_discogs400-discogs-effnet-1 | Genre (400 Discogs styles) | ~0.48 mAP on MTG-Jamendo | CC BY-NC-SA 4.0 | ❌ |
| genre_discogs519-discogs-maest-30s | Genre (519 styles, MAEST) | Best-in-class Discogs | CC BY-NC-SA 4.0 | ❌ |
| mtg_jamendo_genre-discogs-effnet-1 | Genre (87 Jamendo genres) | ~0.85+ ROC-AUC/genre | CC BY-NC-SA 4.0 | ❌ |
| mood_happy/sad/aggressive/relaxed/party/acoustic/electronic | Mood (7 binary) | 0.60-0.98 ROC-AUC | CC BY-NC-SA 4.0 | ❌ |
| danceability-discogs-effnet-1 | Danceable/not | 0.72-0.94 ROC-AUC | CC BY-NC-SA 4.0 | ❌ |
| voice_instrumental-discogs-effnet-1 | Vocal/instrumental | 0.72-0.98 ROC-AUC | CC BY-NC-SA 4.0 | ❌ |
| tempocnn/deeptemp-k4/deepsquare-k16 | Tempo/BPM (30-286, 256 classes) | SOTA tempo estimation | CC BY-NC-SA 4.0 | ❌ |
| OpenL3 (included in Essentia) | Audio embeddings | Outperforms VGGish | MIT/CC BY 4.0 | ✅ |

**Critical Finding:** ALL Essentia pretrained models use CC BY-NC-SA 4.0 (non-commercial only). Commercial license requires negotiation with UPF/MTG - cost and timeline uncertain.

**ARM64 Compatibility:** Essentia builds support ARM64 (GitHub issue #1299), but compiling with TensorFlow support on ARM64 is non-trivial.

### 1.2 Alternative Commercial-Ready Technologies

| Technology | License | Capabilities | Commercial | ARM64 | Recommendation |
|------------|---------|--------------|------------|-------|----------------|
| **OpenL3** | MIT/CC BY 4.0 | 512/6144-dim embeddings, music/env variants | ✅ | ✅ | **Primary embeddings** |
| **EfficientAT** | MIT | AudioSet CNNs (MobileNetV3), multiple sizes | ✅ | ✅ | **Primary embeddings** |
| **MusicNN-PyTorch** | Apache 2.0 | 50 music tags (genre/mood/instrument) | ✅ | ✅ | Fallback taxonomy |
| **LAION CLAP HTSAT-tiny** | CC0-1.0 | Audio-text joint embeddings, zero-shot | ✅ | ⚠️ (transformer) | Semantic search |
| **librosa** | BSD-3 | BPM, beats, onsets, spectral, tonal | ✅ | ✅ | **Essential low-level** |
| **madmom** | BSD-2 (code) / NC (weights) | SOTA beat/downbeat, tempo, chords | ❌ (weights) | ✅ | Avoid (NC weights) |
| **MIT AST** | BSD-3 | AudioSet transformer (86M params) | ✅ | ⚠️ (heavy) | If transformer needed |

### 1.3 Recommended Audio Analysis Stack

**Primary Stack (All Commercial-Ready):**
1. **OpenL3 (music variant, mel256, emb512)** - General audio embeddings
2. **EfficientAT (mn10_as)** - Efficient AudioSet embeddings, multiple complexity tiers
3. **librosa** - Low-level features (BPM, beats, spectral, tonal, loudness)
4. **Custom classifiers** (scikit-learn/XGBoost) on embeddings → genre/mood/danceability

**Fallback Stack:**
- MusicNN-PyTorch for 50-tag taxonomy
- LAION CLAP for zero-shot semantic tagging
- MIT AST if transformer embeddings required

**Resource Profile:**
- Total model footprint: <100MB
- Inference time: ~200-500ms/track on 2-vCPU ARM64
- RAM usage: Well within 12GB
- No GPU required

---

## 2. NVIDIA AI Models Research

### 2.1 Available Models (integrate.api.nvidia.com/v1)

| Model | Type | Context | Key Capabilities | Suitable |
|-------|------|---------|------------------|----------|
| Nemotron 3 Ultra 550B | Reasoning LLM | 1M | Frontier reasoning, tool calling, JSON output | ✅ Primary |
| Nemotron 3 Super 120B | Reasoning LLM | 1M | Efficient reasoning, MTP, NVFP4 | ✅ Balanced |
| Nemotron 3.5 Lightning 30B | Reasoning LLM | 1M | Fastest, speculative decoding | ✅ High-volume |
| Nemotron 3.5 Bolt E4B | Reasoning LLM | 128K | Compact, edge/consumer | ✅ Lightweight |
| Nemotron 3.5 Embed 1B | Embeddings | 10K | Multimodal (text+image), 2048-dim | ❌ Audio only |
| Nemotron 3.5 Embed 8B | Embeddings | 10K | Higher quality multimodal | ❌ Audio only |
| Nemotron 3.5 Rerank 8B | Reranking | 32K | Multimodal reranking, 512 passages | ✅ Re-ranking |
| Nemotron 3 Nano Omni 30B | Multimodal | 262K | Video/audio/image/text understanding | ❌ Not music-specific |
| Nemotron 3.5 Super VL 120B | Multimodal | 1M | Early-access multimodal reasoning | ❌ Early access |

### 2.2 Key API Constraints

- **Endpoint:** `https://integrate.api.nvidia.com/v1` (OpenAI-compatible)
- **Auth:** Bearer token (`nvapi-...`)
- **Rate Limit:** ~40 RPM free tier (shared across ALL models/account)
- **Free Tier Terms:** Prototyping only, no production use, NVIDIA may use prompts/outputs
- **Paid Pricing:** ~$0.50/M input, $2.20/M output (Ultra); cheaper for smaller models

### 2.3 Integration Value Assessment

| Capability | Supported | Notes |
|------------|-----------|-------|
| Context Interpretation | ✅ | 1M context, reasoning over structured features |
| Atmosphere Classification | ✅ | JSON output with structured prompts |
| Ranking Insights | ✅ | Chain-of-thought for candidate scoring |
| Explanation Generation | ✅ | Natural language + structured JSON |
| Structured JSON Output | ✅ | `response_format: {type: "json_object"}` |
| Audio Analysis | ❌ | No suitable audio-input models for music |

### 2.4 Recommended NVIDIA Integration Approach

**Model Assignment:**
- **Nemotron 3 Ultra 550B** - Primary: complex explanations, ranking rationale
- **Nemotron 3.5 Lightning 30B** - High-volume: atmosphere classification, pre-filtering
- **Nemotron 3.5 Embed 1B** - Semantic similarity: track-to-track, context-to-candidate
- **Nemotron 3.5 Rerank 8B** - Final re-ranking: top-10 candidates

**Data Flow:**
```
Local Audio Analysis → Structured Features (BPM, key, energy, spectral, mood tags)
                        ↓
              NVIDIA LLM Reasoning (structured JSON in/out)
                        ↓
              Ranking Insights + Explanations
                        ↓
              Local Fallback if API unavailable
```

**Critical:** NEVER send raw audio to NVIDIA models. Only structured features from local analysis.

### 2.5 Implementation Requirements

1. **Rate Limiting:** Client-side token bucket (40 RPM), request queuing
2. **Timeouts:** 30-60s (Ultra), 15-30s (Lightning/Super), 10s (Embed/Rerank)
3. **Retries:** Exponential backoff + jitter, max 3-5 attempts (429, 5xx, timeout)
4. **Error Logging:** No API keys, no sensitive content in logs
5. **Structured Output:** `response_format: {type: "json_object"}`, Pydantic validation, self-repair retry
6. **Local Fallback:** Cache recent LLM responses, rule-based ranking when API down
7. **Cost Monitoring:** Token tracking, monthly budget alerts ($50-100/mo production)
8. **Metrics:** TTFT, tokens/sec, error rates per model
9. **Streaming:** Use for long generations to reduce perceived latency
10. **Security:** Separate keys per environment, rotation, no hardcoded credentials

---

## 3. Architecture Recommendations

### 3.1 Two-Tier Design

**Tier 1: Local Audio Analysis (Always Available)**
- Runs on VPS, no external dependencies
- Produces structured features for every track
- Cached per track (analyze once, reuse forever)

**Tier 2: NVIDIA Reasoning (Enhancement)**
- Called for recommendation decisions
- Graceful degradation to local-only when unavailable
- Adds semantic understanding and explanations

### 3.2 Station Separation

Each station maintains independent:
- Playback history (last 50+ tracks)
- Recommendation history
- Genre/mood/energy preferences
- Artist repetition controls
- Station-specific configuration

No catalogue mixing unless explicitly configured.

---

## 4. Unresolved Decisions Requiring Approval

1. **NVIDIA Production Use:** Free tier prohibits production. Need paid tier decision or self-hosted NIM.
2. **Custom Classifier Training Data:** MTG-Jamendo (Apache 2.0) vs other sources for mood/genre heads.
3. **Recommendation Algorithm Weights:** Need configurable scoring with A/B testing framework.
4. **Transition Smoothness Metrics:** Define quantitative measures for mood/energy/genre transitions.
5. **Evaluation Framework:** Simulated playback with ground truth vs human evaluation.

---

## 5. Next Phase Deliverables (Phase 2)

Upon approval, Phase 2 will produce:
- `ARCHITECTURE.md` - System architecture, data models, algorithm design
- `tech-stack.yaml` - Pinned dependencies, versions, deployment config
- API contracts (OpenAPI spec)
- Station profile configurations
- Security Engineer review
- Code Reviewer review

---

## Appendix: Full Research Data

- `/opt/data/audio_analysis_research.json` - Complete audio analysis comparison
- `/opt/data/nvidia_models_research.json` - Complete NVIDIA models evaluation