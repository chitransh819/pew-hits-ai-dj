# Pew Hits AI DJ - Architecture Design (Phase 2)

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PEW HITS ECOSYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    HTTP/REST      ┌─────────────────────────────────────┐  │
│  │  PEW HITS       │◄─────────────────►│        AI DJ SERVICE                │  │
│  │  ENGLISH        │  /recommendations │  ┌─────────────────────────────────┐ │  │
│  │  (Container)    │  /tracks/analyze  │  │  FASTAPI APP                    │ │  │
│  │                 │  /stations/{id}/  │  │  ├── /recommendations/next      │ │  │
│  │ • Cache (1,223) │   history         │  │  ├── /tracks/analyze            │ │  │
│  │ • playlist.txt  │  /stations/{id}/  │  │  ├── /stations/{id}/history     │ │  │
│  │ • Downloaders   │   status          │  │  └── /health                    │ │  │
│  │ • FFmpeg/Icecast│                   │  └──────────────┬─────────────────┘ │  │
│  └─────────────────┘                   │                 │                 │  │
│                                        │  ┌──────────────▼─────────────────┐ │  │
│  ┌─────────────────┐                   │  │  ANALYSIS ENGINE              │ │  │
│  │  PEW HITS       │                   │  │  ├── librosa (BPM, spectral)  │ │  │
│  │  DESI           │                   │  │  ├── OpenL3 Embeddings        │ │  │
│  │  (Container)    │                   │  │  ├── EfficientAT Embeddings   │ │  │
│  │                 │                   │  │  ├── Custom Classifiers       │ │  │
│  │ • Cache (1,687) │                   │  │  └── File Watcher + Reconcile │ │  │
│  │ • playlist.txt  │                   │  └──────────────┬─────────────────┘ │  │
│  │ • Downloaders   │                   │                 │                 │  │
│  │ • FFmpeg/Icecast│                   │  ┌──────────────▼─────────────────┐ │  │
│  └─────────────────┘                   │  │  RECOMMENDATION ENGINE        │ │  │
│                                        │  │  ├── Vibe Detection           │ │  │
│  ┌─────────────────┐                   │  │  ├── Candidate Generation     │ │  │
│  │  SHARED VOLUMES │                   │  │  ├── Scoring & Ranking        │ │  │
│  │                 │                   │  │  ├── Transition Validation    │ │  │
│  │ /cache/english  │◄──────────────────┤  │  ├── NVIDIA Reasoning (opt)   │ │  │
│  │ /cache/desi     │   Read-only       │  │  └── Explanation Generator    │ │  │
│  └─────────────────┘                   │  └──────────────┬─────────────────┘ │  │
│                                        │                 │                 │  │
│                                        │  ┌──────────────▼─────────────────┐ │  │
│                                        │  │  DATA LAYER                   │ │  │
│                                        │  │  ├── SQLite (tracks, features)│ │  │
│                                        │  │  ├── Track Identity Mapping   │ │  │
│                                        │  │  ├── Cache Status Sync        │ │  │
│                                        │  │  └── Station Profiles         │ │  │
│                                        │  └─────────────────────────────────┘ │  │
│                                        └─────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Models

### 2.1 Track Identity (Canonical ID System)

```python
# Canonical ID format: track:{source}:{source_id}
# Examples:
# track:spotify:4cOdK2wGLETKBW3PvgPWqT
# track:youtube:dQw4w9WgXcQ
# track:cache:blinding_lights_the_weeknd
# track:soundcloud:123456789
```

### 2.2 SQLite Schema

```sql
-- Track identity mapping (unifies all source IDs)
CREATE TABLE track_identity (
    canonical_id      TEXT PRIMARY KEY,      -- track:spotify:xxx
    spotify_id        TEXT,
    youtube_id        TEXT,
    soundcloud_id     TEXT,
    jiosaavn_id       TEXT,
    cache_filename    TEXT,                  -- Blinding_Lights_-_The_Weeknd.mp3
    title             TEXT NOT NULL,
    artist            TEXT NOT NULL,
    album             TEXT,
    duration_sec      INTEGER,
    language          TEXT,                  -- en, hi, pa, bn, etc.
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audio features (analyzed once, cached forever)
CREATE TABLE track_features (
    canonical_id      TEXT PRIMARY KEY REFERENCES track_identity(canonical_id),
    -- Low-level (librosa)
    bpm               REAL,
    beats             TEXT,                  -- JSON array of beat times
    onset_times       TEXT,                  -- JSON array
    spectral_centroid REAL,
    spectral_rolloff  REAL,
    spectral_bandwidth REAL,
    spectral_contrast TEXT,                  -- JSON array
    chroma            TEXT,                  -- JSON 12-dim
    key               TEXT,                  -- e.g., "C# minor"
    loudness_lufs     REAL,
    -- Embeddings
    openl3_embedding  BLOB,                  -- 512-dim float32
    efficientat_embedding BLOB,              -- 512-dim float32
    -- Classifiers (custom sklearn/XGB)
    genre_top         TEXT,                  -- Primary genre
    genre_scores      TEXT,                  -- JSON: {"pop": 0.8, "rock": 0.1}
    mood_tags         TEXT,                  -- JSON: ["romantic", "relaxed"]
    mood_scores       TEXT,                  -- JSON per mood
    danceability      REAL,                  -- 0.0-1.0
    energy            REAL,                  -- 0.0-1.0
    valence           REAL,                  -- 0.0-1.0
    acousticness      REAL,
    instrumentalness  REAL,
    vocal_presence    REAL,
    -- Cache status
    cache_status      TEXT DEFAULT 'available',  -- available | evicted | missing
    last_cache_check  TIMESTAMP,
    -- Analysis metadata
    analysis_version  TEXT DEFAULT '1.0',
    analyzed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_mtime        INTEGER,               -- For change detection
    file_size         INTEGER
);

-- Station profiles (independent per station)
CREATE TABLE station_profiles (
    station_id        TEXT PRIMARY KEY,      -- pew-hits-english, pew-hits-desi
    name              TEXT NOT NULL,
    languages         TEXT NOT NULL,         -- JSON: ["en"] or ["hi", "pa", "bn"]
    primary_genres    TEXT NOT NULL,         -- JSON array
    mood_preferences  TEXT NOT NULL,         -- JSON: {"romantic": 1.0, "relaxed": 0.8}
    energy_range      TEXT NOT NULL,         -- JSON: [0.1, 0.5]
    artist_cooldown   INTEGER DEFAULT 3,     -- Tracks before same artist
    transition_rules  TEXT,                  -- JSON: max_bpm_delta, max_energy_delta
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Recommendation history (for evaluation)
CREATE TABLE recommendation_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id        TEXT NOT NULL,
    current_track_id  TEXT NOT NULL,
    recommended_id    TEXT NOT NULL,
    candidates_json   TEXT NOT NULL,         -- Full ranked list with scores
    explanation       TEXT,
    confidence        REAL,
    reasoning_source  TEXT,                  -- local | nvidia | local+nvidia
    latency_ms        INTEGER,
    accepted          BOOLEAN,               -- Did pewhits play it?
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Playback history (mirror of pewhits history.json)
CREATE TABLE playback_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id        TEXT NOT NULL,
    canonical_id      TEXT NOT NULL,
    played_at         TIMESTAMP NOT NULL,
    source            TEXT,                  -- cache | youtube | soundcloud | jiosaavn | direct_stream
    duration_sec      INTEGER,
    FOREIGN KEY (canonical_id) REFERENCES track_identity(canonical_id)
);

-- Indexes
CREATE INDEX idx_track_features_cache_status ON track_features(cache_status);
CREATE INDEX idx_playback_history_station_time ON playback_history(station_id, played_at DESC);
CREATE INDEX idx_recommendation_history_station_time ON recommendation_history(station_id, created_at DESC);
```

---

## 3. Recommendation Algorithm

### 3.1 Pipeline Flow

```
CURRENT TRACK + LAST 5 TRACKS (from pewhits)
         │
         ▼
┌─────────────────────────────────────────────┐
│ VIBE DETECTION                              │
│ • Current mood/energy/tempo from features   │
│ • Station baseline preferences              │
│ • Recent trajectory (trending up/down?)     │
│ • Output: target_vibe {mood, energy, tempo} │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ CANDIDATE GENERATION                        │
│ • Filter: station_id match                  │
│ • Filter: playable_track_ids (from pewhits) │
│ • Exclude: last N played (artist + track)   │
│ • Exclude: blocked tracks                   │
│ • Output: ~50-200 candidates                │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ SCORING (per candidate)                     │
│ score = Σ(w_i × feature_i)                  │
│                                             │
│ w_mood      × mood_similarity(candidate)    │
│ w_energy    × energy_compatibility()        │
│ w_tempo     × tempo_compatibility()         │
│ w_genre     × genre_compatibility()         │
│ w_language  × language_match()              │
│ w_artist    × artist_diversity_bonus()      │
│ w_transition × transition_smoothness()      │
│ w_semantic  × embedding_similarity()        │
│ w_recency   × recency_penalty()             │
│                                             │
│ Weights configurable per station            │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ TOP-K SELECTION (K=10)                      │
│ • Sort by score descending                  │
│ • Apply transition validation               │
│ • Ensure artist diversity in top-10         │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ NVIDIA RE-RANKING (optional)                │
│ • Send top-10 + context to Nemotron         │
│ • Get refined ranking + explanation         │
│ • Fallback to local scores if NVIDIA fails  │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ FINAL OUTPUT                                │
│ {                                           │
│   "candidates": [...],                      │
│   "explanation": "...",                     │
│   "confidence": 0.89,                       │
│   "reasoning_source": "local+nvidia"        │
│ }                                           │
└─────────────────────────────────────────────┘
```

### 3.2 Scoring Functions

```python
# All scores normalized 0.0-1.0

def mood_similarity(candidate, target_vibe):
    """Cosine similarity between mood score vectors"""
    return cosine_similarity(candidate.mood_vector, target_vibe.mood_vector)

def energy_compatibility(candidate, target_vibe):
    """Gaussian around target energy"""
    diff = abs(candidate.energy - target_vibe.energy)
    return exp(-0.5 * (diff / 0.15) ** 2)  # σ=0.15

def tempo_compatibility(candidate, target_vibe):
    """BPM difference with harmonic tolerance"""
    bpm_diff = abs(candidate.bpm - target_vibe.tempo)
    # Allow ±10% or harmonic multiples (2x, 0.5x)
    if bpm_diff < target_vibe.tempo * 0.1:
        return 1.0
    harmonic_ratios = [0.5, 2.0]
    for r in harmonic_ratios:
        if abs(candidate.bpm - target_vibe.tempo * r) < target_vibe.tempo * 0.05:
            return 0.9
    return max(0.0, 1.0 - bpm_diff / 60.0)

def genre_compatibility(candidate, station_profile):
    """Station genre weights × candidate genre scores"""
    return sum(station_profile.genre_weights.get(g, 0) * s 
               for g, s in candidate.genre_scores.items())

def language_match(candidate, station_profile):
    return 1.0 if candidate.language in station_profile.languages else 0.0

def artist_diversity_bonus(candidate, recent_history):
    """Penalty if artist in last N tracks"""
    recent_artists = {h.artist for h in recent_history[-5:]}
    return 0.0 if candidate.artist in recent_artists else 1.0

def transition_smoothness(candidate, current_track):
    """Multi-factor transition quality"""
    energy_slope = abs(candidate.energy - current_track.energy)
    mood_shift = 1.0 - mood_similarity(candidate, current_track)
    tempo_jump = abs(candidate.bpm - current_track.bpm) / max(current_track.bpm, 1)
    return 1.0 - (0.4*energy_slope + 0.3*mood_shift + 0.3*tempo_jump)

def recency_penalty(candidate, playback_history, window_hours=24):
    """Exponential decay penalty for recent plays"""
    hours_since = hours_since_last_play(candidate.canonical_id, playback_history)
    if hours_since is None:
        return 1.0
    return 1.0 - exp(-hours_since / 12.0)  # Half-life 12h

def semantic_similarity(candidate, context_embedding):
    """OpenL3/EfficientAT embedding cosine similarity"""
    return max(
        cosine_similarity(candidate.openl3_emb, context_embedding),
        cosine_similarity(candidate.efficientat_emb, context_embedding)
    )
```

### 3.3 Default Weights (Per Station)

| Weight | English | Desi | Description |
|--------|---------|------|-------------|
| `w_mood` | 0.25 | 0.30 | Mood continuity |
| `w_energy` | 0.20 | 0.20 | Energy flow |
| `w_tempo` | 0.15 | 0.10 | BPM compatibility |
| `w_genre` | 0.15 | 0.15 | Genre fit |
| `w_language` | 0.10 | 0.10 | Language match |
| `w_artist` | 0.05 | 0.05 | Artist diversity |
| `w_transition` | 0.05 | 0.05 | Transition smoothness |
| `w_semantic` | 0.05 | 0.05 | Embedding similarity |

---

## 4. API Contracts (OpenAPI 3.0)

```yaml
openapi: 3.0.3
info:
  title: Pew Hits AI DJ API
  version: 1.0.0
  description: Intelligent music recommendation engine for Pew Hits radio stations

servers:
  - url: http://localhost:8000/api/v1
    description: AI DJ Service

paths:
  /recommendations/next:
    post:
      summary: Get next track recommendation
      operationId: getNextRecommendation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RecommendationRequest'
      responses:
        '200':
          description: Ranked recommendations
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RecommendationResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '503':
          $ref: '#/components/responses/ServiceUnavailable'

  /tracks/analyze:
    post:
      summary: Analyze a new track (incremental)
      operationId: analyzeTrack
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TrackAnalysisRequest'
      responses:
        '200':
          description: Analysis complete
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TrackAnalysisResponse'
        '202':
          description: Analysis queued (large file)
        '409':
          description: Already analyzed (unchanged)

  /stations/{station_id}/history:
    post:
      summary: Update playback history
      operationId: updateHistory
      parameters:
        - name: station_id
          in: path
          required: true
          schema:
            type: string
            enum: [pew-hits-english, pew-hits-desi]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/HistoryUpdateRequest'
      responses:
        '200':
          description: History updated

  /stations/{station_id}/recommendations:
    get:
      summary: Get recent recommendation history
      operationId: getRecommendationHistory
      parameters:
        - name: station_id
          in: path
          required: true
          schema:
            type: string
            enum: [pew-hits-english, pew-hits-desi]
        - name: limit
          in: query
          schema:
            type: integer
            default: 50
      responses:
        '200':
          description: Recommendation history
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RecommendationHistoryResponse'

  /stations/{station_id}/status:
    get:
      summary: Get station status and config
      operationId: getStationStatus
      parameters:
        - name: station_id
          in: path
          required: true
          schema:
            type: string
            enum: [pew-hits-english, pew-hits-desi]
      responses:
        '200':
          description: Station status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/StationStatusResponse'

  /health:
    get:
      summary: Health check
      operationId: healthCheck
      responses:
        '200':
          description: Service healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

components:
  schemas:
    RecommendationRequest:
      type: object
      required: [station, current_track, recent_history, playable_track_ids]
      properties:
        station:
          type: string
          enum: [pew-hits-english, pew-hits-desi]
        current_track:
          $ref: '#/components/schemas/TrackContext'
        recent_history:
          type: array
          items:
            $ref: '#/components/schemas/TrackContext'
          maxItems: 10
        playable_track_ids:
          type: array
          items:
            type: string
          description: Canonical IDs currently playable (cache + searchable)
          maxItems: 5000

    TrackContext:
      type: object
      required: [canonical_id, title, artist]
      properties:
        canonical_id:
          type: string
          pattern: '^track:(spotify|youtube|soundcloud|jiosaavn|cache):.+'
        title: {type: string}
        artist: {type: string}
        album: {type: string}
        duration_sec: {type: integer, minimum: 0}
        played_at: {type: string, format: date-time}
        source: {type: string, enum: [cache, youtube, soundcloud, jiosaavn, direct_stream, spotify_playlist]}

    RecommendationResponse:
      type: object
      required: [candidates, explanation, confidence, reasoning_source]
      properties:
        candidates:
          type: array
          items:
            $ref: '#/components/schemas/Candidate'
          minItems: 1
          maxItems: 10
        explanation:
          type: string
          description: Human-readable explanation
        confidence:
          type: number
          minimum: 0
          maximum: 1
        reasoning_source:
          type: string
          enum: [local, nvidia, local+nvidia]

    Candidate:
      type: object
      required: [canonical_id, score, reason]
      properties:
        canonical_id:
          type: string
          pattern: '^track:(spotify|youtube|soundcloud|jiosaavn|cache):.+'
        score:
          type: number
          minimum: 0
          maximum: 1
        reason:
          type: string
        metadata:
          $ref: '#/components/schemas/TrackContext'

    TrackAnalysisRequest:
      type: object
      required: [file_path]
      properties:
        file_path:
          type: string
          description: Path relative to cache mount
        force:
          type: boolean
          default: false
          description: Re-analyze even if unchanged

    TrackAnalysisResponse:
      type: object
      required: [canonical_id, status]
      properties:
        canonical_id:
          type: string
        status:
          type: string
          enum: [analyzed, skipped_unchanged, queued, failed]
        features:
          $ref: '#/components/schemas/TrackFeatures'
        duration_ms:
          type: integer

    TrackFeatures:
      type: object
      properties:
        bpm: {type: number}
        key: {type: string}
        energy: {type: number}
        danceability: {type: number}
        mood_tags: {type: array, items: {type: string}}
        genre_top: {type: string}
        genre_scores: {type: object}
        vocal_presence: {type: number}
        acousticness: {type: number}
        instrumentalness: {type: number}

    HistoryUpdateRequest:
      type: object
      required: [canonical_id, played_at, source]
      properties:
        canonical_id: {type: string}
        played_at: {type: string, format: date-time}
        source: {type: string}
        duration_sec: {type: integer}

    RecommendationHistoryResponse:
      type: object
      properties:
        station: {type: string}
        recommendations:
          type: array
          items:
            $ref: '#/components/schemas/RecommendationRecord'

    RecommendationRecord:
      type: object
      properties:
        timestamp: {type: string, format: date-time}
        current_track: {type: string}
        recommended_track: {type: string}
        confidence: {type: number}
        accepted: {type: boolean}
        latency_ms: {type: integer}

    StationStatusResponse:
      type: object
      properties:
        station: {type: string}
        status: {type: string, enum: [ready, analyzing, degraded]}
        tracks_analyzed: {type: integer}
        cache_status: {type: string}
        last_reconciliation: {type: string, format: date-time}
        profile:
          $ref: '#/components/schemas/StationProfile'

    StationProfile:
      type: object
      properties:
        languages: {type: array, items: {type: string}}
        primary_genres: {type: array, items: {type: string}}
        mood_preferences: {type: object}
        energy_range: {type: array, items: {type: number}, minItems: 2, maxItems: 2}
        artist_cooldown: {type: integer}

    HealthResponse:
      type: object
      required: [status, version, tracks_analyzed]
      properties:
        status: {type: string, enum: [healthy, starting, degraded]}
        version: {type: string}
        tracks_analyzed: {type: integer}
        uptime_seconds: {type: integer}
        models_loaded: {type: boolean}
        db_connected: {type: boolean}

  responses:
    BadRequest:
      description: Invalid request
    ServiceUnavailable:
      description: AI DJ not ready (startup analysis in progress)
```

---

## 5. Station Profiles

### 5.1 Pew Hits English

```yaml
station_id: pew-hits-english
name: "Pew Hits English"
languages: ["en"]
primary_genres:
  - pop
  - soft_pop
  - acoustic
  - indie_pop
  - relaxing
  - romantic
mood_preferences:
  relaxed: 1.0
  romantic: 0.9
  acoustic: 0.8
  calm: 0.8
  happy: 0.5
  nostalgic: 0.4
energy_range: [0.15, 0.50]  # Low to medium
artist_cooldown: 3
transition_rules:
  max_bpm_delta: 20
  max_energy_delta: 0.25
  prefer_smooth_transitions: true
  avoid_abrupt_genre_shifts: true
```

### 5.2 Pew Hits Desi

```yaml
station_id: pew-hits-desi
name: "Pew Hits Desi"
languages: ["hi", "pa", "bn", "ta", "te", "mr", "gu"]
primary_genres:
  - bollywood
  - hindi_pop
  - indian_classical
  - punjabi
  - romantic
  - emotional
  - soft_melodies
  - devotional
  - indie_hindi
mood_preferences:
  romantic: 1.0
  emotional: 0.95
  soft: 0.9
  nostalgic: 0.85
  devotional: 0.6
  peaceful: 0.7
energy_range: [0.10, 0.40]  # Very low to low-medium
artist_cooldown: 3
transition_rules:
  max_bpm_delta: 15
  max_energy_delta: 0.20
  prefer_smooth_transitions: true
  avoid_abrupt_genre_shifts: true
  emotional_continuity: true
```

---

## 6. Tech Stack (tech-stack.yaml)

```yaml
# tech-stack.yaml
project: pew-hits-ai-dj
version: 1.0.0
python: "3.11"

dependencies:
  # Web Framework
  fastapi: "0.109.0"
  uvicorn: "0.27.0"
  pydantic: "2.5.3"
  pydantic-settings: "2.1.0"

  # Audio Analysis
  librosa: "0.10.2"
  numpy: "1.26.3"
  scipy: "1.11.4"
  mutagen: "1.47.0"

  # ML Models (CPU only)
  torch: "2.2.0"              # CPU build via --index-url https://download.pytorch.org/whl/cpu
  torchaudio: "2.2.0"
  onnxruntime: "1.17.0"       # For OpenL3/EfficientAT ONNX models
  scikit-learn: "1.3.2"
  xgboost: "2.0.3"
  joblib: "1.3.2"

  # Model Hub
  huggingface-hub: "0.20.3"
  safetensors: "0.4.2"

  # Database
  aiosqlite: "0.19.0"
  sqlalchemy: "2.0.25"

  # NVIDIA API
  openai: "1.12.0"            # OpenAI-compatible client for NVIDIA
  httpx: "0.26.0"
  tenacity: "8.2.3"           # Retry logic

  # File Watching
  watchdog: "3.0.0"

  # Config & Logging
  python-dotenv: "1.0.0"
  structlog: "24.1.0"
  loguru: "0.7.2"

  # Monitoring
  prometheus-client: "0.19.0"

dev_dependencies:
  pytest: "7.4.4"
  pytest-asyncio: "0.23.3"
  pytest-cov: "4.1.0"
  httpx: "0.26.0"
  black: "23.12.0"
  ruff: "0.1.15"
  mypy: "1.7.1"
  pre-commit: "3.6.0"

docker:
  base_image: "python:3.11-slim-bookworm"
  # For ARM64: python:3.11-slim-bookworm works natively
  # Install system deps: ffmpeg, libsndfile1, libmagic1

models:
  openl3:
    variant: "music_mel256_emb512"
    format: "onnx"
    source: "huggingface: facebook/openl3-music-mel256-emb512"
    size_mb: 50
    input: "16kHz mono audio, 30s clips"
    output: "512-dim embedding"

  efficientat:
    variant: "mn10_as"
    format: "onnx"
    source: "huggingface: fschmid56/EfficientAT-mn10_as"
    size_mb: 5
    input: "16kHz mono audio, 10s clips"
    output: "512-dim embedding"

  classifiers:
    genre:
      classes: 87
      source: "trained_on_mtg_jamendo"
      format: "joblib"
    mood:
      classes: ["happy", "sad", "relaxed", "aggressive", "party", "acoustic", "electronic"]
      format: "joblib"
    danceability:
      classes: ["danceable", "not_danceable"]
      format: "joblib"
    vocal_instrumental:
      classes: ["vocal", "instrumental"]
      format: "joblib"

nvidia:
  base_url: "https://integrate.api.nvidia.com/v1"
  models:
    primary_reasoning: "nvidia/nemotron-3-ultra-550b-a55b"
    fast_reasoning: "nvidia/nemotron-3.5-lightning-30b-a3b"
    embeddings: "nvidia/nemotron-3.5-embed-1b"
    reranking: "nvidia/nemotron-3.5-rerank-8b"
  limits:
    rpm: 40
    timeout_reasoning: 60
    timeout_embedding: 15
    timeout_rerank: 30
    max_retries: 3

analysis:
  workers: 4
  chunk_duration_sec: 30
  sample_rate: 16000
  max_concurrent_analysis: 2
```

---

## 7. Docker Compose (Production)

```yaml
# docker-compose.ai-dj.yml
version: "3.9"

services:
  ai-dj:
    build:
      context: .
      dockerfile: Dockerfile.ai-dj
    container_name: pew-hits-ai-dj
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 3G
        reservations:
          cpus: '0.5'
          memory: 1G
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
      - AI_DJ_HOST=0.0.0.0
      - AI_DJ_PORT=8000
      - CACHE_DIR_ENGLISH=/cache/english
      - CACHE_DIR_DESI=/cache/desi
      - DB_PATH=/data/ai_dj.db
      - MODELS_DIR=/models
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
      - NVIDIA_RPM_LIMIT=40
      - STATION_ENGLISH_PROFILE=/config/station_english.yaml
      - STATION_DESI_PROFILE=/config/station_desi.yaml
      - ANALYSIS_WORKERS=4
      - ANALYSIS_CHUNK_DURATION=30
    volumes:
      - ./pew-hits/Cache:/cache/english:ro
      - ./pew-hits-desi/Cache:/cache/desi:ro
      - ./data:/data
      - ./models:/models:ro
      - ./config:/config:ro
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 300s  # Allow 5 min for startup analysis
    depends_on: []
    networks:
      - pew-hits-network

  # Existing services (with resource limits added)
  pew-hits-radio:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 512M

  pew-hits-desi-radio:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 512M

  liquidsoap-relay:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  shoutcast-dnas:
    deploy:
      resources:
        limits:
          cpus: '0.25'
          memory: 128M

  icecast:
    deploy:
      resources:
        limits:
          cpus: '0.1'
          memory: 64M

  hermes:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

networks:
  pew-hits-network:
    external: true
    name: pew-hits-network

volumes:
  ai_dj_data:
  ai_dj_models:
```

---

## 8. Integration Adapter (pewhits Side)

### 8.1 DJ.py Modifications

```python
# Add to config/DJ.py

import httpx
from config.config import config

class AIDJClient:
    def __init__(self):
        self.base_url = getattr(config, "AI_DJ_URL", "http://ai-dj:8000/api/v1")
        self.timeout = getattr(config, "AI_DJ_TIMEOUT", 2.0)
        self.enabled = getattr(config, "AI_DJ_ENABLED", True)
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    async def get_playable_tracks(self) -> list[dict]:
        """Build playable track list for AI DJ"""
        playable = []
        
        # 1. Cached files
        for mp3_path in Path("Cache").glob("*.mp3"):
            metadata = cache_manager.extract_metadata_from_file(str(mp3_path))
            # Try to get Spotify ID from filename or tags
            spotify_id = metadata.get("spotifyID")
            if spotify_id and not spotify_id.startswith("cache_"):
                canonical_id = f"track:spotify:{spotify_id}"
            else:
                canonical_id = f"track:cache:{cache_manager.sanitize_filename(mp3_path.stem)}"
            
            playable.append({
                "canonical_id": canonical_id,
                "spotify_id": spotify_id if spotify_id and not spotify_id.startswith("cache_") else None,
                "youtube_id": metadata.get("youtube_id"),
                "title": metadata["title"],
                "artist": metadata["artist"],
                "duration_sec": metadata["durationsec"],
                "source": "cache",
                "file_path": str(mp3_path),
            })
        
        # 2. Playlist DB (Spotify imports)
        try:
            from config.PlaylistHandler import pl
            active = pl.db.get_active_playlist()
            if active:
                songs = pl.db.list_songs(active["id"], limit=5000)
                for song in songs.get("songs", []):
                    if song.get("source_id"):
                        playable.append({
                            "canonical_id": f"track:spotify:{song['source_id']}",
                            "spotify_id": song["source_id"],
                            "title": song["title"],
                            "artist": song["artist"],
                            "duration_sec": song.get("duration_sec", 180),
                            "source": "spotify_playlist",
                            "file_path": None,
                        })
        except Exception as e:
            logger.warning(f"Could not load playlist tracks: {e}")
        
        return playable
    
    async def get_recommendation(self, station: str, current_track: dict, recent_history: list) -> dict | None:
        """Call AI DJ for next track recommendation"""
        if not self.enabled:
            return None
        
        playable = await self.get_playable_tracks()
        
        payload = {
            "station": station,
            "current_track": current_track,
            "recent_history": recent_history[-5:],
            "playable_track_ids": [t["canonical_id"] for t in playable],
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/recommendations/next",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            logger.warning(f"AI DJ timeout ({self.timeout}s), falling back")
        except httpx.HTTPStatusError as e:
            logger.warning(f"AI DJ HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.warning(f"AI DJ error: {e}, falling back")
        
        return None

# In Downloader class, add:
ai_dj_client = AIDJClient()

async def get_next_song_from_AIplaylist(self):
    # ... existing setup code ...
    
    # Try AI DJ first
    if self.ai_dj_client.enabled:
        station_name = "pew-hits-english" if config.RADIO_NAME == "Pew Hits English" else "pew-hits-desi"
        
        current = self.songHandler.get_now_playing_data()
        current_track = {
            "canonical_id": current.spotifyID if current else "",
            "title": current.title if current else "",
            "artist": current.artist if current else "",
        } if current else {}
        
        recent = self.songHandler.get_history()[-5:] if current else []
        
        ai_response = await self.ai_dj_client.get_recommendation(
            station_name, current_track, recent
        )
        
        if ai_response and ai_response.get("candidates"):
            # Try each candidate in order using EXISTING streaming logic
            for candidate in ai_response["candidates"]:
                track_id = candidate["canonical_id"]
                # Convert canonical_id to something pewhits can play
                if await self.try_stream_ai_dj_candidate(track_id, candidate):
                    return self.format_for_streamer(track_id, candidate["score"], candidate["reason"])
    
    # Fallback to existing logic
    return await self.get_next_song_from_AIplaylist_legacy()

async def try_stream_ai_dj_candidate(self, canonical_id: str, candidate: dict) -> bool:
    """Try to stream a candidate using existing cache/download logic"""
    # Parse canonical_id
    # track:spotify:xxx -> use existing Spotify download flow
    # track:cache:xxx -> find in Cache/ and play directly
    # track:youtube:xxx -> try direct stream or download
    
    if canonical_id.startswith("track:cache:"):
        filename = canonical_id.replace("track:cache:", "") + ".mp3"
        cache_path = Path("Cache") / filename
        if cache_path.exists():
            adder.add_song(str(cache_path))
            # Update next_coming with AI DJ metadata
            self.update_next_coming_with_ai_metadata(candidate)
            return True
    
    elif canonical_id.startswith("track:spotify:"):
        spotify_id = canonical_id.replace("track:spotify:", "")
        # Trigger download using existing flow
        self.requestHandler.add_spotify_request(spotify_id, "AI DJ", "AutoDJ")
        # Wait briefly for download to start
        await asyncio.sleep(2)
        if self.songHandler.next_coming_file_exists(spotify_id):
            return True
    
    elif canonical_id.startswith("track:youtube:"):
        youtube_id = canonical_id.replace("track:youtube:", "")
        youtube_url = f"https://youtube.com/watch?v={youtube_id}"
        # Try direct stream first
        stream_info = await self.unified_downloader.resolve_youtube_stream(youtube_url, candidate["title"])
        if stream_info and stream_info.get("direct_url"):
            await self.queue_direct_stream_result(...)
            return True
        # Fallback to download
        self.requestHandler.add_youtube_request(youtube_url, "AI DJ", "AutoDJ")
        await asyncio.sleep(2)
        if self.songHandler.next_coming_file_exists(f"youtube_{youtube_id}"):
            return True
    
    return False
```

---

## 9. Security Considerations

| Aspect | Implementation |
|--------|----------------|
| **NVIDIA API Key** | Stored in Docker secret / env var, never in code or logs |
| **API Authentication** | Optional: Add API key header for AI DJ endpoints |
| **Rate Limiting** | Client-side token bucket (40 RPM) + server-side per-station |
| **Input Validation** | Pydantic models on all endpoints, canonical_id regex |
| **File Access** | Read-only cache mounts, no write access to pewhits cache |
| **Error Sanitization** | No stack traces in responses, generic error messages |
| **CORS** | Restrict to pewhits container IPs only |
| **Secrets Rotation** | NVIDIA key rotated via Docker secret update |

---

## 10. Cache Synchronization Design

```python
# In AI DJ service - services/cache_sync.py

class CacheSyncService:
    def __init__(self, db, cache_dirs: list[Path]):
        self.db = db
        self.cache_dirs = cache_dirs
        self.observer = None
    
    async def start(self):
        # 1. Initial reconciliation
        await self.reconcile_all()
        
        # 2. Start file watchers
        self.observer = Observer()
        for cache_dir in self.cache_dirs:
            handler = CacheEventHandler(self.db)
            self.observer.schedule(handler, str(cache_dir), recursive=False)
        self.observer.start()
        
        # 3. Periodic reconciliation (every 5 min)
        asyncio.create_task(self.periodic_reconcile())
    
    async def reconcile_all(self):
        for cache_dir in self.cache_dirs:
            await self.reconcile_dir(cache_dir)
    
    async def reconcile_dir(self, cache_dir: Path):
        # Get actual files
        actual_files = set()
        for mp3 in cache_dir.glob("*.mp3"):
            canonical_id = f"track:cache:{cache_manager.sanitize_filename(mp3.stem)}"
            actual_files.add(canonical_id)
        
        # Get DB entries for this cache dir
        db_entries = set(await self.db.get_cache_canonical_ids(str(cache_dir)))
        
        # Missing in cache -> mark evicted
        for missing in db_entries - actual_files:
            await self.db.update_cache_status(missing, "evicted")
            logger.warning(f"Cache evicted detected: {missing}")
        
        # New in cache -> analyze
        for new in actual_files - db_entries:
            mp3_path = cache_dir / f"{new.replace('track:cache:', '')}.mp3"
            await self.analyze_new_file(mp3_path, new)
    
    async def periodic_reconcile(self):
        while True:
            await asyncio.sleep(300)  # 5 minutes
            await self.reconcile_all()
```

---

## 11. Phase 2 Deliverables Checklist

| Deliverable | Status |
|-------------|--------|
| ✅ `ARCHITECTURE.md` | This document |
| ✅ `tech-stack.yaml` | Above |
| ✅ `docker-compose.ai-dj.yml` | Above |
| ✅ API Contracts (OpenAPI) | Above |
| ✅ Station Profiles | Above |
| ✅ Data Models (SQLite) | Above |
| ✅ Recommendation Algorithm | Above |
| ✅ Integration Adapter Code | Above |
| ✅ Cache Sync Design | Above |
| ⏳ Security Review | Pending (Security Engineer) |
| ⏳ Code Review | Pending (Code Reviewer) |

---

## 12. Next Steps (Phase 3 - Implementation)

1. **Security Engineer Review** of this architecture
2. **Code Reviewer Review** of data models and API contracts
3. **Incorporate feedback** → Finalize `ARCHITECTURE.md`
4. **Begin Phase 3**: Build catalogue analysis, mood/energy classification, recommendation ranking, NVIDIA integration, REST APIs, station profiles, fallback mechanisms

---

**Ready for Security Engineer and Code Reviewer feedback on this architecture?**