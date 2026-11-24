"""
FastAPI Routes for AI DJ Service
Single endpoint: POST /api/v1/recommendations/next
Station sends only: station, current_track, recent_history (last 5)
AI DJ auto-decides: local cached song OR new song from Spotify
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ai_dj.analysis.analyzer import AnalysisWorker, AudioAnalyzer
from ai_dj.db.database import Database
from ai_dj.recommendation.engine import RecommendationEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


# ===== Pydantic Models ====

class TrackContext(BaseModel):
    canonical_id: str = Field(..., pattern=r"^track:(spotify|youtube|soundcloud|jiosaavn|cache):.+")
    title: str
    artist: str
    album: str = ""
    duration_sec: int = 0
    played_at: str | None = None
    source: str | None = None


class RecommendationRequest(BaseModel):
    station: str = Field(..., pattern=r"^(pew-hits-english|pew-hits-desi)$")
    current_track: TrackContext
    recent_history: list[TrackContext] = Field(default_factory=list, max_length=5)


class RecommendationResponse(BaseModel):
    canonical_id: str
    title: str
    artist: str
    album: str = ""
    duration_sec: int = 0
    genre: str = ""
    language: str = "english"
    source: str  # "local" or "spotify"
    source_track_id: str | None = None
    score: float
    reason: str
    confidence: float
    preview_url: str | None = None
    external_url: str | None = None
    release_date: str | None = None


class TrackAnalysisRequest(BaseModel):
    file_path: str
    force: bool = False


class TrackAnalysisResponse(BaseModel):
    canonical_id: str
    status: str
    features: dict[str, Any] | None = None
    duration_ms: int


class HistoryUpdateRequest(BaseModel):
    canonical_id: str
    played_at: str
    source: str
    duration_sec: int = 0


class StationStatusResponse(BaseModel):
    station: str
    status: str
    tracks_analyzed: int
    cache_status: str
    last_reconciliation: str | None = None
    profile: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    tracks_analyzed: int
    uptime_seconds: int
    models_loaded: bool
    db_connected: bool


# ===== Dependencies ====

# These will be set by main.py on startup
_db: Database | None = None
_engines: dict[str, RecommendationEngine] | None = None
_analyzer: AudioAnalyzer | None = None
_worker: AnalysisWorker | None = None
_startup_time: datetime = datetime.now()


def get_db() -> Database:
    if _db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return _db


def get_engine(station: str) -> RecommendationEngine:
    if _engines is None or station not in _engines:
        raise HTTPException(status_code=503, detail=f"Recommendation engine for {station} not initialized")
    return _engines[station]


def get_analyzer() -> AudioAnalyzer:
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Audio analyzer not initialized")
    return _analyzer


def get_worker() -> AnalysisWorker:
    if _worker is None:
        raise HTTPException(status_code=503, detail="Analysis worker not initialized")
    return _worker


def set_dependencies(db: Database, engines: dict[str, RecommendationEngine],
                     analyzer: AudioAnalyzer, worker: AnalysisWorker):
    global _db, _engines, _analyzer, _worker
    _db = db
    _engines = engines
    _analyzer = analyzer
    _worker = worker


# ===== Endpoints ====

@router.post("/recommendations/next", response_model=RecommendationResponse)
async def get_next_recommendation(
    request: RecommendationRequest,
):
    """
    Get next track recommendation.
    Station sends: station, current_track, recent_history (last 5).
    AI DJ auto-decides: local cached song OR new song from Spotify.
    """
    try:
        engine = get_engine(request.station)
        result = await engine.get_next_track(
            station_id=request.station,
            current_track=request.current_track.model_dump(),
            recent_history=[h.model_dump() for h in request.recent_history],
        )
        return RecommendationResponse(**result)
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tracks/analyze", response_model=TrackAnalysisResponse)
async def analyze_track(
    request: TrackAnalysisRequest,
    background_tasks: BackgroundTasks,
    analyzer: AudioAnalyzer = Depends(get_analyzer),
    worker: AnalysisWorker = Depends(get_worker),
    db: Database = Depends(get_db),
):
    """Analyze a new track (incremental)"""
    import time
    start = time.time()

    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Generate canonical ID from file path
    canonical_id = f"track:cache:{file_path.stem}"

    # Check if already analyzed and unchanged
    if not request.force:
        stat = file_path.stat()
        existing = await db.get_track_features(canonical_id)
        if (existing and existing.get("file_mtime") == int(stat.st_mtime)
            and existing.get("file_size") == stat.st_size):
            return TrackAnalysisResponse(
                canonical_id=canonical_id,
                status="skipped_unchanged",
                features=existing,
                duration_ms=int((time.time() - start) * 1000),
            )

    # Run analysis in background
    background_tasks.add_task(worker.queue_file, file_path, canonical_id, request.force)

    return TrackAnalysisResponse(
        canonical_id=canonical_id,
        status="queued",
        duration_ms=int((time.time() - start) * 1000),
    )


@router.post("/stations/{station}/history")
async def update_history(
    station: str,
    request: HistoryUpdateRequest,
    db: Database = Depends(get_db),
):
    """Update playback history"""
    if station not in ["pew-hits-english", "pew-hits-desi"]:
        raise HTTPException(status_code=400, detail="Invalid station")

    try:
        await db.add_playback(
            station_id=station,
            canonical_id=request.canonical_id,
            played_at=request.played_at,
            source=request.source,
            duration_sec=request.duration_sec,
        )
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"History update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stations/{station}/recommendations")
async def get_recommendation_history(
    station: str,
    limit: int = Query(50, ge=1, le=200),
    db: Database = Depends(get_db),
):
    """Get recent recommendation history"""
    return {"station": station, "recommendations": []}


@router.get("/stations/{station}/status", response_model=StationStatusResponse)
async def get_station_status(
    station: str,
    db: Database = Depends(get_db),
):
    """Get station status and config"""
    if station not in ["pew-hits-english", "pew-hits-desi"]:
        raise HTTPException(status_code=400, detail="Invalid station")

    profile = await db.get_station_profile(station)
    stats = await db.get_stats()

    return StationStatusResponse(
        station=station,
        status="ready",
        tracks_analyzed=stats.get("track_features", 0),
        cache_status="healthy",
        profile=profile or {},
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: Database = Depends(get_db),
):
    """Health check endpoint"""
    stats = await db.get_stats()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        tracks_analyzed=stats.get("track_features", 0),
        uptime_seconds=int((datetime.now() - _startup_time).total_seconds()),
        models_loaded=True,
        db_connected=True,
    )
