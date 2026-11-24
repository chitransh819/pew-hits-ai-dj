"""
FastAPI Routes for AI DJ Service
All REST API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ai_dj.db.database import Database
from ai_dj.recommendation.engine import RecommendationEngine, NVIDIAClient
from ai_dj.analysis.analyzer import AudioAnalyzer, AnalysisWorker
from ai_dj.services.discovery import DiscoveryService, DiscoveredTrack, DiscoverySource
from ai_dj.services.spotify_client import SpotifyClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


# ===== Pydantic Models =====

class TrackContext(BaseModel):
    canonical_id: str = Field(..., pattern=r"^track:(spotify|youtube|soundcloud|jiosaavn|cache):.+")
    title: str
    artist: str
    album: str = ""
    duration_sec: int = 0
    played_at: Optional[str] = None
    source: Optional[str] = None


class RecommendationRequest(BaseModel):
    station: str = Field(..., pattern=r"^(pew-hits-english|pew-hits-desi)$")
    current_track: TrackContext
    recent_history: List[TrackContext] = Field(default_factory=list, max_length=10)
    playable_track_ids: List[str] = Field(default_factory=list, max_length=5000)


class CandidateResponse(BaseModel):
    canonical_id: str
    score: float
    reason: str
    metadata: Dict[str, Any]


class RecommendationResponse(BaseModel):
    candidates: List[CandidateResponse]
    explanation: str
    confidence: float
    reasoning_source: str
    latency_ms: int


class TrackAnalysisRequest(BaseModel):
    file_path: str
    force: bool = False


class TrackAnalysisResponse(BaseModel):
    canonical_id: str
    status: str
    features: Optional[Dict[str, Any]] = None
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
    last_reconciliation: Optional[str] = None
    profile: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    tracks_analyzed: int
    uptime_seconds: int
    models_loaded: bool
    db_connected: bool


# ===== Discovery Models =====

class DiscoveryRequest(BaseModel):
    station: str = Field(..., pattern=r"^(pew-hits-english|pew-hits-desi)$")
    language: str = Field(default="english")
    genres: list[str] = Field(default_factory=list)
    mood_keywords: list[str] = Field(default_factory=list)
    limit_per_source: int = Field(default=10, ge=1, le=50)


class DiscoveryResponse(BaseModel):
    station: str
    total_found: int
    tracks: list[dict[str, Any]]
    sources_used: list[str]


# ===== Dependencies ====

# These will be set by main.py on startup
_db: Optional[Database] = None
_engine: Optional[RecommendationEngine] = None
_analyzer: Optional[AudioAnalyzer] = None
_worker: Optional[AnalysisWorker] = None
_discovery: Optional[DiscoveryService] = None
_spotify: Optional[SpotifyClient] = None
_startup_time: datetime = datetime.now()


def get_db() -> Database:
    if _db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return _db


def get_engine() -> RecommendationEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    return _engine


def get_analyzer() -> AudioAnalyzer:
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Audio analyzer not initialized")
    return _analyzer


def get_worker() -> AnalysisWorker:
    if _worker is None:
        raise HTTPException(status_code=503, detail="Analysis worker not initialized")
    return _worker


def get_discovery() -> DiscoveryService:
    if _discovery is None:
        raise HTTPException(status_code=503, detail="Discovery service not initialized")
    return _discovery


def get_spotify() -> SpotifyClient:
    if _spotify is None:
        raise HTTPException(status_code=503, detail="Spotify client not initialized")
    return _spotify


def set_dependencies(db: Database, engine: RecommendationEngine, 
                     analyzer: AudioAnalyzer, worker: AnalysisWorker,
                     discovery: Optional[DiscoveryService] = None,
                     spotify: Optional[SpotifyClient] = None):
    global _db, _engine, _analyzer, _worker, _discovery, _spotify
    _db = db
    _engine = engine
    _analyzer = analyzer
    _worker = worker
    _discovery = discovery
    _spotify = spotify


# ===== Endpoints =====

@router.post("/recommendations/next", response_model=RecommendationResponse)
async def get_next_recommendation(
    request: RecommendationRequest,
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get next track recommendation"""
    try:
        result = await engine.get_recommendations(
            station_id=request.station,
            current_track=request.current_track.model_dump(),
            recent_history=[h.model_dump() for h in request.recent_history],
            playable_ids=request.playable_track_ids,
            limit=10,
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
    # Implementation would query recommendation_history table
    # For now return empty
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
        models_loaded=True,  # Would check actual model loading
        db_connected=True,
    )


# ===== Discovery Endpoints =====

@router.post("/discovery/new", response_model=DiscoveryResponse)
async def discover_new_songs(
    request: DiscoveryRequest,
    discovery: DiscoveryService = Depends(get_discovery),
):
    """Discover new song metadata from Spotify, YouTube, JioSaavn, and free sources.
    Returns metadata only - no audio download. Station handles download via existing fallback chain."""
    try:
        tracks = await discovery.discover_new_songs(
            station_id=request.station,
            language=request.language,
            genres=request.genres,
            mood_keywords=request.mood_keywords,
            limit_per_source=request.limit_per_source,
        )

        # Cache results
        discovery.cache_discovery(request.station, tracks)

        sources_used = list(set(t.source.value for t in tracks))

        return DiscoveryResponse(
            station=request.station,
            total_found=len(tracks),
            tracks=[t.to_dict() for t in tracks],
            sources_used=sources_used,
        )
    except Exception as e:
        logger.error(f"Discovery error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discovery/cached/{station}")
async def get_cached_discovery(
    station: str,
    discovery: DiscoveryService = Depends(get_discovery),
):
    """Get previously cached discovery results for a station."""
    if station not in ("pew-hits-english", "pew-hits-desi"):
        raise HTTPException(status_code=400, detail="Invalid station")
    tracks = discovery.get_cached_discovery(station)
    if tracks is None:
        return {"station": station, "cached": False, "tracks": []}
    return {
        "station": station,
        "cached": True,
        "tracks": [t.to_dict() for t in tracks],
    }


from pathlib import Path