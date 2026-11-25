"""
AI DJ Service - Main Entry Point
Initializes all components and starts the FastAPI server
"""

import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from ai_dj.analysis.analyzer import AnalysisWorker, AudioAnalyzer
from ai_dj.api.routes import router, set_dependencies
from ai_dj.db.database import Database
from ai_dj.recommendation.engine import (
    NVIDIAClient,
    RecommendationEngine,
    create_engine_from_config,
)
from ai_dj.services.cache_sync import create_cache_sync_service
from ai_dj.services.discovery import DiscoveryService
from ai_dj.services.spotify_client import SpotifyClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai_dj")


# Global components
_db: Database = None
_engines: dict[str, RecommendationEngine] = {}
_analyzer: AudioAnalyzer = None
_worker: AnalysisWorker = None
_cache_sync = None
_discovery: DiscoveryService | None = None
_spotify: SpotifyClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global _db, _engines, _analyzer, _worker, _cache_sync
    
    logger.info("=" * 60)
    logger.info("Starting Pew Hits AI DJ Service")
    logger.info("=" * 60)
    
    # 1. Initialize Database
    db_path = Path("/data/ai_dj.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _db = Database(str(db_path))
    await _db.initialize()
    logger.info("Database initialized")
    
    # 2. Load Station Profiles
    await load_station_profiles(_db)
    logger.info("Station profiles loaded")
    
    # 3. Initialize Audio Analyzer
    models_dir = Path("/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    _analyzer = AudioAnalyzer(
        models_dir=models_dir,
        sample_rate=16000,
        chunk_duration=30,
        max_workers=4,
    )
    logger.info("Audio analyzer initialized")
    
    # 4. Initialize Spotify Client (for music discovery metadata)
    spotify_client = None
    spotify_client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    spotify_client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if spotify_client_id and spotify_client_secret:
        spotify_client = SpotifyClient(
            client_id=spotify_client_id,
            client_secret=spotify_client_secret,
            base_url=os.environ.get("SPOTIFY_BASE_URL", "https://api.spotify.com/v1"),
            timeout_seconds=int(os.environ.get("SPOTIFY_TIMEOUT", "10")),
        )
        logger.info("Spotify client initialized")
    else:
        logger.warning("SPOTIFY_CLIENT_ID/SECRET not set - Spotify discovery disabled")

    # 5. Initialize Discovery Service
    _discovery = DiscoveryService(
        spotify_client=spotify_client,
    )
    logger.info("Discovery service initialized")

    # 6. Initialize NVIDIA Client
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    if nvidia_api_key:
        NVIDIAClient(
            api_key=nvidia_api_key,
            base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            rate_limit_rpm=int(os.environ.get("NVIDIA_RPM_LIMIT", "40")),
        )
        logger.info("NVIDIA client initialized")
    else:
        logger.warning("NVIDIA_API_KEY not set - running in local-only mode")
    
    # 5. Initialize Recommendation Engines (one per station)
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    _engines = {}
    for station_id, config_path in [
        ("pew-hits-english", Path("/config/station_english.yaml")),
        ("pew-hits-desi", Path("/config/station_desi.yaml")),
    ]:
        if config_path.exists():
            _engines[station_id] = await create_engine_from_config(
                station_config_path=config_path,
                db=_db,
                discovery_service=_discovery,
                spotify_client=_spotify,
                nvidia_api_key=nvidia_api_key,
            )
            logger.info(f"Recommendation engine initialized for {station_id}")
    logger.info("Recommendation engines initialized")
    
    # 6. Initialize Analysis Worker
    cache_dirs = [
        Path("/cache/english"),
        Path("/cache/desi"),
    ]
    for d in cache_dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    _worker = AnalysisWorker(_analyzer, _db, cache_dirs)
    await _worker.start()
    logger.info("Analysis worker started")
    
    # 7. Start Cache Sync Service
    _cache_sync = await create_cache_sync_service(
        db=_db,
        cache_dirs=cache_dirs,
        analyzer=_analyzer,
        worker=_worker,
        reconcile_interval=300,
    )
    logger.info("Cache sync service started")
    
    # 8. Run Initial Cache Analysis (background)
    asyncio.create_task(run_initial_analysis(_analyzer, _worker, _db, cache_dirs))
    logger.info("Initial cache analysis started in background")
    
    # 9. Set dependencies for routes
    set_dependencies(_db, _engines, _analyzer, _worker)
    
    logger.info("=" * 60)
    logger.info("AI DJ Service ready!")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI DJ Service...")

    if _spotify:
        await _spotify.close()

    if _cache_sync:
        await _cache_sync.stop()
    
    if _worker:
        await _worker.stop()
    
    logger.info("Shutdown complete")


async def load_station_profiles(db: Database):
    """Load station profiles from YAML configs into database"""
    import yaml
    
    profiles = [
        Path("/config/station_english.yaml"),
        Path("/config/station_desi.yaml"),
    ]
    
    for profile_path in profiles:
        if profile_path.exists():
            with open(profile_path) as f:
                profile = yaml.safe_load(f)
            await db.upsert_station_profile(profile)


async def load_station_profiles_from_db(db: Database) -> dict[str, dict]:
    """Load station profiles from database"""
    profiles = {}
    for station_id in ["pew-hits-english", "pew-hits-desi"]:
        profile = await db.get_station_profile(station_id)
        if profile:
            profiles[station_id] = profile
    return profiles


async def run_initial_analysis(analyzer: AudioAnalyzer, worker: AnalysisWorker, 
                                db: Database, cache_dirs: list[Path]):
    """Run initial analysis of all cached tracks"""
    logger.info("[INITIAL_ANALYSIS] Starting...")
    
    total_files = 0
    analyzed = 0
    skipped = 0
    
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        
        mp3_files = list(cache_dir.glob("*.mp3"))
        total_files += len(mp3_files)
        
        for mp3_path in mp3_files:
            canonical_id = f"track:cache:{mp3_path.stem}"
            
            # Check if already analyzed
            stat = mp3_path.stat()
            existing = await db.get_track_features(canonical_id)
            if (existing and existing.get("file_mtime") == int(stat.st_mtime) 
                and existing.get("file_size") == stat.st_size):
                skipped += 1
                continue
            
            # Queue for analysis
            await worker.queue_file(mp3_path, canonical_id, force=False)
            analyzed += 1
            
            # Log progress
            if (analyzed + skipped) % 50 == 0:
                logger.info(f"[INITIAL_ANALYSIS] Progress: {analyzed + skipped}/{total_files}")
    
    logger.info(f"[INITIAL_ANALYSIS] Queued {analyzed} new, skipped {skipped} unchanged")


# API Key Middleware
class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow health check, docs, and status endpoints without auth
        exempt_paths = [
            "/health", "/api/v1/health",
            "/docs", "/openapi.json", "/redoc",
            "/api/v1/stations/",
        ]
        if request.url.path in exempt_paths or request.url.path.startswith("/api/v1/stations/"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        expected_key = os.environ.get("AI_DJ_API_KEY")

        if expected_key and api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return await call_next(request)


# Create FastAPI app
app = FastAPI(
    title="Pew Hits AI DJ",
    version="1.0.0",
    description="Intelligent music recommendation engine for Pew Hits radio stations",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Add API key middleware (after routes so it wraps everything)
app.add_middleware(APIKeyMiddleware)


# Signal handlers for graceful shutdown
def setup_signal_handlers():
    loop = asyncio.get_event_loop()
    
    def shutdown():
        logger.info("Received shutdown signal")
        asyncio.create_task(shutdown_async())
    
    async def shutdown_async():
        if _cache_sync:
            await _cache_sync.stop()
        if _worker:
            await _worker.stop()
        loop.stop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown)


if __name__ == "__main__":
    import os
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Run server
    uvicorn.run(
        "ai_dj.main:app",
        host=os.environ.get("AI_DJ_HOST", "0.0.0.0"),
        port=int(os.environ.get("AI_DJ_PORT", "8000")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        access_log=True,
    )