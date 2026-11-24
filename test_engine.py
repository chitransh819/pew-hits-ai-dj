"""
Test suite for AI DJ Recommendation Engine
Run with: pytest test_engine.py -v
"""

import pytest
import asyncio
import tempfile
import os
import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from ai_dj.db.database import Database
from ai_dj.recommendation.engine import RecommendationEngine, NVIDIAClient, StationConfig, Candidate
from ai_dj.analysis.analyzer import AudioAnalyzer, AudioFeatures


# ===== Fixtures =====

@pytest.fixture
def temp_db():
    """Create temporary database for testing.
    
    Pattern: sync def fixture, initialize with asyncio.run(), close with os.unlink only.
    Never call db.close() — Database uses async with context manager pattern.
    """
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    db = Database(db_path)
    asyncio.run(db.initialize())
    yield db
    os.unlink(db_path)


# ===== Station Configurations =====

@pytest.fixture
def station_configs():
    """Test station configurations matching YAML station_english.yaml and station_desi.yaml."""
    english_config = StationConfig(
        station_id="pew-hits-english",
        name="Pew Hits English",
        language="en",
        target_audience="english",
        weights={"w_mood": 0.25, "w_energy": 0.20, "w_tempo": 0.15, "w_genre": 0.15, "w_language": 0.10, "w_artist": 0.05, "w_transition": 0.05, "w_semantic": 0.05},
        transition_rules={"max_bpm_delta": 20, "max_energy_delta": 0.25},
        filters={
            "allowed_languages": ["en"],
            "allowed_genres": ["pop", "soft_pop", "acoustic"],
            "energy_range": [0.15, 0.50],
            "tempo_range": [60, 180],
        },
        artist_cooldown_minutes=15,
        recent_history_exclusion_count=3,
        diversity={"min_artist_gap": 3, "max_same_genre": 2},
        nvidia_rerank={"enabled": False, "top_k": 10},
    )
    
    desi_config = StationConfig(
        station_id="pew-hits-desi",
        name="Pew Hits Desi",
        language="hi",
        target_audience="desi",
        weights={"w_mood": 0.30, "w_energy": 0.20, "w_tempo": 0.10, "w_genre": 0.15, "w_language": 0.10, "w_artist": 0.05, "w_transition": 0.05, "w_semantic": 0.05},
        transition_rules={"max_bpm_delta": 15, "max_energy_delta": 0.20},
        filters={
            "allowed_languages": ["hi", "pa", "bn"],
            "allowed_genres": ["bollywood", "romantic", "emotional"],
            "energy_range": [0.10, 0.40],
            "tempo_range": [60, 160],
        },
        artist_cooldown_minutes=15,
        recent_history_exclusion_count=3,
        diversity={"min_artist_gap": 3, "max_same_genre": 2},
        nvidia_rerank={"enabled": False, "top_k": 10},
    )
    
    return {
        "pew-hits-english": english_config,
        "pew-hits-desi": desi_config,
    }


# ===== Core Database Tests =====

def test_database_initialization():
    """Test database schema creation."""
    import tempfile
    import asyncio
    
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    db = Database(db_path)
    asyncio.run(db.initialize())
    stats = asyncio.run(db.get_stats())
    asyncio.run(db.close())  # OK: called inside test, not fixture
    os.unlink(db_path)
    
    assert "track_identity" in stats
    assert "track_features" in stats
    assert "playback_history" in stats
    assert "recommendation_history" in stats


def test_track_identity_crud():
    """Test track identity create/read."""
    import tempfile
    import asyncio
    
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    db = Database(db_path)
    asyncio.run(db.initialize())
    
    canonical_id = "track:spotify:test123"
    data = {
        "canonical_id": canonical_id,
        "spotify_id": "test123",
        "title": "Test Song",
        "artist": "Test Artist",
        "duration_sec": 180,
    }
    
    result = asyncio.run(db.upsert_track_identity(data))
    assert result == canonical_id
    
    retrieved = asyncio.run(db.get_track_identity(canonical_id))
    assert retrieved is not None
    assert retrieved["title"] == "Test Song"
    assert retrieved["artist"] == "Test Artist"
    
    asyncio.run(db.close())
    os.unlink(db_path)


def test_track_features_crud():
    """Test track features create/read.
    
    Pattern: upsert_track_features expects exactly 29 feature columns.
    Ensure features dict includes ALL required keys.
    """
    import tempfile
    import asyncio
    
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    db = Database(db_path)
    asyncio.run(db.initialize())
    
    canonical_id = "track:spotify:test456"
    asyncio.run(db.upsert_track_identity({
        "canonical_id": canonical_id,
        "title": "Test",
        "artist": "Artist",
    }))
    
    # ALL 29 required features for the DB schema
    features = {
        "bpm": 120.5,
        "key": "C major",
        "energy": 0.7,
        "danceability": 0.8,
        "genre_top": "pop",
        "genre_scores": {"pop": 0.9},
        "mood_tags": ["happy"],
        "mood_scores": {"happy": 1.0},
        "beats": [0.5, 1.0],
        "onset_times": [0.5, 1.0],
        "spectral_centroid": 2000.0,
        "spectral_rolloff": 4000.0,
        "spectral_bandwidth": 1500.0,
        "spectral_contrast": [10, 20, 30],
        "chroma": [0.1] * 12,
        "loudness_lufs": -12.0,
        "openl3_embedding": bytes(64),
        "efficientat_embedding": bytes(64),
        "valence": 0.6,
        "acousticness": 0.3,
        "instrumentalness": 0.1,
        "vocal_presence": 0.8,
        "cache_status": "available",
        "analysis_version": "1.0",
        "file_mtime": 1234567890,
        "file_size": 4567890,
    }
    
    asyncio.run(db.upsert_track_features(canonical_id, features))
    retrieved = asyncio.run(db.get_track_features(canonical_id))
    
    assert retrieved is not None
    assert retrieved["bpm"] == 120.5
    assert retrieved["key"] == "C major"
    assert retrieved["genre_scores"]["pop"] == 0.9
    
    asyncio.run(db.close())
    os.unlink(db_path)


def test_station_profiles():
    """Test station profile storage."""
    import tempfile
    import asyncio
    
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    db = Database(db_path)
    asyncio.run(db.initialize())
    
    profile = {
        "station_id": "pew-hits-english",
        "name": "Pew Hits English",
        "languages": ["en"],
        "primary_genres": ["pop"],
        "mood_preferences": {"relaxed": 1.0},
        "energy_range": [0.1, 0.5],
        "artist_cooldown": 3,
        "transition_rules": {"max_bpm_delta": 20, "max_energy_delta": 0.25},
        "scoring_weights": {"w_mood": 0.25, "w_energy": 0.20, "w_tempo": 0.15, "w_genre": 0.15, "w_language": 0.10, "w_artist": 0.05, "w_transition": 0.05, "w_semantic": 0.05},
    }
    
    asyncio.run(db.upsert_station_profile(profile))
    retrieved = asyncio.run(db.get_station_profile("pew-hits-english"))
    
    assert retrieved is not None
    assert retrieved["station_id"] == "pew-hits-english"
    assert retrieved["languages"] == ["en"]
    
    asyncio.run(db.close())
    os.unlink(db_path)


# ===== Recommendation Engine Tests =====

@pytest.mark.asyncio
async def test_recommendation_engine_basic(temp_db, station_configs):
    """Test basic recommendation engine functionality."""
    engine = RecommendationEngine(
        station_config=station_configs["pew-hits-english"],
        nvidia_client=None,
    )
    
    # Add some test tracks
    for i in range(5):
        cid = f"track:spotify:test{i}"
        await temp_db.upsert_track_identity({
            "canonical_id": cid,
            "spotify_id": f"test{i}",
            "title": f"Song {i}",
            "artist": f"Artist {i}",
            "duration_sec": 180 + i * 10,
            "language": "en",
        })
        await temp_db.upsert_track_features(cid, {
            "bpm": 120 + i * 5,
            "energy": 0.5 + i * 0.05,
            "genre_top": "pop",
            "genre_scores": {"pop": 0.9},
            "mood_tags": ["relaxed"],
            "mood_scores": {"relaxed": 0.9},
            "danceability": 0.6,
        })
    
    # Get recommendations
    result = await engine.get_recommendations(
        station_id="pew-hits-english",
        current_track={"canonical_id": "track:spotify:test0"},
        recent_history=[],
        playable_ids=[f"track:spotify:test{i}" for i in range(1, 5)],
        limit=3,
    )
    
    assert "candidates" in result
    assert len(result["candidates"]) <= 3
    assert "explanation" in result
    assert "confidence" in result
    assert result["reasoning_source"] == "local"


@pytest.mark.asyncio
async def test_scoring_functions(station_configs):
    """Test individual scoring functions."""
    from ai_dj.recommendation.engine import RecommendationEngine
    
    engine = RecommendationEngine(
        station_config=station_configs["pew-hits-english"],
        nvidia_client=None,
    )
    
    # Test cosine similarity
    vec1 = {"happy": 0.8, "sad": 0.1}
    vec2 = {"happy": 0.7, "sad": 0.2}
    sim = engine._cosine_similarity(vec1, vec2)
    assert 0 <= sim <= 1
    
    # Test gaussian similarity
    sim = engine._gaussian_similarity(0.5, 0.5, 0.15)
    assert sim == 1.0  # Exact match
    
    sim = engine._gaussian_similarity(0.3, 0.5, 0.15)
    assert 0 < sim < 1
    
    # Test tempo compatibility
    sim = engine._tempo_compatibility(120, 120)
    assert sim == 1.0
    
    # Harmonic relationship
    sim = engine._tempo_compatibility(120, 60)  # half tempo
    assert sim >= 0.8


def test_nvidia_client_rate_limiting():
    """Test NVIDIA client rate limiter."""
    client = NVIDIAClient(api_key="test", rate_limit_rpm=40)
    
    import asyncio
    
    async def test_rate_limit():
        for _ in range(5):
            await client._wait_for_rate_limit()
    
    asyncio.run(test_rate_limit())


def test_audio_features_dataclass():
    """Test AudioFeatures dataclass."""
    import numpy as np
    
    features = AudioFeatures(
        bpm=120.0,
        beats=np.array([0.5, 1.0, 1.5]),
        onset_times=np.array([0.5, 1.0]),
        spectral_centroid=2000.0,
        spectral_rolloff=4000.0,
        spectral_bandwidth=1500.0,
        spectral_contrast=np.array([10, 20, 30]),
        chroma=np.array([0.1] * 12),
        key="C major",
        loudness_lufs=-12.0,
        openl3_embedding=np.random.randn(512),
        efficientat_embedding=np.random.randn(512),
        genre_top="pop",
        genre_scores={"pop": 0.9},
        mood_tags=["happy"],
        mood_scores={"happy": 0.9},
        danceability=0.8,
        energy=0.7,
        valence=0.6,
        acousticness=0.3,
        instrumentalness=0.1,
        vocal_presence=0.8,
    )
    
    assert features.bpm == 120.0
    assert features.key == "C major"
    assert len(features.openl3_embedding) == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
