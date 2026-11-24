"""
Database layer for AI DJ Service
SQLite with async support via aiosqlite
"""

import aiosqlite
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS track_identity (
    canonical_id TEXT PRIMARY KEY,
    source TEXT,
    title TEXT,
    artist TEXT,
    album TEXT,
    duration_sec INTEGER,
    language TEXT,
    genre TEXT,
    cache_filename TEXT,
    added_at TEXT
);

CREATE TABLE IF NOT EXISTS track_features (
    canonical_id TEXT PRIMARY KEY,
    bpm REAL,
    key TEXT,
    energy REAL,
    valence REAL,
    danceability REAL,
    acousticness REAL,
    instrumentalness REAL,
    loudness_lufs REAL,
    spectral_centroid REAL,
    file_mtime INTEGER,
    file_size INTEGER,
    openl3_embedding BLOB,
    efficientat_embedding BLOB,
    genre_top TEXT,
    genre_scores TEXT,
    mood_tags TEXT,
    mood_scores TEXT,
    analysis_version TEXT,
    analyzed_at REAL,
    FOREIGN KEY (canonical_id) REFERENCES track_identity(canonical_id)
);

CREATE TABLE IF NOT EXISTS playback_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT,
    canonical_id TEXT,
    played_at TEXT,
    source TEXT,
    duration_sec INTEGER,
    FOREIGN KEY (station_id) REFERENCES station_profiles(station_id)
);

CREATE TABLE IF NOT EXISTS station_profiles (
    station_id TEXT PRIMARY KEY,
    name TEXT,
    language TEXT,
    target_audience TEXT,
    config TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS recommendation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT,
    canonical_id TEXT,
    score REAL,
    source TEXT,
    recommended_at TEXT,
    FOREIGN KEY (station_id) REFERENCES station_profiles(station_id)
);
"""


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def initialize(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.executescript(SCHEMA)
            await conn.commit()
        logger.info("Database initialized")

    async def upsert_track_identity(
        self,
        canonical_id: str,
        source: str,
        title: str,
        artist: str,
        album: str = "",
        duration_sec: int = 0,
        language: str = "english",
        genre: str = "",
        cache_filename: Optional[str] = None,
    ):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO track_identity
                   (canonical_id, source, title, artist, album, duration_sec, language, genre, cache_filename, added_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (canonical_id, source, title, artist, album, duration_sec, language, genre, cache_filename, datetime.utcnow().isoformat()),
            )
            await conn.commit()

    async def upsert_track_features(self, canonical_id: str, features: Dict[str, Any]):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO track_features
                   (canonical_id, bpm, key, energy, valence, danceability, acousticness,
                    instrumentalness, loudness_lufs, spectral_centroid, file_mtime, file_size,
                    openl3_embedding, efficientat_embedding, genre_top, genre_scores,
                    mood_tags, mood_scores, analysis_version, analyzed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    canonical_id,
                    features.get("bpm"),
                    features.get("key"),
                    features.get("energy"),
                    features.get("valence"),
                    features.get("danceability"),
                    features.get("acousticness"),
                    features.get("instrumentalness"),
                    features.get("loudness_lufs"),
                    features.get("spectral_centroid"),
                    features.get("file_mtime"),
                    features.get("file_size"),
                    features.get("openl3_embedding", b""),
                    features.get("efficientat_embedding", b""),
                    features.get("genre_top", ""),
                    json.dumps(features.get("genre_scores", {})),
                    json.dumps(features.get("mood_tags", [])),
                    json.dumps(features.get("mood_scores", {})),
                    features.get("analysis_version", "1.0"),
                    features.get("analyzed_at", datetime.utcnow().timestamp()),
                ),
            )
            await conn.commit()

    async def get_track_features(self, canonical_id: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM track_features WHERE canonical_id = ?", (canonical_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return dict(row)

    async def get_all_tracks(self) -> List[Dict[str, Any]]:
        """Get all tracks with identity and features joined."""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """SELECT t.*, f.bpm, f.key, f.energy, f.valence, f.danceability,
                          f.acousticness, f.instrumentalness, f.loudness_lufs,
                          f.spectral_centroid, f.genre_top, f.mood_tags, f.mood_scores
                   FROM track_identity t
                   LEFT JOIN track_features f ON t.canonical_id = f.canonical_id"""
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def add_playback(
        self,
        station_id: str,
        canonical_id: str,
        played_at: str,
        source: str,
        duration_sec: int = 0,
    ):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """INSERT INTO playback_history
                   (station_id, canonical_id, played_at, source, duration_sec)
                   VALUES (?, ?, ?, ?, ?)""",
                (station_id, canonical_id, played_at, source, duration_sec),
            )
            await conn.commit()

    async def upsert_station_profile(self, profile: Dict[str, Any]):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO station_profiles
                   (station_id, name, language, target_audience, config, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    profile.get("station_id", ""),
                    profile.get("name", ""),
                    profile.get("language", ""),
                    profile.get("target_audience", ""),
                    json.dumps(profile),
                    datetime.utcnow().isoformat(),
                ),
            )
            await conn.commit()

    async def get_station_profile(self, station_id: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM station_profiles WHERE station_id = ?", (station_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            if result.get("config"):
                try:
                    result["config"] = json.loads(result["config"])
                except Exception:
                    pass
            return result

    async def get_stats(self) -> Dict[str, int]:
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) as count FROM track_features")
            row = await cursor.fetchone()
            track_features = row[0] if row else 0

            cursor = await conn.execute("SELECT COUNT(*) as count FROM track_identity")
            row = await cursor.fetchone()
            track_identity = row[0] if row else 0

            return {
                "track_features": track_features,
                "track_identity": track_identity,
            }

    async def add_recommendation(
        self,
        station_id: str,
        canonical_id: str,
        score: float,
        source: str,
    ):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """INSERT INTO recommendation_history
                   (station_id, canonical_id, score, source, recommended_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (station_id, canonical_id, score, source, datetime.utcnow().isoformat()),
            )
            await conn.commit()
