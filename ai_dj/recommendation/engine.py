"""
Recommendation Engine for PEW Hits AI DJ.

Auto-decides: recommend a local cached song OR a new song discovered from Spotify.
Single best track returned per request.
Station sends only: station, current_track, recent_history (last 5).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import yaml

from ai_dj.db.database import Database
from ai_dj.services.discovery import DiscoveryService
from ai_dj.services.spotify_client import SpotifyClient

logger = logging.getLogger(__name__)


class Language(str, Enum):
    ENGLISH = "english"
    HINDI = "hindi"
    PUNJABI = "punjabi"
    TAMIL = "tamil"
    TELUGU = "telugu"
    BENGALI = "bengali"
    GUJARATI = "gujarati"
    MARATHI = "marathi"


class Genre(str, Enum):
    POP = "pop"
    ROCK = "rock"
    HIP_HOP = "hip-hop"
    RNB = "r&b"
    ELECTRONIC = "electronic"
    INDIE = "indie"
    ALTERNATIVE = "alternative"
    BOLLYWOOD = "bollywood"
    PUNJABI_POP = "punjabi_pop"
    TAMIL_CINEMA = "tamil_cinema"
    INDIE_HINDI = "indie_hindi"
    SUFI = "sufi"
    CLASSICAL_FUSION = "classical_fusion"
    DESI_HIP_HOP = "desi_hip_hop"


@dataclass
class VibeProfile:
    """Current vibe state of the station."""
    mood: float = 0.5
    energy: float = 0.5
    tempo: float = 120.0
    genre: str = "pop"
    language: str = "english"
    key: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood,
            "energy": self.energy,
            "tempo": self.tempo,
            "genre": self.genre,
            "language": self.language,
            "key": self.key,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TrackMetadata:
    """Metadata for a candidate track."""
    track_id: str
    title: str
    artist: str
    album: str | None = None
    duration_ms: int = 0
    bpm: float = 120.0
    key: int = 0
    energy: float = 0.5
    valence: float = 0.5
    danceability: float = 0.5
    acousticness: float = 0.5
    instrumentalness: float = 0.0
    liveness: float = 0.0
    speechiness: float = 0.0
    genre: str = "pop"
    language: str = "english"
    release_date: str | None = None
    popularity: int = 50
    cache_status: str = "available"
    embedding: list[float] | None = None
    source: str = "local"
    source_track_id: str | None = None
    preview_url: str | None = None
    external_url: str | None = None


@dataclass
class Candidate:
    """A candidate track with computed scores."""
    track: TrackMetadata
    scores: dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    explanation: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.track.track_id,
            "title": self.track.title,
            "artist": self.track.artist,
            "album": self.track.album or "",
            "duration_sec": self.track.duration_ms // 1000,
            "genre": self.track.genre,
            "language": self.track.language,
            "source": self.track.source,
            "source_track_id": self.track.source_track_id,
            "score": self.total_score,
            "reason": self.explanation,
            "confidence": self.confidence,
            "preview_url": self.track.preview_url,
            "external_url": self.track.external_url,
            "release_date": self.track.release_date,
        }


@dataclass
class StationConfig:
    """Station configuration loaded from YAML."""
    station_id: str
    name: str
    language: str
    target_audience: str
    weights: dict[str, float]
    transition_rules: dict[str, Any]
    filters: dict[str, Any]
    artist_cooldown_minutes: int
    recent_history_exclusion_count: int
    diversity: dict[str, Any]
    nvidia_rerank: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> StationConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class NVIDIAClient:
    """Client for NVIDIA Nemotron 3.5 Rerank 8B API with rate limiting."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/nemotron-3.5-rerank-8b",
        rate_limit_rpm: int = 40,
        timeout_seconds: int = 10,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.rate_limit_rpm = rate_limit_rpm
        self.timeout_seconds = timeout_seconds
        self._request_times: list[float] = []
        self._lock = asyncio.Lock()

    async def _wait_for_rate_limit(self) -> None:
        async with self._lock:
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
            if len(self._request_times) >= self.rate_limit_rpm:
                oldest = self._request_times[0]
                wait_time = 60 - (now - oldest) + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limit hit, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            self._request_times.append(time.time())

    async def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int = 10,
    ) -> list[Candidate]:
        """Re-rank candidates using NVIDIA Nemotron 3.5 Rerank 8B."""
        if not candidates:
            return candidates

        await self._wait_for_rate_limit()

        passages = []
        for c in candidates:
            passage = (
                f"Track: {c.track.title} by {c.track.artist}. "
                f"Genre: {c.track.genre}, Language: {c.track.language}, "
                f"BPM: {c.track.bpm}, Energy: {c.track.energy:.2f}, "
                f"Mood: {c.track.valence:.2f}, Source: {c.track.source}. "
                f"Current scores: {json.dumps(c.scores)}"
            )
            passages.append({"text": passage, "id": c.track.track_id})

        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": passages,
            "top_k": min(top_k, len(candidates)),
            "truncate": "END",
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        f"{self.base_url}/ranking",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                ranking_map = {}
                for item in data.get("rankings", []):
                    idx = item.get("index", 0)
                    score = item.get("logit", 0.0)
                    ranking_map[idx] = score

                reranked = []
                for i, c in enumerate(candidates):
                    if i in ranking_map:
                        c.scores["nvidia_rerank"] = self._normalize_rerank_score(ranking_map[i])
                        c.total_score = self._recalculate_total(c.scores)
                        c.confidence = min(1.0, c.confidence + 0.1)
                    reranked.append(c)

                reranked.sort(key=lambda x: x.total_score, reverse=True)
                return reranked[:top_k]

            except Exception as e:
                logger.warning(f"NVIDIA rerank error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))

        logger.info("NVIDIA rerank failed, using local scores")
        for c in candidates:
            c.scores["nvidia_rerank"] = 0.5
            c.explanation += " [NVIDIA rerank unavailable, using local scoring]"
        return candidates[:top_k]

    def _normalize_rerank_score(self, logit: float) -> float:
        return 1.0 / (1.0 + math.exp(-logit))

    def _recalculate_total(self, scores: dict[str, float]) -> float:
        return sum(scores.values()) / max(len(scores), 1)


class RecommendationEngine:
    """
    Main recommendation engine.
    Auto-decides: local cached song OR new song from Spotify.
    Returns single best track per request.
    """

    def __init__(
        self,
        db: Database,
        station_config: StationConfig,
        discovery_service: DiscoveryService | None = None,
        spotify_client: SpotifyClient | None = None,
        nvidia_client: NVIDIAClient | None = None,
    ):
        self.db = db
        self.station = station_config
        self.discovery = discovery_service
        self.spotify = spotify_client
        self.nvidia = nvidia_client
        self.recent_history: list[str] = []
        self.artist_last_played: dict[str, datetime] = {}

    def filter_candidates(self, tracks: list[TrackMetadata]) -> list[TrackMetadata]:
        """Apply station-specific filters to candidate tracks."""
        filtered = []
        filters = self.station.filters

        for track in tracks:
            allowed_langs = filters.get("allowed_languages", [])
            if allowed_langs and track.language not in allowed_langs:
                continue

            allowed_genres = filters.get("allowed_genres", [])
            if allowed_genres and track.genre not in allowed_genres:
                continue

            energy_range = filters.get("energy_range", [0.0, 1.0])
            if not (energy_range[0] <= track.energy <= energy_range[1]):
                continue

            tempo_range = filters.get("tempo_range", [0, 300])
            if not (tempo_range[0] <= track.bpm <= tempo_range[1]):
                continue

            if track.cache_status != "available":
                continue

            if track.track_id in self.recent_history[: self.station.recent_history_exclusion_count]:
                continue

            cooldown_minutes = self.station.artist_cooldown_minutes
            if track.artist in self.artist_last_played:
                last_played = self.artist_last_played[track.artist]
                if datetime.utcnow() - last_played < timedelta(minutes=cooldown_minutes):
                    continue

            filtered.append(track)

        return filtered

    def score_mood(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        diff = abs(track.valence - vibe.mood)
        return 1.0 - diff

    def score_energy(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        diff = abs(track.energy - vibe.energy)
        return 1.0 - diff

    def score_tempo(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        if vibe.tempo <= 0:
            return 0.5
        diff_ratio = abs(track.bpm - vibe.tempo) / vibe.tempo
        return max(0.0, 1.0 - diff_ratio)

    def score_genre(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        if track.genre.lower() == vibe.genre.lower():
            return 1.0
        related = {
            "pop": ["electronic", "indie", "r&b"],
            "rock": ["alternative", "indie"],
            "hip-hop": ["r&b", "desi_hip_hop"],
            "bollywood": ["punjabi_pop", "indie_hindi", "sufi"],
            "punjabi_pop": ["bollywood", "desi_hip_hop"],
            "tamil_cinema": ["bollywood", "classical_fusion"],
        }
        vibe_related = related.get(vibe.genre.lower(), [])
        if track.genre.lower() in vibe_related:
            return 0.7
        return 0.3

    def score_language(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        if track.language.lower() == vibe.language.lower():
            return 1.0
        if vibe.language.lower() in ("hindi", "punjabi", "tamil", "telugu", "bengali", "gujarati", "marathi"):
            desi_langs = {"hindi", "punjabi", "tamil", "telugu", "bengali", "gujarati", "marathi"}
            if track.language.lower() in desi_langs:
                return 0.8
        return 0.2

    def score_artist(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        pop_score = track.popularity / 100.0
        if track.artist in self.artist_last_played:
            hours_ago = (datetime.utcnow() - self.artist_last_played[track.artist]).total_seconds() / 3600
            if hours_ago < 24:
                pop_score = min(1.0, pop_score + 0.1)
        return pop_score

    def score_transition(
        self,
        track: TrackMetadata,
        vibe: VibeProfile,
        prev_track: TrackMetadata | None = None,
    ) -> float:
        if prev_track is None:
            return 0.5
        rules = self.station.transition_rules
        score = 1.0

        bpm_delta = abs(track.bpm - prev_track.bpm)
        max_bpm_delta = rules.get("max_bpm_delta", 30)
        if bpm_delta > max_bpm_delta:
            score *= max(0.0, 1.0 - (bpm_delta - max_bpm_delta) / max_bpm_delta)

        energy_delta = abs(track.energy - prev_track.energy)
        max_energy_delta = rules.get("max_energy_delta", 0.3)
        if energy_delta > max_energy_delta:
            score *= max(0.0, 1.0 - (energy_delta - max_energy_delta) / max_energy_delta)

        key_distance = min(
            abs(track.key - prev_track.key),
            24 - abs(track.key - prev_track.key),
        )
        max_key_distance = rules.get("max_key_distance", 5)
        if key_distance > max_key_distance:
            score *= max(0.0, 1.0 - (key_distance - max_key_distance) / max_key_distance)

        return score

    def score_recency(self, track: TrackMetadata) -> float:
        if not track.release_date:
            return 0.5
        try:
            release = datetime.fromisoformat(track.release_date.replace("Z", "+00:00"))
            age_days = (datetime.utcnow() - release.replace(tzinfo=None)).days
            return max(0.3, 1.0 - age_days / 730.0)
        except Exception:
            return 0.5

    def compute_scores(
        self,
        track: TrackMetadata,
        vibe: VibeProfile,
        prev_track: TrackMetadata | None = None,
    ) -> dict[str, float]:
        return {
            "mood": self.score_mood(track, vibe),
            "energy": self.score_energy(track, vibe),
            "tempo": self.score_tempo(track, vibe),
            "genre": self.score_genre(track, vibe),
            "language": self.score_language(track, vibe),
            "artist": self.score_artist(track, vibe),
            "transition": self.score_transition(track, vibe, prev_track),
            "recency": self.score_recency(track),
        }

    def apply_weights(self, scores: dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        for component, weight in self.station.weights.items():
            if component in scores:
                total += scores[component] * weight
                weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0

    def generate_explanation(
        self,
        track: TrackMetadata,
        scores: dict[str, float],
        vibe: VibeProfile,
        prev_track: TrackMetadata | None = None,
    ) -> str:
        parts = []
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_factors = sorted_scores[:3]

        for factor, score in top_factors:
            if score > 0.7:
                if factor == "mood":
                    mood_desc = "upbeat" if vibe.mood > 0.6 else "chill" if vibe.mood < 0.4 else "balanced"
                    parts.append(f"matches the {mood_desc} mood")
                elif factor == "energy":
                    energy_desc = "high-energy" if vibe.energy > 0.7 else "low-key" if vibe.energy < 0.3 else "moderate"
                    parts.append(f"fits the {energy_desc} vibe")
                elif factor == "tempo":
                    parts.append(f"tempo ({track.bpm:.0f} BPM) aligns well")
                elif factor == "genre":
                    parts.append(f"{track.genre} genre matches station style")
                elif factor == "language":
                    parts.append(f"in {track.language}")
                elif factor == "artist":
                    parts.append(f"popular artist {track.artist}")
                elif factor == "transition":
                    if prev_track:
                        parts.append(f"smooth transition from {prev_track.title}")
                elif factor == "recency":
                    parts.append("fresh release")

        if not parts:
            parts.append("good overall fit")

        source_desc = "new song from Spotify" if track.source == "spotify" else "from local cache"
        explanation = f"Recommended {source_desc} because it {', '.join(parts)}."
        return explanation

    def validate_transition(
        self,
        track: TrackMetadata,
        prev_track: TrackMetadata | None = None,
    ) -> tuple[bool, str]:
        if prev_track is None:
            return True, "First track, no transition needed"
        rules = self.station.transition_rules
        issues = []

        bpm_delta = abs(track.bpm - prev_track.bpm)
        max_bpm_delta = rules.get("max_bpm_delta", 30)
        if bpm_delta > max_bpm_delta:
            issues.append(f"BPM delta {bpm_delta:.0f} exceeds max {max_bpm_delta}")

        energy_delta = abs(track.energy - prev_track.energy)
        max_energy_delta = rules.get("max_energy_delta", 0.3)
        if energy_delta > max_energy_delta:
            issues.append(f"Energy delta {energy_delta:.2f} exceeds max {max_energy_delta}")

        key_distance = min(
            abs(track.key - prev_track.key),
            24 - abs(track.key - prev_track.key),
        )
        max_key_distance = rules.get("max_key_distance", 5)
        if key_distance > max_key_distance:
            issues.append(f"Key distance {key_distance} exceeds max {max_key_distance}")

        if issues:
            return False, "; ".join(issues)
        return True, "Valid transition"

    async def _get_local_tracks(self) -> list[TrackMetadata]:
        """Get all locally analyzed tracks from DB."""
        tracks = []
        try:
            rows = await self.db.get_all_tracks()
            for row in rows:
                if not row.get("canonical_id"):
                    continue
                tracks.append(TrackMetadata(
                    track_id=row["canonical_id"],
                    title=row.get("title", ""),
                    artist=row.get("artist", ""),
                    album=row.get("album", "") or "",
                    duration_ms=row.get("duration_sec", 0) * 1000,
                    bpm=row.get("bpm", 120.0) or 120.0,
                    key=0,
                    energy=row.get("energy", 0.5) or 0.5,
                    valence=row.get("valence", 0.5) or 0.5,
                    danceability=row.get("danceability", 0.5) or 0.5,
                    acousticness=row.get("acousticness", 0.5) or 0.5,
                    genre=row.get("genre_top", "pop") or "pop",
                    language=row.get("language", "english") or "english",
                    release_date=row.get("release_date"),
                    popularity=50,
                    cache_status="available",
                    source="local",
                ))
        except Exception as e:
            logger.warning(f"Failed to get local tracks: {e}")
        return tracks

    async def _get_spotify_discovered(
        self,
        language: str,
        genres: list[str],
        mood_keywords: list[str],
        limit: int = 20,
    ) -> list[TrackMetadata]:
        """Get new song candidates from Spotify discovery."""
        tracks = []
        if not self.discovery or not self.spotify:
            return tracks

        try:
            discovered = await self.discovery.discover_new_songs(
                station_id=self.station.station_id,
                language=language,
                genres=genres,
                mood_keywords=mood_keywords,
                limit_per_source=limit,
            )
            for dt in discovered:
                tracks.append(TrackMetadata(
                    track_id=dt.to_canonical_id(),
                    title=dt.title,
                    artist=dt.artist,
                    album=dt.album,
                    duration_ms=dt.duration_sec * 1000,
                    bpm=120.0,
                    key=0,
                    energy=0.5,
                    valence=0.5,
                    genre=dt.genre,
                    language=dt.language,
                    release_date=dt.release_date,
                    popularity=dt.popularity,
                    cache_status="available",
                    source="spotify",
                    source_track_id=dt.source_track_id,
                    preview_url=dt.preview_url,
                    external_url=dt.external_url,
                ))
        except Exception as e:
            logger.warning(f"Spotify discovery failed: {e}")
        return tracks

    def _build_vibe(self, current_track: dict, recent_history: list[dict]) -> VibeProfile:
        """Build vibe profile from current track and history."""
        vibe = VibeProfile(
            mood=0.5,
            energy=0.5,
            tempo=120.0,
            genre="pop",
            language="english",
        )

        if current_track:
            vibe.language = current_track.get("language", "english") or "english"
            vibe.genre = current_track.get("genre", "pop") or "pop"
            if current_track.get("duration_sec"):
                vibe.tempo = 120.0

        title_artist = (current_track.get("title", "") + " " + current_track.get("artist", "")).lower()
        if any(w in title_artist for w in ["dance", "party", "upbeat", "high"]):
            vibe.energy = 0.8
            vibe.mood = 0.8
        elif any(w in title_artist for w in ["sad", "slow", "chill", "relax"]):
            vibe.energy = 0.3
            vibe.mood = 0.3

        return vibe

    def _build_prev_track(self, current_track: dict) -> TrackMetadata | None:
        """Build previous track metadata from current track context."""
        if not current_track:
            return None
        return TrackMetadata(
            track_id=current_track.get("canonical_id", ""),
            title=current_track.get("title", ""),
            artist=current_track.get("artist", ""),
            bpm=120.0,
            energy=0.5,
            valence=0.5,
            genre=current_track.get("genre", "pop"),
            language=current_track.get("language", "english"),
            source=current_track.get("source", "local"),
        )

    def _select_top_k_diverse(
        self,
        candidates: list[Candidate],
        k: int,
        prev_track: TrackMetadata | None = None,
    ) -> list[Candidate]:
        """Select top-K candidates ensuring artist/genre diversity."""
        if not candidates:
            return []
        diversity = self.station.diversity
        max_same_artist = diversity.get("max_same_artist_in_top_k", 2)
        max_same_genre = diversity.get("max_same_genre_in_top_k", 3)
        max_same_language = diversity.get("max_same_language_in_top_k", k)

        candidates.sort(key=lambda c: c.total_score, reverse=True)
        selected = []
        artist_counts: dict[str, int] = {}
        genre_counts: dict[str, int] = {}
        language_counts: dict[str, int] = {}

        for candidate in candidates:
            if len(selected) >= k:
                break
            artist = candidate.track.artist
            genre = candidate.track.genre
            language = candidate.track.language
            if artist_counts.get(artist, 0) >= max_same_artist:
                continue
            if genre_counts.get(genre, 0) >= max_same_genre:
                continue
            if language_counts.get(language, 0) >= max_same_language:
                continue
            selected.append(candidate)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            language_counts[language] = language_counts.get(language, 0) + 1

        return selected[:k]

    async def get_next_track(
        self,
        station_id: str,
        current_track: dict,
        recent_history: list[dict],
    ) -> dict[str, Any]:
        """
        Main entry point: auto-decide local vs Spotify, return single best track.
        """
        vibe = self._build_vibe(current_track, recent_history)
        prev_track = self._build_prev_track(current_track)

        # Step 1: Get local cached tracks
        local_tracks = await self._get_local_tracks()
        logger.info(f"[ENGINE] {len(local_tracks)} local tracks available")

        # Step 2: Get Spotify-discovered new songs
        spotify_tracks = await self._get_spotify_discovered(
            language=vibe.language,
            genres=[vibe.genre],
            mood_keywords=[],
            limit=20,
        )
        logger.info(f"[ENGINE] {len(spotify_tracks)} Spotify tracks discovered")

        # Step 3: Combine and filter
        all_tracks = local_tracks + spotify_tracks
        filtered = self.filter_candidates(all_tracks)
        logger.info(f"[ENGINE] {len(filtered)} candidates after filtering")

        if not filtered:
            return {
                "canonical_id": current_track.get("canonical_id", "track:cache:fallback"),
                "title": current_track.get("title", "Unknown"),
                "artist": current_track.get("artist", "Unknown"),
                "album": current_track.get("album", ""),
                "duration_sec": current_track.get("duration_sec", 180),
                "genre": current_track.get("genre", "pop"),
                "language": current_track.get("language", "english"),
                "source": "local",
                "source_track_id": None,
                "score": 0.5,
                "reason": "No candidates available, repeating current track",
                "confidence": 0.3,
                "preview_url": None,
                "external_url": None,
                "release_date": None,
            }

        # Step 4: Score all candidates
        candidates = []
        for track in filtered:
            scores = self.compute_scores(track, vibe, prev_track)
            total = self.apply_weights(scores)
            explanation = self.generate_explanation(track, scores, vibe, prev_track)
            candidates.append(Candidate(
                track=track,
                scores=scores,
                total_score=total,
                explanation=explanation,
                confidence=min(1.0, total + 0.1),
            ))

        # Step 5: Select top-K with diversity
        rerank_k = self.station.nvidia_rerank.get("top_k_before_rerank", 50)
        top_candidates = self._select_top_k_diverse(candidates, min(rerank_k, len(candidates)), prev_track)

        # Step 6: Validate transitions
        validated = []
        for c in top_candidates:
            valid, reason = self.validate_transition(c.track, prev_track)
            if not valid:
                c.explanation += f" [Transition warning: {reason}]"
                c.confidence *= 0.8
            validated.append(c)

        # Step 7: NVIDIA re-rank
        if self.nvidia and self.station.nvidia_rerank.get("enabled", True):
            query = (
                f"Select the best next track for a {self.station.name} station. "
                f"Current vibe: mood={vibe.mood:.2f}, energy={vibe.energy:.2f}, "
                f"tempo={vibe.tempo:.0f} BPM, genre={vibe.genre}, language={vibe.language}. "
                f"Previous track: {prev_track.title if prev_track else 'None'}."
            )
            final_k = 1
            reranked = await self.nvidia.rerank(query, validated, top_k=final_k)
            best = reranked[0] if reranked else validated[0]
        else:
            validated.sort(key=lambda c: c.total_score, reverse=True)
            best = validated[0]

        # Step 8: Record recommendation
        try:
            await self.db.add_recommendation(
                station_id=station_id,
                canonical_id=best.track.track_id,
                score=best.total_score,
                source=best.track.source,
            )
        except Exception as e:
            logger.warning(f"Failed to record recommendation: {e}")

        # Update history
        self.recent_history.append(best.track.track_id)
        if best.track.artist:
            self.artist_last_played[best.track.artist] = datetime.utcnow()

        return best.to_dict()


async def create_engine_from_config(
    station_config_path: Path,
    db: Database,
    discovery_service: DiscoveryService | None = None,
    spotify_client: SpotifyClient | None = None,
    nvidia_api_key: str | None = None,
) -> RecommendationEngine:
    """Factory function to create engine from config file."""
    station = StationConfig.from_yaml(station_config_path)

    nvidia_client = None
    if nvidia_api_key and station.nvidia_rerank.get("enabled", True):
        nvidia_client = NVIDIAClient(
            api_key=nvidia_api_key,
            model=station.nvidia_rerank.get("model", "nvidia/nemotron-3.5-rerank-8b"),
            rate_limit_rpm=station.nvidia_rerank.get("rate_limit_rpm", 40),
            timeout_seconds=station.nvidia_rerank.get("timeout_seconds", 10),
        )

    return RecommendationEngine(
        db=db,
        station_config=station,
        discovery_service=discovery_service,
        spotify_client=spotify_client,
        nvidia_client=nvidia_client,
    )
