"""
Recommendation Engine for PEW Hits AI DJ.

Implements multi-factor scoring, candidate generation, top-K selection with diversity,
transition validation, NVIDIA re-ranking integration, and explanation generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

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
    mood: float = 0.5           # 0.0 (sad) to 1.0 (happy)
    energy: float = 0.5         # 0.0 (calm) to 1.0 (high energy)
    tempo: float = 120.0        # BPM
    genre: str = "pop"
    language: str = "english"
    key: int = 0                # Camelot key (0-23)
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
    album: Optional[str] = None
    duration_ms: int = 0
    bpm: float = 120.0
    key: int = 0                # Camelot key (0-23)
    energy: float = 0.5         # 0.0 to 1.0
    valence: float = 0.5        # 0.0 (sad) to 1.0 (happy)
    danceability: float = 0.5
    acousticness: float = 0.5
    instrumentalness: float = 0.0
    liveness: float = 0.0
    speechiness: float = 0.0
    genre: str = "pop"
    language: str = "english"
    release_date: Optional[str] = None
    popularity: int = 50        # 0-100
    cache_status: str = "available"  # available, processing, missing
    embedding: Optional[list[float]] = None  # Semantic embedding vector


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
            "track_id": self.track.track_id,
            "title": self.track.title,
            "artist": self.track.artist,
            "scores": self.scores,
            "total_score": self.total_score,
            "explanation": self.explanation,
            "confidence": self.confidence,
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
        """Wait if we've hit the rate limit."""
        async with self._lock:
            now = time.time()
            # Remove requests older than 60 seconds
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
        """
        Re-rank candidates using NVIDIA Nemotron 3.5 Rerank 8B.
        Returns re-ranked candidates with updated scores and explanations.
        """
        if not candidates:
            return candidates

        await self._wait_for_rate_limit()

        # Prepare passages for reranking
        passages = []
        for c in candidates:
            passage = (
                f"Track: {c.track.title} by {c.track.artist}. "
                f"Genre: {c.track.genre}, Language: {c.track.language}, "
                f"BPM: {c.track.bpm}, Energy: {c.track.energy:.2f}, "
                f"Mood (valence): {c.track.valence:.2f}. "
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

                # Parse rankings and update candidates
                ranking_map = {}
                for item in data.get("rankings", []):
                    idx = item.get("index", 0)
                    score = item.get("logit", 0.0)
                    ranking_map[idx] = score

                # Update candidates with rerank scores
                reranked = []
                for i, c in enumerate(candidates):
                    if i in ranking_map:
                        c.scores["nvidia_rerank"] = self._normalize_rerank_score(ranking_map[i])
                        c.total_score = self._recalculate_total(c.scores)
                        c.confidence = min(1.0, c.confidence + 0.1)
                    reranked.append(c)

                # Sort by new total score
                reranked.sort(key=lambda x: x.total_score, reverse=True)
                return reranked[:top_k]

            except httpx.TimeoutException:
                logger.warning(f"NVIDIA rerank timeout (attempt {attempt + 1}/{max_retries})")
            except httpx.HTTPStatusError as e:
                logger.warning(f"NVIDIA rerank HTTP error: {e.response.status_code} (attempt {attempt + 1}/{max_retries})")
                if e.response.status_code == 429:
                    # Rate limited, wait longer
                    await asyncio.sleep(base_delay * (2 ** attempt) * 2)
                    continue
            except Exception as e:
                logger.warning(f"NVIDIA rerank error: {e} (attempt {attempt + 1}/{max_retries})")

            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.5))

        # Fallback: return original candidates with a note
        logger.info("NVIDIA rerank failed, using local scores")
        for c in candidates:
            c.scores["nvidia_rerank"] = 0.5
            c.explanation += " [NVIDIA rerank unavailable, using local scoring]"
        return candidates[:top_k]

    def _normalize_rerank_score(self, logit: float) -> float:
        """Normalize rerank logit to 0-1 range using sigmoid."""
        return 1.0 / (1.0 + math.exp(-logit))

    def _recalculate_total(self, scores: dict[str, float]) -> float:
        """Recalculate total score from components (placeholder, uses engine weights)."""
        return sum(scores.values()) / max(len(scores), 1)


class RecommendationEngine:
    """Main recommendation engine with multi-factor scoring and NVIDIA re-ranking."""

    def __init__(
        self,
        station_config: StationConfig,
        nvidia_client: Optional[NVIDIAClient] = None,
        playable_track_ids: Optional[set[str]] = None,
        track_metadata_provider: Optional[callable] = None,
        recent_history: Optional[list[str]] = None,
        artist_last_played: Optional[dict[str, datetime]] = None,
    ):
        self.station = station_config
        self.nvidia = nvidia_client
        self.playable_track_ids = playable_track_ids or set()
        self.track_metadata_provider = track_metadata_provider
        self.recent_history = recent_history or []
        self.artist_last_played = artist_last_played or {}

    def load_tracks(self, track_ids: list[str]) -> list[TrackMetadata]:
        """Load track metadata for given IDs."""
        tracks = []
        for tid in track_ids:
            if self.track_metadata_provider:
                track = self.track_metadata_provider(tid)
                if track and track.cache_status == "available":
                    tracks.append(track)
        return tracks

    def filter_candidates(self, tracks: list[TrackMetadata]) -> list[TrackMetadata]:
        """Apply station-specific filters to candidate tracks."""
        filtered = []
        filters = self.station.filters

        for track in tracks:
            # Language filter
            allowed_langs = filters.get("allowed_languages", [])
            if allowed_langs and track.language not in allowed_langs:
                continue

            # Genre filter
            allowed_genres = filters.get("allowed_genres", [])
            if allowed_genres and track.genre not in allowed_genres:
                continue

            # Energy range filter
            energy_range = filters.get("energy_range", [0.0, 1.0])
            if not (energy_range[0] <= track.energy <= energy_range[1]):
                continue

            # Tempo range filter
            tempo_range = filters.get("tempo_range", [0, 300])
            if not (tempo_range[0] <= track.bpm <= tempo_range[1]):
                continue

            # Playable track IDs filter
            if self.playable_track_ids and track.track_id not in self.playable_track_ids:
                continue

            # Cache status filter
            if track.cache_status != "available":
                continue

            # Recent history exclusion
            if track.track_id in self.recent_history[: self.station.recent_history_exclusion_count]:
                continue

            # Artist cooldown
            cooldown_minutes = self.station.artist_cooldown_minutes
            if track.artist in self.artist_last_played:
                last_played = self.artist_last_played[track.artist]
                if datetime.utcnow() - last_played < timedelta(minutes=cooldown_minutes):
                    continue

            filtered.append(track)

        return filtered

    def score_mood(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        """Score based on mood (valence) alignment."""
        # Valence 0-1, vibe.mood 0-1
        diff = abs(track.valence - vibe.mood)
        return 1.0 - diff

    def score_energy(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        """Score based on energy alignment."""
        diff = abs(track.energy - vibe.energy)
        return 1.0 - diff

    def score_tempo(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        """Score based on tempo (BPM) alignment."""
        if vibe.tempo <= 0:
            return 0.5
        # Normalize difference relative to vibe tempo
        diff_ratio = abs(track.bpm - vibe.tempo) / vibe.tempo
        return max(0.0, 1.0 - diff_ratio)

    def score_genre(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        """Score based on genre match."""
        if track.genre.lower() == vibe.genre.lower():
            return 1.0
        # Partial credit for related genres (simplified)
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
        """Score based on language match."""
        if track.language.lower() == vibe.language.lower():
            return 1.0
        # For Desi station, related languages get partial credit
        if vibe.language.lower() in ("hindi", "punjabi", "tamil", "telugu", "bengali", "gujarati", "marathi"):
            desi_langs = {"hindi", "punjabi", "tamil", "telugu", "bengali", "gujarati", "marathi"}
            if track.language.lower() in desi_langs:
                return 0.8
        return 0.2

    def score_artist(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        """Score based on artist familiarity/popularity."""
        # Normalize popularity (0-100) to 0-1
        pop_score = track.popularity / 100.0
        # Boost if artist was recently played (familiarity)
        if track.artist in self.artist_last_played:
            hours_ago = (datetime.utcnow() - self.artist_last_played[track.artist]).total_seconds() / 3600
            if hours_ago < 24:
                pop_score = min(1.0, pop_score + 0.1)
        return pop_score

    def score_transition(
        self,
        track: TrackMetadata,
        vibe: VibeProfile,
        prev_track: Optional[TrackMetadata] = None,
    ) -> float:
        """Score based on transition quality from previous track."""
        if prev_track is None:
            return 0.5  # Neutral for first track

        rules = self.station.transition_rules
        score = 1.0

        # BPM delta
        bpm_delta = abs(track.bpm - prev_track.bpm)
        max_bpm_delta = rules.get("max_bpm_delta", 30)
        if bpm_delta > max_bpm_delta:
            score *= max(0.0, 1.0 - (bpm_delta - max_bpm_delta) / max_bpm_delta)

        # Energy delta
        energy_delta = abs(track.energy - prev_track.energy)
        max_energy_delta = rules.get("max_energy_delta", 0.3)
        if energy_delta > max_energy_delta:
            score *= max(0.0, 1.0 - (energy_delta - max_energy_delta) / max_energy_delta)

        # Key distance (Camelot wheel)
        key_distance = min(
            abs(track.key - prev_track.key),
            24 - abs(track.key - prev_track.key),
        )
        max_key_distance = rules.get("max_key_distance", 5)
        if key_distance > max_key_distance:
            score *= max(0.0, 1.0 - (key_distance - max_key_distance) / max_key_distance)

        # Genre shift allowance
        if not rules.get("allow_genre_shift", True) and track.genre != prev_track.genre:
            score *= 0.5

        return score

    def score_semantic(self, track: TrackMetadata, vibe: VibeProfile) -> float:
        """Score based on semantic embedding similarity."""
        if track.embedding is None or not hasattr(vibe, "embedding") or vibe.embedding is None:
            return 0.5  # Neutral if no embeddings

        # Cosine similarity
        dot = sum(a * b for a, b in zip(track.embedding, vibe.embedding, strict=False))
        norm_a = math.sqrt(sum(a * a for a in track.embedding))
        norm_b = math.sqrt(sum(b * b for b in vibe.embedding))
        if norm_a == 0 or norm_b == 0:
            return 0.5
        similarity = dot / (norm_a * norm_b)
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

    def score_recency(self, track: TrackMetadata) -> float:
        """Score based on track recency (newer tracks get slight boost)."""
        if not track.release_date:
            return 0.5
        try:
            release = datetime.fromisoformat(track.release_date.replace("Z", "+00:00"))
            age_days = (datetime.utcnow() - release.replace(tzinfo=None)).days
            # Decay over ~2 years
            return max(0.3, 1.0 - age_days / 730.0)
        except Exception:
            return 0.5

    def compute_scores(
        self,
        track: TrackMetadata,
        vibe: VibeProfile,
        prev_track: Optional[TrackMetadata] = None,
    ) -> dict[str, float]:
        """Compute all scoring components for a track."""
        return {
            "mood": self.score_mood(track, vibe),
            "energy": self.score_energy(track, vibe),
            "tempo": self.score_tempo(track, vibe),
            "genre": self.score_genre(track, vibe),
            "language": self.score_language(track, vibe),
            "artist": self.score_artist(track, vibe),
            "transition": self.score_transition(track, vibe, prev_track),
            "semantic": self.score_semantic(track, vibe),
            "recency": self.score_recency(track),
        }

    def apply_weights(self, scores: dict[str, float]) -> float:
        """Apply station weights to compute weighted total score."""
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
        prev_track: Optional[TrackMetadata] = None,
    ) -> str:
        """Generate human-readable explanation for recommendation."""
        parts = []

        # Top contributing factors
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
                elif factor == "semantic":
                    parts.append("semantically similar to current vibe")
                elif factor == "recency":
                    parts.append("fresh release")

        if not parts:
            parts.append("good overall fit")

        explanation = f"Recommended because it {', '.join(parts)}."
        return explanation

    def select_top_k_diverse(
        self,
        candidates: list[Candidate],
        k: int,
        prev_track: Optional[TrackMetadata] = None,
    ) -> list[Candidate]:
        """Select top-K candidates ensuring artist/genre diversity."""
        if not candidates:
            return []

        diversity = self.station.diversity
        max_same_artist = diversity.get("max_same_artist_in_top_k", 2)
        max_same_genre = diversity.get("max_same_genre_in_top_k", 3)
        min_unique_artists = diversity.get("min_unique_artists", 5)
        max_same_language = diversity.get("max_same_language_in_top_k", k)

        # Sort by total score descending
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

        # If we don't have enough unique artists, relax constraints
        if len({c.track.artist for c in selected}) < min_unique_artists:
            # Second pass with relaxed constraints
            for candidate in candidates:
                if len(selected) >= k:
                    break
                if candidate in selected:
                    continue
                selected.append(candidate)

        return selected[:k]

    def validate_transition(
        self,
        track: TrackMetadata,
        prev_track: Optional[TrackMetadata] = None,
    ) -> tuple[bool, str]:
        """Validate transition against station rules."""
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

        if not rules.get("allow_genre_shift", True) and track.genre != prev_track.genre:
            issues.append(f"Genre shift not allowed: {prev_track.genre} -> {track.genre}")

        if issues:
            return False, "; ".join(issues)
        return True, "Valid transition"

    async def recommend(
        self,
        vibe: VibeProfile,
        prev_track: Optional[TrackMetadata] = None,
        k: int = 10,
        candidate_pool_size: int = 100,
    ) -> list[Candidate]:
        """
        Main recommendation pipeline:
        1. Load candidate pool from playable tracks
        2. Apply station filters
        3. Score all candidates
        4. Select top-K with diversity
        5. Validate transitions
        6. NVIDIA re-rank (if enabled)
        7. Generate explanations
        """
        # Step 1: Load candidate pool
        if self.track_metadata_provider:
            # Get all playable track IDs
            all_ids = list(self.playable_track_ids) if self.playable_track_ids else []
            if not all_ids:
                logger.warning("No playable track IDs available")
                return []

            # Sample pool if too large
            if len(all_ids) > candidate_pool_size:
                pool_ids = random.sample(all_ids, candidate_pool_size)
            else:
                pool_ids = all_ids

            tracks = self.load_tracks(pool_ids)
        else:
            tracks = []

        if not tracks:
            logger.warning("No tracks loaded")
            return []

        # Step 2: Apply filters
        filtered = self.filter_candidates(tracks)
        logger.info(f"Filtered {len(tracks)} tracks to {len(filtered)} candidates")

        if not filtered:
            return []

        # Step 3: Score all candidates
        candidates = []
        for track in filtered:
            scores = self.compute_scores(track, vibe, prev_track)
            total = self.apply_weights(scores)
            explanation = self.generate_explanation(track, scores, vibe, prev_track)
            confidence = min(1.0, total + 0.1)  # Base confidence from score

            candidates.append(Candidate(
                track=track,
                scores=scores,
                total_score=total,
                explanation=explanation,
                confidence=confidence,
            ))

        # Step 4: Select top-K with diversity
        # Get more than K for reranking
        rerank_k = self.station.nvidia_rerank.get("top_k_before_rerank", 50)
        top_candidates = self.select_top_k_diverse(candidates, min(rerank_k, len(candidates)), prev_track)

        # Step 5: Validate transitions
        validated = []
        for c in top_candidates:
            valid, reason = self.validate_transition(c.track, prev_track)
            if valid:
                validated.append(c)
            else:
                logger.debug(f"Transition invalid for {c.track.track_id}: {reason}")
                c.explanation += f" [Transition warning: {reason}]"
                c.confidence *= 0.8
                validated.append(c)  # Still include but with penalty

        # Step 6: NVIDIA re-rank
        if self.nvidia and self.station.nvidia_rerank.get("enabled", True):
            query = (
                f"Select the best next track for a {self.station.name} station. "
                f"Current vibe: mood={vibe.mood:.2f}, energy={vibe.energy:.2f}, "
                f"tempo={vibe.tempo:.0f} BPM, genre={vibe.genre}, language={vibe.language}. "
                f"Previous track: {prev_track.title if prev_track else 'None'}."
            )
            final_k = min(k, len(validated))
            reranked = await self.nvidia.rerank(query, validated, top_k=final_k)
            return reranked

        # Step 7: Final sort and return top-K
        validated.sort(key=lambda c: c.total_score, reverse=True)
        return validated[:k]


async def create_engine_from_config(
    station_config_path: Path,
    nvidia_api_key: Optional[str] = None,
    playable_track_ids: Optional[set[str]] = None,
    track_metadata_provider: Optional[callable] = None,
    recent_history: Optional[list[str]] = None,
    artist_last_played: Optional[dict[str, datetime]] = None,
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
        station_config=station,
        nvidia_client=nvidia_client,
        playable_track_ids=playable_track_ids,
        track_metadata_provider=track_metadata_provider,
        recent_history=recent_history,
        artist_last_played=artist_last_played,
    )