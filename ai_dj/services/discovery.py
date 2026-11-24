"""
Music Discovery Service for Pew Hits AI DJ
Discovers new song metadata from Spotify, YouTube, JioSaavn, and free music sources.
Only returns metadata - NO audio download or streaming. Station handles that via existing fallback chain.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ai_dj.services.spotify_client import SpotifyClient, SpotifyTrack

logger = logging.getLogger(__name__)


class DiscoverySource(str, Enum):
    SPOTIFY = "spotify"
    YOUTUBE = "youtube"
    JIOSAAVN = "jiosaavn"
    FREE_MUSIC = "free_music"


@dataclass
class DiscoveredTrack:
    """New song metadata discovered from external sources."""
    source: DiscoverySource
    source_track_id: str
    title: str
    artist: str
    album: str = ""
    duration_sec: int = 0
    genre: str = ""
    language: str = "english"
    release_date: Optional[str] = None
    popularity: int = 50
    explicit: bool = False
    preview_url: Optional[str] = None
    external_url: str = ""
    isrc: Optional[str] = None
    confidence: float = 0.5  # How confident we are this matches station vibe
    matched_keywords: list[str] = field(default_factory=list)

    def to_canonical_id(self) -> str:
        """Generate canonical ID for this discovered track."""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{self.artist}_{self.title}").lower()
        return f"track:{self.source.value}:{safe}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "source_track_id": self.source_track_id,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "duration_sec": self.duration_sec,
            "genre": self.genre,
            "language": self.language,
            "release_date": self.release_date,
            "popularity": self.popularity,
            "explicit": self.explicit,
            "preview_url": self.preview_url,
            "external_url": self.external_url,
            "isrc": self.isrc,
            "confidence": self.confidence,
            "matched_keywords": self.matched_keywords,
            "canonical_id": self.to_canonical_id(),
        }


class DiscoveryService:
    """
    Discovers new song metadata from multiple sources.
    Returns metadata only - no audio download. Station handles download via existing fallback chain.
    """

    def __init__(
        self,
        spotify_client: Optional[SpotifyClient] = None,
        youtube_api_key: Optional[str] = None,
        jiosaavn_base_url: Optional[str] = None,
        free_music_api_key: Optional[str] = None,
    ):
        self.spotify = spotify_client
        self.youtube_api_key = youtube_api_key
        self.jiosaavn_base_url = jiosaavn_base_url
        self.free_music_api_key = free_music_api_key
        self._discovered_cache: dict[str, list[DiscoveredTrack]] = {}
        self._cache_ttl: dict[str, float] = {}
        self._ttl_seconds = 3600  # 1 hour cache

    async def discover_new_songs(
        self,
        station_id: str,
        language: str,
        genres: list[str],
        mood_keywords: list[str] = None,
        limit_per_source: int = 10,
    ) -> list[DiscoveredTrack]:
        """
        Discover new songs from all configured sources.
        Returns deduplicated metadata list.
        """
        mood_keywords = mood_keywords or []
        all_tracks: list[DiscoveredTrack] = []

        # Spotify discovery
        if self.spotify:
            try:
                spotify_tracks = await self._discover_spotify(
                    language, genres, mood_keywords, limit_per_source
                )
                all_tracks.extend(spotify_tracks)
                logger.info(f"[DISCOVERY] Spotify: {len(spotify_tracks)} tracks found")
            except Exception as e:
                logger.warning(f"[DISCOVERY] Spotify failed: {e}")

        # YouTube discovery
        if self.youtube_api_key:
            try:
                youtube_tracks = await self._discover_youtube(
                    language, genres, mood_keywords, limit_per_source
                )
                all_tracks.extend(youtube_tracks)
                logger.info(f"[DISCOVERY] YouTube: {len(youtube_tracks)} tracks found")
            except Exception as e:
                logger.warning(f"[DISCOVERY] YouTube failed: {e}")

        # JioSaavn discovery (for Desi station)
        if self.jiosaavn_base_url and language in ("hindi", "punjabi", "tamil", "telugu", "bengali", "gujarati", "marathi"):
            try:
                jiosaavn_tracks = await self._discover_jiosaavn(
                    language, genres, mood_keywords, limit_per_source
                )
                all_tracks.extend(jiosaavn_tracks)
                logger.info(f"[DISCOVERY] JioSaavn: {len(jiosaavn_tracks)} tracks found")
            except Exception as e:
                logger.warning(f"[DISCOVERY] JioSaavn failed: {e}")

        # Free music sources
        try:
            free_tracks = await self._discover_free_music(
                language, genres, mood_keywords, limit_per_source
            )
            all_tracks.extend(free_tracks)
            logger.info(f"[DISCOVERY] Free music: {len(free_tracks)} tracks found")
        except Exception as e:
            logger.warning(f"[DISCOVERY] Free music failed: {e}")

        # Deduplicate by title + artist
        seen: set[str] = set()
        unique_tracks: list[DiscoveredTrack] = []
        for track in all_tracks:
            key = f"{track.artist.lower()}|{track.title.lower()}"
            if key not in seen:
                seen.add(key)
                unique_tracks.append(track)

        # Sort by confidence * popularity
        unique_tracks.sort(key=lambda t: t.confidence * (t.popularity / 100.0), reverse=True)

        logger.info(f"[DISCOVERY] Total unique tracks for {station_id}: {len(unique_tracks)}")
        return unique_tracks[: limit_per_source * 4]  # Cap total

    async def _discover_spotify(
        self,
        language: str,
        genres: list[str],
        mood_keywords: list[str],
        limit: int,
    ) -> list[DiscoveredTrack]:
        """Discover tracks from Spotify metadata search."""
        tracks: list[DiscoveredTrack] = []

        # Build search queries from genres + mood
        queries = []
        for genre in genres[:3]:
            queries.append(genre)
        for mood in mood_keywords[:2]:
            queries.append(mood)
        if language == "english":
            queries.extend(["new pop", "new release", "trending"])
        elif language == "hindi":
            queries.extend(["bollywood new", "hindi pop", "new release"])

        for query in queries[:5]:  # Limit API calls
            try:
                spotify_tracks = await self.spotify.search_tracks(
                    query=query,
                    limit=limit,
                    market="IN" if language in ("hindi", "punjabi") else "US",
                )
                for st in spotify_tracks:
                    # Calculate confidence based on genre/language match
                    confidence = self._calc_spotify_confidence(st, language, genres, mood_keywords)
                    tracks.append(DiscoveredTrack(
                        source=DiscoverySource.SPOTIFY,
                        source_track_id=st.track_id,
                        title=st.title,
                        artist=st.artist,
                        album=st.album,
                        duration_sec=st.duration_ms // 1000,
                        genre=st.genres[0] if st.genres else "",
                        language=language,
                        release_date=st.release_date,
                        popularity=st.popularity,
                        explicit=st.explicit,
                        preview_url=st.preview_url,
                        external_url=st.external_url,
                        isrc=st.isrc,
                        confidence=confidence,
                        matched_keywords=[q for q in queries if q.lower() in (st.title + " " + st.artist).lower()],
                    ))
            except Exception as e:
                logger.warning(f"[DISCOVERY] Spotify query '{query}' failed: {e}")

        return tracks

    async def _discover_youtube(
        self,
        language: str,
        genres: list[str],
        mood_keywords: list[str],
        limit: int,
    ) -> list[DiscoveredTrack]:
        """Discover tracks from YouTube Data API search."""
        tracks: list[DiscoveredTrack] = []
        # YouTube API integration would go here - requires google-api-python-client or raw HTTP
        # For now, return empty - YouTube discovery can be added when API key is configured
        logger.info("[DISCOVERY] YouTube source available but not yet configured (needs google-api-python-client)")
        return tracks

    async def _discover_jiosaavn(
        self,
        language: str,
        genres: list[str],
        mood_keywords: list[str],
        limit: int,
    ) -> list[DiscoveredTrack]:
        """Discover tracks from JioSaavn unofficial API."""
        tracks: list[DiscoveredTrack] = []
        # JioSaavn integration would go here - unofficial API, no key needed
        # Can be added when base_url is configured
        logger.info("[DISCOVERY] JioSaavn source available but not yet configured (needs base_url)")
        return tracks

    async def _discover_free_music(
        self,
        language: str,
        genres: list[str],
        mood_keywords: list[str],
        limit: int,
    ) -> list[DiscoveredTrack]:
        """Discover tracks from free music sources (Free Music Archive, etc.)."""
        tracks: list[DiscoveredTrack] = []
        # Free music source integration - can be added when API key configured
        logger.info("[DISCOVERY] Free music source available but not yet configured")
        return tracks

    def _calc_spotify_confidence(
        self,
        track: SpotifyTrack,
        language: str,
        genres: list[str],
        mood_keywords: list[str],
    ) -> float:
        """Calculate confidence score for a Spotify track matching station vibe."""
        score = 0.5  # Base

        # Genre match
        track_genre_lower = (track.genres[0] if track.genres else "").lower()
        if any(g.lower() in track_genre_lower or track_genre_lower in g.lower() for g in genres):
            score += 0.2

        # Language match (Spotify doesn't return language directly, use heuristic)
        if language == "hindi" and any(k in (track.title + " " + track.artist).lower() for k in ["dil", "pyar", "tum", "mere"]):
            score += 0.2

        # Popularity boost
        if track.popularity > 70:
            score += 0.1

        # Recency boost (if release_date is recent)
        if track.release_date:
            try:
                from datetime import datetime
                release = datetime.fromisoformat(track.release_date.replace("Z", "+00:00"))
                days_old = (datetime.now(release.tzinfo) - release).days
                if days_old < 90:
                    score += 0.1
            except Exception:
                pass

        return min(1.0, score)

    def get_cached_discovery(self, station_id: str) -> Optional[list[DiscoveredTrack]]:
        """Get cached discovery results if not expired."""
        import time
        if station_id in self._cache_ttl:
            if time.time() < self._cache_ttl[station_id]:
                return self._discovered_cache.get(station_id, [])
        return None

    def cache_discovery(self, station_id: str, tracks: list[DiscoveredTrack]) -> None:
        """Cache discovery results with TTL."""
        import time
        self._discovered_cache[station_id] = tracks
        self._cache_ttl[station_id] = time.time() + self._ttl_seconds
