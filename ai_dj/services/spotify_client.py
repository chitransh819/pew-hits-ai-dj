# Spotify Web API client for music metadata discovery (metadata only - no audio download)
# Spotify Developer Terms: metadata search allowed, no stream ripping, no full-track download

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SpotifyTrack:
    """Spotify track metadata (no audio file)."""
    track_id: str
    title: str
    artist: str
    album: str = ""
    duration_ms: int = 0
    popularity: int = 50
    explicit: bool = False
    preview_url: str | None = None  # 30-second preview max per Spotify terms
    external_url: str = ""
    genres: list[str] = field(default_factory=list)
    release_date: str | None = None
    isrc: str | None = None


class SpotifyClient:
    """
    Spotify Web API client for metadata discovery only.

    Uses OAuth 2.0 Client Credentials flow for app-level access.
    Per Spotify Developer Terms:
    - Metadata search allowed ✅
    - 30-second preview clips allowed ✅
    - NO full-track download ❌
    - NO DJ/mixing of Spotify content ❌
    - NO saving Spotify content ❌
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str = "https://api.spotify.com/v1",
        token_url: str = "https://accounts.spotify.com/api/token",
        timeout_seconds: int = 10,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.token_url = token_url
        self.timeout_seconds = timeout_seconds
        self._access_token: str | None = None
        self._token_expires_at: float = 0
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client

    async def _get_access_token(self) -> str:
        """Get or refresh OAuth 2.0 client credentials token."""
        async with self._lock:
            now = time.time()
            if self._access_token and now < self._token_expires_at - 60:
                return self._access_token

            client = await self._get_client()
            response = await client.post(
                self.token_url,
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
            )
            response.raise_for_status()
            data = response.json()
            self._access_token = data["access_token"]
            self._token_expires_at = now + data.get("expires_in", 3600)
            return self._access_token

    async def _auth_headers(self) -> dict[str, str]:
        token = await self._get_access_token()
        return {"Authorization": f"Bearer {token}"}

    async def search_tracks(
        self,
        query: str,
        limit: int = 20,
        market: str = "IN",
    ) -> list[SpotifyTrack]:
        """Search Spotify for tracks by query (metadata only)."""
        client = await self._get_client()
        headers = await self._auth_headers()

        response = await client.get(
            f"{self.base_url}/search",
            headers=headers,
            params={
                "q": query,
                "type": "track",
                "limit": min(limit, 50),
                "market": market,
            },
        )
        response.raise_for_status()
        data = response.json()

        tracks = []
        for item in data.get("tracks", {}).get("items", []):
            tracks.append(SpotifyTrack(
                track_id=item["id"],
                title=item["name"],
                artist=", ".join(a["name"] for a in item["artists"]),
                album=item.get("album", {}).get("name", ""),
                duration_ms=item.get("duration_ms", 0),
                popularity=item.get("popularity", 50),
                explicit=item.get("explicit", False),
                preview_url=item.get("preview_url"),
                external_url=item.get("external_urls", {}).get("spotify", ""),
                isrc=item.get("external_ids", {}).get("isrc"),
                release_date=item.get("album", {}).get("release_date"),
            ))
        return tracks

    async def search_by_genre(
        self,
        genre: str,
        limit: int = 20,
        market: str = "IN",
    ) -> list[SpotifyTrack]:
        """Search Spotify tracks by genre (metadata only)."""
        return await self.search_tracks(
            query=f"genre:\"{genre}\"",
            limit=limit,
            market=market,
        )

    async def search_new_releases_by_artist(
        self,
        artist_ids: list[str],
        limit: int = 20,
        market: str = "IN",
    ) -> list[SpotifyTrack]:
        """Get tracks from specific artists (for discovering new songs by known artists)."""
        tracks = []
        for artist_id in artist_ids[:5]:  # Spotify API limits
            client = await self._get_client()
            headers = await self._auth_headers()
            response = await client.get(
                f"{self.base_url}/artists/{artist_id}/top-tracks",
                headers=headers,
                params={"market": market},
            )
            if response.status_code == 200:
                data = response.json()
                for item in data.get("tracks", [])[:limit]:
                    tracks.append(SpotifyTrack(
                        track_id=item["id"],
                        title=item["name"],
                        artist=", ".join(a["name"] for a in item["artists"]),
                        album=item.get("album", {}).get("name", ""),
                        duration_ms=item.get("duration_ms", 0),
                        popularity=item.get("popularity", 50),
                        explicit=item.get("explicit", False),
                        external_url=item.get("external_urls", {}).get("spotify", ""),
                        isrc=item.get("external_ids", {}).get("isrc"),
                        release_date=item.get("album", {}).get("release_date"),
                    ))
        return tracks

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
