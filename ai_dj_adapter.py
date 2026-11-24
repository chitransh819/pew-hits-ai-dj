"""
AI DJ Integration Adapter for PewHits
Drop this into pew-hits/config/ai_dj_adapter.py
Import and use in DJ.py
"""

import httpx
import asyncio
from pathlib import Path
from typing import Optional
from config.config import config
from config.cacheManager import cache_manager
from config.PlaylistHandler import pl
from config.songHandler import songHandler


class AIDJClient:
    """Client for communicating with AI DJ microservice"""
    
    def __init__(self):
        self.base_url = getattr(config, "AI_DJ_URL", "http://ai-dj:8000/api/v1")
        self.timeout = getattr(config, "AI_DJ_TIMEOUT", 2.0)
        self.enabled = getattr(config, "AI_DJ_ENABLED", True)
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self._station_name = self._get_station_name()
    
    def _get_station_name(self) -> str:
        """Determine station name from config"""
        radio_name = getattr(config, "RADIO_NAME", "").lower()
        if "desi" in radio_name or "hindi" in radio_name:
            return "pew-hits-desi"
        return "pew-hits-english"
    
    async def get_playable_tracks(self) -> list[dict]:
        """Build playable track list for AI DJ"""
        playable = []
        
        # 1. Cached files
        cache_dir = Path("Cache")
        if cache_dir.exists():
            for mp3_path in cache_dir.glob("*.mp3"):
                metadata = cache_manager.extract_metadata_from_file(str(mp3_path))
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
            print(f"[AI_DJ_ADAPTER] Could not load playlist tracks: {e}")
        
        return playable
    
    async def get_recommendation(
        self, 
        current_track: dict, 
        recent_history: list
    ) -> Optional[dict]:
        """Call AI DJ for next track recommendation"""
        if not self.enabled:
            return None
        
        playable = await self.get_playable_tracks()
        
        payload = {
            "station": self._station_name,
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
            print(f"[AI_DJ_ADAPTER] Timeout ({self.timeout}s), falling back")
        except httpx.HTTPStatusError as e:
            print(f"[AI_DJ_ADAPTER] HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"[AI_DJ_ADAPTER] Error: {e}, falling back")
        
        return None
    
    async def close(self):
        await self.client.aclose()


# Integration helper functions for DJ.py

def build_current_track_context() -> dict:
    """Build current track context from songHandler"""
    current = songHandler.get_now_playing_data()
    if not current:
        return {}
    
    return {
        "canonical_id": _to_canonical_id(current.spotifyID, current.title, current.artist),
        "title": current.title or "",
        "artist": current.artist or "",
        "album": current.album or "",
        "duration_sec": current.durationsec or 0,
    }


def build_recent_history(limit: int = 5) -> list[dict]:
    """Build recent history from songHandler"""
    history = songHandler.get_history()
    recent = []
    for item in history[-limit:]:
        recent.append({
            "canonical_id": _to_canonical_id(item.get("spotifyID"), item.get("title"), item.get("artist")),
            "title": item.get("title", ""),
            "artist": item.get("artist", ""),
            "album": item.get("album", ""),
            "duration_sec": item.get("durationsec", 0),
            "played_at": item.get("played", ""),
        })
    return recent


def _to_canonical_id(spotify_id: str, title: str, artist: str) -> str:
    """Convert Spotify ID or generate cache canonical ID"""
    if spotify_id and not spotify_id.startswith("cache_"):
        return f"track:spotify:{spotify_id}"
    safe_title = cache_manager.sanitize_filename(f"{title}_{artist}")
    return f"track:cache:{safe_title}"


async def try_stream_ai_dj_candidate(canonical_id: str, candidate: dict, downloader) -> bool:
    """Try to stream a candidate using existing pewhits logic"""
    # track:spotify:xxx -> use existing Spotify download flow
    if canonical_id.startswith("track:spotify:"):
        spotify_id = canonical_id.replace("track:spotify:", "")
        downloader.requestHandler.add_spotify_request(spotify_id, "AI DJ", "AutoDJ")
        await asyncio.sleep(2)
        if songHandler.next_coming_file_exists(spotify_id):
            update_next_coming_with_ai_metadata(candidate)
            return True
    
    # track:cache:xxx -> find in Cache/ and play directly
    elif canonical_id.startswith("track:cache:"):
        filename = canonical_id.replace("track:cache:", "") + ".mp3"
        cache_path = Path("Cache") / filename
        if cache_path.exists():
            from config.DJ import adder
            adder.add_song(str(cache_path))
            update_next_coming_with_ai_metadata(candidate)
            return True
    
    # track:youtube:xxx -> try direct stream or download
    elif canonical_id.startswith("track:youtube:"):
        youtube_id = canonical_id.replace("track:youtube:", "")
        youtube_url = f"https://youtube.com/watch?v={youtube_id}"
        stream_info = await downloader.unified_downloader.resolve_youtube_stream(
            youtube_url, candidate["title"]
        )
        if stream_info and stream_info.get("direct_url"):
            await downloader.queue_direct_stream_result(
                track_id=f"youtube_{youtube_id}",
                track_name=candidate["title"],
                artist_name=candidate["artist"],
                album_name=candidate.get("album", "YouTube"),
                requester="AI DJ",
                appreq="AutoDJ",
                duration=candidate.get("duration_sec", 180),
                youtube_url=youtube_url,
                stream_info=stream_info,
                queue_reason="AI DJ direct stream",
            )
            return True
        # Fallback to download
        downloader.requestHandler.add_youtube_request(youtube_url, "AI DJ", "AutoDJ")
        await asyncio.sleep(2)
        if songHandler.next_coming_file_exists(f"youtube_{youtube_id}"):
            return True
    
    return False


def update_next_coming_with_ai_metadata(candidate: dict):
    """Update next_coming.json with AI DJ selection metadata"""
    next_coming_data = songHandler.load_json(songHandler.next_coming_file)
    if next_coming_data:
        if isinstance(next_coming_data, list):
            next_coming_data = next_coming_data[0] if next_coming_data else {}
        
        if isinstance(next_coming_data, dict):
            next_coming_data["ai_selected"] = True
            next_coming_data["ai_score"] = candidate.get("score")
            next_coming_data["ai_reason"] = candidate.get("reason")
            next_coming_data["ai_confidence"] = candidate.get("confidence")
            songHandler.save_json(songHandler.next_coming_file, next_coming_data)


# Example integration in DJ.py:
"""
# In Downloader.__init__:
self.ai_dj_client = AIDJClient()

# In get_next_song_from_AIplaylist():
async def get_next_song_from_AIplaylist(self):
    # ... existing setup ...
    
    if self.ai_dj_client.enabled:
        current = build_current_track_context()
        recent = build_recent_history(5)
        
        ai_response = await self.ai_dj_client.get_recommendation(current, recent)
        
        if ai_response and ai_response.get("candidates"):
            for candidate in ai_response["candidates"]:
                track_id = candidate["canonical_id"]
                if await try_stream_ai_dj_candidate(track_id, candidate, self):
                    return self.format_for_streamer(track_id, candidate["score"], candidate["reason"])
    
    # Fallback to existing logic
    return await self.get_next_song_from_AIplaylist_legacy()
"""