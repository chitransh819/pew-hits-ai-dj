"""
Cache Synchronization Service
Watches cache directories for file changes and performs periodic reconciliation
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import List, Optional, Set

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from ai_dj.db.database import Database

logger = logging.getLogger(__name__)


class CacheEventHandler(FileSystemEventHandler):
    """Handles file system events for cache directories"""
    
    def __init__(self, db: Database, analyze_callback: Callable[[Path, str], Awaitable[None]]):
        self.db = db
        self.analyze_callback = analyze_callback
        self._pending_deletes = set()
        self._pending_creates = set()
    
    def on_created(self, event: FileCreatedEvent):
        if event.is_directory or not event.src_path.endswith('.mp3'):
            return
        self._handle_created(event.src_path)
    
    def on_deleted(self, event: FileDeletedEvent):
        if event.is_directory or not event.src_path.endswith('.mp3'):
            return
        self._handle_deleted(event.src_path)
    
    def on_modified(self, event: FileModifiedEvent):
        if event.is_directory or not event.src_path.endswith('.mp3'):
            return
        self._handle_modified(event.src_path)
    
    def on_moved(self, event: FileMovedEvent):
        if event.is_directory:
            return
        if event.src_path.endswith('.mp3'):
            self._handle_deleted(event.src_path)
        if event.dest_path.endswith('.mp3'):
            self._handle_created(event.dest_path)
    
    def _handle_created(self, file_path: str):
        path = Path(file_path)
        canonical_id = f"track:cache:{self._sanitize_filename(path.stem)}"
        logger.info(f"[CACHE_SYNC] File created: {canonical_id}")
        # Schedule analysis
        asyncio.run_coroutine_threadsafe(self.analyze_callback(path, canonical_id), self._loop)

    def _handle_deleted(self, file_path: str):
        path = Path(file_path)
        canonical_id = f"track:cache:{self._sanitize_filename(path.stem)}"
        logger.warning(f"[CACHE_SYNC] File deleted: {canonical_id}")
        # Mark as evicted in database
        asyncio.run_coroutine_threadsafe(self.db.update_cache_status(canonical_id, "evicted"), self._loop)

    def _handle_modified(self, file_path: str):
        path = Path(file_path)
        canonical_id = f"track:cache:{self._sanitize_filename(path.stem)}"
        logger.info(f"[CACHE_SYNC] File modified: {canonical_id}")
        # Re-analyze
        asyncio.run_coroutine_threadsafe(self.analyze_callback(path, canonical_id, force=True), self._loop)
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename for canonical ID (matches cacheManager.sanitize_filename)"""
        import re
        return re.sub(r'[^\w\-_]', '_', filename)


class CacheSyncService:
    """Main cache synchronization service"""

    def __init__(
        self,
        db: Database,
        cache_dirs: List[Path],
        analyzer,
        worker,
        reconcile_interval: int = 300,  # 5 minutes
    ):
        self.db = db
        self.cache_dirs = cache_dirs
        self.analyzer = analyzer
        self.worker = worker
        self.reconcile_interval = reconcile_interval

        self.observer = Observer()
        self._running = False
        self._reconcile_task = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def start(self):
        """Start the cache sync service"""
        self._running = True
        self._loop = asyncio.get_running_loop()

        # 1. Initial reconciliation
        logger.info("[CACHE_SYNC] Starting initial reconciliation...")
        await self.reconcile_all()
        
        # 2. Start file watchers
        for cache_dir in self.cache_dirs:
            cache_dir.mkdir(parents=True, exist_ok=True)
            handler = CacheEventHandler(
                self.db, 
                self._analyze_file_callback
            )
            self.observer.schedule(handler, str(cache_dir), recursive=False)
            logger.info(f"[CACHE_SYNC] Watching: {cache_dir}")
        
        self.observer.start()
        
        # 3. Start periodic reconciliation
        self._reconcile_task = asyncio.create_task(self._periodic_reconcile())
        
        logger.info("[CACHE_SYNC] Service started")
    
    async def stop(self):
        """Stop the cache sync service"""
        self._running = False
        
        if self._reconcile_task:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                pass
        
        self.observer.stop()
        self.observer.join()
        
        logger.info("[CACHE_SYNC] Service stopped")
    
    async def _analyze_file_callback(self, file_path: Path, canonical_id: str, force: bool = False):
        """Callback for analyzing a file"""
        try:
            await self.worker.queue_file(file_path, canonical_id, force)
        except Exception as e:
            logger.error(f"[CACHE_SYNC] Failed to queue analysis for {file_path}: {e}")
    
    async def _periodic_reconcile(self):
        """Periodic full reconciliation"""
        while self._running:
            try:
                await asyncio.sleep(self.reconcile_interval)
                if self._running:
                    await self.reconcile_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CACHE_SYNC] Periodic reconciliation error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def reconcile_all(self):
        """Reconcile all cache directories"""
        for cache_dir in self.cache_dirs:
            await self.reconcile_dir(cache_dir)
        
        # Also check for tracks in DB that are not in any cache dir
        await self.reconcile_orphaned()
    
    async def reconcile_dir(self, cache_dir: Path):
        """Reconcile a single cache directory"""
        if not cache_dir.exists():
            logger.warning(f"[CACHE_SYNC] Cache dir does not exist: {cache_dir}")
            return
        
        # Get actual files in cache
        actual_files: set[str] = set()
        for mp3 in cache_dir.glob("*.mp3"):
            canonical_id = f"track:cache:{self._sanitize_filename(mp3.stem)}"
            actual_files.add(canonical_id)
        
        # Get DB entries for cache tracks
        db_entries = await self._get_db_cache_ids()
        
        # Files in cache but not in DB -> analyze
        new_files = actual_files - db_entries
        for canonical_id in new_files:
            mp3_path = self._find_file_for_canonical(cache_dir, canonical_id)
            if mp3_path:
                logger.info(f"[CACHE_SYNC] New file detected: {canonical_id}")
                await self._analyze_file_callback(mp3_path, canonical_id)
        
        # Files in DB but not in cache -> mark evicted
        missing_files = db_entries - actual_files
        for canonical_id in missing_files:
            logger.warning(f"[CACHE_SYNC] Cache evicted detected: {canonical_id}")
            await self.db.update_cache_status(canonical_id, "evicted")
        
        logger.info(f"[CACHE_SYNC] Reconciled {cache_dir}: "
                   f"{len(actual_files)} files, {len(new_files)} new, {len(missing_files)} evicted")
    
    async def _get_db_cache_ids(self) -> set[str]:
        """Get all canonical_ids from DB that are cache tracks"""
        # This would need a new DB method - for now use a query
        import aiosqlite
        async with aiosqlite.connect(self.db.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT canonical_id FROM track_identity WHERE cache_filename IS NOT NULL"
            )
            rows = await cursor.fetchall()
            return {row["canonical_id"] for row in rows}
    
    def _find_file_for_canonical(self, cache_dir: Path, canonical_id: str) -> Path | None:
        """Find the actual file for a canonical ID"""
        expected_name = canonical_id.replace("track:cache:", "") + ".mp3"
        file_path = cache_dir / expected_name
        if file_path.exists():
            return file_path
        
        # Try fuzzy match
        for mp3 in cache_dir.glob("*.mp3"):
            if self._sanitize_filename(mp3.stem) == canonical_id.replace("track:cache:", ""):
                return mp3
        return None
    
    async def reconcile_orphaned(self):
        """Check for tracks in DB that are not in any cache dir"""
        import aiosqlite
        async with aiosqlite.connect(self.db.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT canonical_id FROM track_identity WHERE cache_filename IS NOT NULL"
            )
            rows = await cursor.fetchall()
        
        all_cache_files = set()
        for cache_dir in self.cache_dirs:
            for mp3 in cache_dir.glob("*.mp3"):
                canonical_id = f"track:cache:{self._sanitize_filename(mp3.stem)}"
                all_cache_files.add(canonical_id)
        
        for row in rows:
            canonical_id = row["canonical_id"]
            if canonical_id not in all_cache_files:
                # Check if it's a Spotify/YouTube track that just doesn't have a cache file
                # Only mark evicted if it's a pure cache track
                await self.db.update_cache_status(canonical_id, "evicted")
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        import re
        return re.sub(r'[^\w\-_]', '_', filename)


async def create_cache_sync_service(
    db: Database,
    cache_dirs: list[Path],
    analyzer,
    worker,
    reconcile_interval: int = 300
) -> CacheSyncService:
    """Factory function to create and start cache sync service"""
    service = CacheSyncService(db, cache_dirs, analyzer, worker, reconcile_interval)
    await service.start()
    return service