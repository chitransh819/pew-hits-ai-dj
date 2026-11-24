"""
Audio Analysis Engine for Pew Hits AI DJ
Uses librosa (DSP), OpenL3 (embeddings), EfficientAT (embeddings), custom classifiers
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Container for all extracted audio features"""
    # Low-level (librosa)
    bpm: float = 0.0
    beats: np.ndarray = field(default_factory=lambda: np.array([]))
    onset_times: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_contrast: np.ndarray = field(default_factory=lambda: np.array([]))
    chroma: np.ndarray = field(default_factory=lambda: np.array([]))
    key: str = ""
    loudness_lufs: float = 0.0
    
    # Embeddings
    openl3_embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    efficientat_embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Classifiers
    genre_top: str = ""
    genre_scores: Dict[str, float] = field(default_factory=dict)
    mood_tags: List[str] = field(default_factory=list)
    mood_scores: Dict[str, float] = field(default_factory=dict)
    danceability: float = 0.0
    energy: float = 0.0
    valence: float = 0.0
    acousticness: float = 0.0
    instrumentalness: float = 0.0
    vocal_presence: float = 0.0
    
    # Metadata
    analysis_version: str = "1.0"
    analyzed_at: float = field(default_factory=time.time)
    file_mtime: int = 0
    file_size: int = 0


class AudioAnalyzer:
    """Main audio analysis pipeline"""
    
    def __init__(
        self, 
        models_dir: Path, 
        sample_rate: int = 16000, 
        chunk_duration: int = 30,
        max_workers: int = 4
    ):
        self.models_dir = Path(models_dir)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = sample_rate * chunk_duration
        self.max_workers = max_workers
        
        # Models (lazy loaded)
        self._openl3_session: Optional[ort.InferenceSession] = None
        self._efficientat_session: Optional[ort.InferenceSession] = None
        self._classifiers: Dict[str, Any] = {}
        self._models_loaded = False
        
        # Model paths
        self.openl3_path = self.models_dir / "openl3_music_mel256_emb512.onnx"
        self.efficientat_path = self.models_dir / "efficientat_mn10_as.onnx"
        self.classifier_dir = self.models_dir / "classifiers"
        
    def load_models(self) -> bool:
        """Load all models at startup"""
        try:
            self._load_openl3()
            self._load_efficientat()
            self._load_classifiers()
            self._models_loaded = True
            logger.info("All audio analysis models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def _load_openl3(self) -> Optional[ort.InferenceSession]:
        """Load OpenL3 ONNX model"""
        if self._openl3_session is None:
            if not self.openl3_path.exists():
                logger.warning(f"OpenL3 model not found at {self.openl3_path}")
                return None
            self._openl3_session = ort.InferenceSession(
                str(self.openl3_path), 
                providers=['CPUExecutionProvider']
            )
            logger.info("OpenL3 model loaded")
        return self._openl3_session
    
    def _load_efficientat(self) -> Optional[ort.InferenceSession]:
        """Load EfficientAT ONNX model"""
        if self._efficientat_session is None:
            if not self.efficientat_path.exists():
                logger.warning(f"EfficientAT model not found at {self.efficientat_path}")
                return None
            self._efficientat_session = ort.InferenceSession(
                str(self.efficientat_path),
                providers=['CPUExecutionProvider']
            )
            logger.info("EfficientAT model loaded")
        return self._efficientat_session
    
    def _load_classifiers(self):
        """Load custom classifiers from joblib files"""
        try:
            import joblib
            clf_dir = self.classifier_dir
            if clf_dir.exists():
                for clf_name in ["genre", "mood", "danceability", "vocal_instrumental"]:
                    clf_path = clf_dir / f"{clf_name}.joblib"
                    if clf_path.exists():
                        self._classifiers[clf_name] = joblib.load(clf_path)
                        logger.info(f"Loaded classifier: {clf_name}")
                    else:
                        logger.debug(f"Classifier not found: {clf_path}")
            else:
                logger.warning(f"Classifier directory not found: {clf_dir}")
        except Exception as e:
            logger.warning(f"Failed to load classifiers: {e}")
    
    def analyze(self, file_path: Path) -> Optional[AudioFeatures]:
        """
        Full analysis pipeline for a single track
        Returns AudioFeatures or None on failure
        """
        start_time = time.time()
        
        try:
            # Load audio
            y, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            if len(y) == 0:
                logger.error(f"Empty audio file: {file_path}")
                return None
            
            # Take first chunk_duration seconds (or pad)
            if len(y) > self.chunk_samples:
                y = y[:self.chunk_samples]
            elif len(y) < self.chunk_samples:
                y = np.pad(y, (0, self.chunk_samples - len(y)))
            
            features = AudioFeatures()
            
            # 1. Librosa low-level features
            self._extract_librosa_features(y, sr, features)
            
            # 2. OpenL3 embedding
            self._extract_openl3(y, sr, features)
            
            # 3. EfficientAT embedding
            self._extract_efficientat(y, sr, features)
            
            # 4. Custom classifiers
            self._run_classifiers(features)
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Analyzed {file_path.name}: BPM={features.bpm:.1f}, "
                       f"Key={features.key}, Energy={features.energy:.2f} "
                       f"({elapsed:.0f}ms)")
            return features
            
        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}", exc_info=True)
            return None
    
    def _extract_librosa_features(self, y: np.ndarray, sr: int, features: AudioFeatures):
        """Extract low-level features using librosa"""
        # BPM and beats
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features.bpm = float(tempo)
        features.beats = librosa.frames_to_time(beats, sr=sr)
        features.onset_times = librosa.frames_to_time(
            librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr), sr=sr
        )
        
        # Spectral features
        stft = np.abs(librosa.stft(y))
        features.spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr)))
        features.spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr)))
        features.spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr)))
        features.spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr), axis=1)
        
        # Chroma and key
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.chroma = np.mean(chroma, axis=1)
        features.key = self._estimate_key(chroma)
        
        # Loudness (EBU R128 approximation)
        rms = librosa.feature.rms(y=y)
        features.loudness_lufs = float(20 * np.log10(np.mean(rms) + 1e-10))
    
    def _estimate_key(self, chroma: np.ndarray) -> str:
        """Simple key estimation from chroma"""
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        chroma_mean = np.mean(chroma, axis=1)
        chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-10)
        
        best_score = -1
        best_key = "C major"
        
        for i in range(12):
            # Major
            rotated_major = np.roll(major_profile, i)
            score_major = np.corrcoef(chroma_norm, rotated_major)[0, 1]
            if score_major > best_score:
                best_score = score_major
                best_key = f"{key_names[i]} major"
            
            # Minor
            rotated_minor = np.roll(minor_profile, i)
            score_minor = np.corrcoef(chroma_norm, rotated_minor)[0, 1]
            if score_minor > best_score:
                best_score = score_minor
                best_key = f"{key_names[i]} minor"
        
        return best_key
    
    def _extract_openl3(self, y: np.ndarray, sr: int, features: AudioFeatures):
        """Extract OpenL3 embedding"""
        session = self._load_openl3()
        if session is None:
            return
        
        try:
            # OpenL3 expects mel-spectrogram input
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=256, n_fft=1024, hop_length=512
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1] range
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)
            
            # OpenL3 expects shape: (batch, time, freq, 1) or similar
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            # Reshape based on expected input
            if len(input_shape) == 4:  # (batch, time, freq, channels)
                inp = mel_norm.T[np.newaxis, ..., np.newaxis].astype(np.float32)
            elif len(input_shape) == 3:  # (batch, freq, time)
                inp = mel_norm[np.newaxis, ...].astype(np.float32)
            else:
                inp = mel_norm.astype(np.float32)
            
            embedding = session.run(None, {input_name: inp})[0]
            features.openl3_embedding = embedding.flatten()
            
        except Exception as e:
            logger.warning(f"OpenL3 extraction failed: {e}")
    
    def _extract_efficientat(self, y: np.ndarray, sr: int, features: AudioFeatures):
        """Extract EfficientAT embedding"""
        session = self._load_efficientat()
        if session is None:
            return
        
        try:
            # EfficientAT typically expects log-mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=320
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)
            
            input_name = session.get_inputs()[0].name
            inp = mel_norm[np.newaxis, ..., np.newaxis].astype(np.float32)
            
            embedding = session.run(None, {input_name: inp})[0]
            features.efficientat_embedding = embedding.flatten()
            
        except Exception as e:
            logger.warning(f"EfficientAT extraction failed: {e}")
    
    def _run_classifiers(self, features: AudioFeatures):
        """Run custom classifiers on embeddings"""
        # Use OpenL3 embedding for classification (or concatenate both)
        if len(features.openl3_embedding) > 0:
            emb = features.openl3_embedding.reshape(1, -1)
        elif len(features.efficientat_embedding) > 0:
            emb = features.efficientat_embedding.reshape(1, -1)
        else:
            # Fallback: use chroma + spectral features
            emb = np.concatenate([
                features.chroma,
                [features.spectral_centroid, features.spectral_rolloff, 
                 features.spectral_bandwidth, features.bpm / 200.0]
            ]).reshape(1, -1)
        
        # Genre classifier
        if "genre" in self._classifiers:
            clf = self._classifiers["genre"]
            try:
                proba = clf.predict_proba(emb)[0]
                classes = clf.classes_
                features.genre_scores = {c: float(p) for c, p in zip(classes, proba)}
                features.genre_top = classes[np.argmax(proba)]
            except Exception as e:
                logger.warning(f"Genre classification failed: {e}")
        
        # Mood classifier
        if "mood" in self._classifiers:
            clf = self._classifiers["mood"]
            try:
                proba = clf.predict_proba(emb)[0]
                classes = clf.classes_
                features.mood_scores = {c: float(p) for c, p in zip(classes, proba)}
                features.mood_tags = [c for c, p in features.mood_scores.items() if p > 0.5]
            except Exception as e:
                logger.warning(f"Mood classification failed: {e}")
        
        # Danceability
        if "danceability" in self._classifiers:
            clf = self._classifiers["danceability"]
            try:
                features.danceability = float(clf.predict_proba(emb)[0][1])
            except Exception as e:
                logger.warning(f"Danceability classification failed: {e}")
        
        # Vocal/Instrumental
        if "vocal_instrumental" in self._classifiers:
            clf = self._classifiers["vocal_instrumental"]
            try:
                proba = clf.predict_proba(emb)[0]
                classes = clf.classes_
                for c, p in zip(classes, proba):
                    if "vocal" in c.lower() or "voice" in c.lower():
                        features.vocal_presence = float(p)
                    elif "instrumental" in c.lower():
                        features.instrumentalness = float(p)
            except Exception as e:
                logger.warning(f"Vocal/Instrumental classification failed: {e}")
        
        # Heuristic energy/valence/acousticness from low-level features
        features.energy = min(1.0, max(0.0, features.loudness_lufs / -10.0 + 0.5))
        features.valence = features.mood_scores.get("happy", 0.5)
        features.acousticness = features.mood_scores.get("acoustic", 0.5)
    
    def to_dict(self, features: AudioFeatures) -> Dict[str, Any]:
        """Convert AudioFeatures to dictionary for database storage"""
        return {
            "bpm": features.bpm,
            "beats": features.beats.tolist() if len(features.beats) > 0 else [],
            "onset_times": features.onset_times.tolist() if len(features.onset_times) > 0 else [],
            "spectral_centroid": features.spectral_centroid,
            "spectral_rolloff": features.spectral_rolloff,
            "spectral_bandwidth": features.spectral_bandwidth,
            "spectral_contrast": features.spectral_contrast.tolist() if len(features.spectral_contrast) > 0 else [],
            "chroma": features.chroma.tolist() if len(features.chroma) > 0 else [],
            "key": features.key,
            "loudness_lufs": features.loudness_lufs,
            "openl3_embedding": features.openl3_embedding.tobytes() if len(features.openl3_embedding) > 0 else b"",
            "efficientat_embedding": features.efficientat_embedding.tobytes() if len(features.efficientat_embedding) > 0 else b"",
            "genre_top": features.genre_top,
            "genre_scores": features.genre_scores,
            "mood_tags": features.mood_tags,
            "mood_scores": features.mood_scores,
            "danceability": features.danceability,
            "energy": features.energy,
            "valence": features.valence,
            "acousticness": features.acousticness,
            "instrumentalness": features.instrumentalness,
            "vocal_presence": features.vocal_presence,
            "analysis_version": features.analysis_version,
        }


class AnalysisWorker:
    """Background worker for incremental analysis"""
    
    def __init__(self, analyzer: AudioAnalyzer, db, cache_dirs: List[Path]):
        self.analyzer = analyzer
        self.db = db
        self.cache_dirs = cache_dirs
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._process_queue())
    
    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def queue_file(self, file_path: Path, canonical_id: str, force: bool = False):
        await self._queue.put((file_path, canonical_id, force))
    
    async def _process_queue(self):
        while self._running:
            try:
                file_path, canonical_id, force = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                await self._analyze_file(file_path, canonical_id, force)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def _analyze_file(self, file_path: Path, canonical_id: str, force: bool):
        """Analyze a single file and store results"""
        try:
            # Check if already analyzed and unchanged
            if not force:
                stat = file_path.stat()
                existing = await self.db.get_track_features(canonical_id)
                if (existing and existing.get("file_mtime") == int(stat.st_mtime) 
                    and existing.get("file_size") == stat.st_size):
                    logger.debug(f"Skipping unchanged: {file_path.name}")
                    return
            
            # Run analysis in thread pool (CPU-bound)
            loop = asyncio.get_event_loop()
            features = await loop.run_in_executor(
                None, self.analyzer.analyze, file_path
            )
            
            if features:
                feature_dict = self.analyzer.to_dict(features)
                feature_dict["file_mtime"] = int(file_path.stat().st_mtime)
                feature_dict["file_size"] = file_path.stat().st_size
                await self.db.upsert_track_features(canonical_id, feature_dict)
                logger.info(f"Analyzed and stored: {canonical_id}")
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")


class StartupScanner:
    """Scans cache directories on startup for bulk analysis"""
    
    def __init__(self, analyzer: AudioAnalyzer, db, cache_dirs: List[Path], worker: AnalysisWorker):
        self.analyzer = analyzer
        self.db = db
        self.cache_dirs = cache_dirs
        self.worker = worker
    
    async def scan_and_queue(self, force: bool = False) -> int:
        """Scan all cache directories and queue files for analysis"""
        queued = 0
        
        for cache_dir in self.cache_dirs:
            if not cache_dir.exists():
                logger.warning(f"Cache directory not found: {cache_dir}")
                continue
            
            logger.info(f"Scanning {cache_dir}...")
            
            for mp3_path in cache_dir.glob("*.mp3"):
                # Generate canonical ID from filename
                canonical_id = f"track:cache:{self._sanitize_filename(mp3_path.stem)}"
                
                # Check if already in DB
                existing = await self.db.get_track_features(canonical_id)
                if existing and not force:
                    # Check if file changed
                    stat = mp3_path.stat()
                    if (existing.get("file_mtime") == int(stat.st_mtime) 
                        and existing.get("file_size") == stat.st_size):
                        continue
                
                # Queue for analysis
                await self.worker.queue_file(mp3_path, canonical_id, force)
                queued += 1
        
        logger.info(f"Queued {queued} files for analysis")
        return queued
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize filename for canonical ID"""
        import re
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name).lower()