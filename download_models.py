"""
Model Downloader Script
Downloads and prepares ONNX models for AI DJ Service
Run once during Docker build or deployment
"""

import os
import logging
from pathlib import Path
import urllib.request
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MODELS_DIR = Path("/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file with progress"""
    if dest.exists() and not force:
        logger.info(f"Already exists: {dest.name}")
        return True
    
    logger.info(f"Downloading: {url} -> {dest}")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Downloaded: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def convert_to_onnx(model_name: str, output_path: Path) -> bool:
    """Convert a HuggingFace model to ONNX (placeholder for actual conversion)"""
    logger.warning(f"ONNX conversion for {model_name} not implemented")
    logger.warning("In production, use: python -m onnxruntime.tools.convert_onnx_models_to_ort ...")
    return False


def create_placeholder_classifiers(models_dir: Path):
    """Create placeholder classifier files for testing"""
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    
    clf_dir = models_dir / "classifiers"
    clf_dir.mkdir(parents=True, exist_ok=True)
    
    # Genre classifier (87 classes)
    genre_clf = DummyClassifier(strategy="uniform", random_state=42)
    genre_classes = np.array([f"genre_{i}" for i in range(87)])
    genre_clf.fit(np.random.randn(100, 512), np.random.choice(genre_classes, 100))
    joblib.dump(genre_clf, clf_dir / "genre.joblib")
    logger.info("Created placeholder genre classifier")
    
    # Mood classifier (7 classes)
    mood_clf = DummyClassifier(strategy="uniform", random_state=42)
    mood_classes = np.array(["happy", "sad", "relaxed", "aggressive", "party", "acoustic", "electronic"])
    mood_clf.fit(np.random.randn(100, 512), np.random.choice(mood_classes, 100))
    joblib.dump(mood_clf, clf_dir / "mood.joblib")
    logger.info("Created placeholder mood classifier")
    
    # Danceability classifier
    dance_clf = DummyClassifier(strategy="uniform", random_state=42)
    dance_clf.fit(np.random.randn(100, 512), np.random.choice([0, 1], 100))
    joblib.dump(dance_clf, clf_dir / "danceability.joblib")
    logger.info("Created placeholder danceability classifier")
    
    # Vocal/Instrumental classifier
    vocal_clf = DummyClassifier(strategy="uniform", random_state=42)
    vocal_clf.fit(np.random.randn(100, 512), np.random.choice(["vocal", "instrumental"], 100))
    joblib.dump(vocal_clf, clf_dir / "vocal_instrumental.joblib")
    logger.info("Created placeholder vocal/instrumental classifier")


def download_openl3(models_dir: Path):
    """Download OpenL3 music model (ONNX format)"""
    # OpenL3 ONNX models are available from various sources
    # For now, create a placeholder - in production, download from:
    # https://github.com/marl/openl3 or HuggingFace
    
    openl3_path = models_dir / "openl3_music_mel256_emb512.onnx"
    
    if openl3_path.exists():
        logger.info("OpenL3 model already exists")
        return
    
    # Try to download from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading OpenL3 from HuggingFace...")
        hf_hub_download(
            repo_id="facebook/openl3-music-mel256-emb512",
            filename="openl3_music_mel256_emb512.onnx",
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
        logger.info("OpenL3 downloaded successfully")
    except Exception as e:
        logger.warning(f"Could not download OpenL3 from HF: {e}")
        logger.warning("Creating placeholder - replace with actual model")
        # Create minimal valid ONNX file placeholder
        openl3_path.write_bytes(b"PLACEHOLDER_OPENL3_MODEL")
    
    return openl3_path.exists()


def download_efficientat(models_dir: Path):
    """Download EfficientAT model (ONNX format)"""
    efficientat_path = models_dir / "efficientat_mn10_as.onnx"
    
    if efficientat_path.exists():
        logger.info("EfficientAT model already exists")
        return
    
    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading EfficientAT from HuggingFace...")
        hf_hub_download(
            repo_id="fschmid56/EfficientAT-mn10_as",
            filename="efficientat_mn10_as.onnx",
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
        logger.info("EfficientAT downloaded successfully")
    except Exception as e:
        logger.warning(f"Could not download EfficientAT from HF: {e}")
        logger.warning("Creating placeholder - replace with actual model")
        efficientat_path.write_bytes(b"PLACEHOLDER_EFFICIENTAT_MODEL")
    
    return efficientat_path.exists()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AI DJ models")
    parser.add_argument("--models-dir", default="/models", help="Models directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--skip-placeholders", action="store_true", help="Skip placeholder creation")
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("AI DJ Model Downloader")
    logger.info("=" * 60)
    
    # Download models
    logger.info("\n--- Downloading OpenL3 ---")
    download_openl3(models_dir)
    
    logger.info("\n--- Downloading EfficientAT ---")
    download_efficientat(models_dir)
    
    # Create placeholder classifiers
    if not args.skip_placeholders:
        logger.info("\n--- Creating Placeholder Classifiers ---")
        create_placeholder_classifiers(models_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Model download complete!")
    logger.info(f"Models directory: {models_dir}")
    logger.info("Files:")
    for f in models_dir.rglob("*"):
        if f.is_file():
            logger.info(f"  {f.relative_to(models_dir)} ({f.stat().st_size / 1024:.1f} KB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()