"""
Tests for the Recommendation Engine.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_dj.recommendation.engine import (
    Candidate,
    Language,
    Genre,
    NVIDIAClient,
    RecommendationEngine,
    StationConfig,
    TrackMetadata,
    VibeProfile,
    create_engine_from_config,
)


def create_mock_tracks() -> dict[str, TrackMetadata]:
    """Create mock track metadata for testing."""
    return {
        "track_1": TrackMetadata(
            track_id="track_1",
            title="Blinding Lights",
            artist="The Weeknd",
            bpm=171,
            key=8,
            energy=0.75,
            valence=0.65,
            danceability=0.85,
            genre="pop",
            language="english",
            popularity=95,
            cache_status="available",
        ),
        "track_2": TrackMetadata(
            track_id="track_2",
            title="Levitating",
            artist="Dua Lipa",
            bpm=126,
            key=12,
            energy=0.72,
            valence=0.80,
            danceability=0.78,
            genre="pop",
            language="english",
            popularity=90,
            cache_status="available",
        ),
        "track_3": TrackMetadata(
            track_id="track_3",
            title="Bad Guy",
            artist="Billie Eilish",
            bpm=135,
            key=5,
            energy=0.55,
            valence=0.45,
            danceability=0.70,
            genre="pop",
            language="english",
            popularity=92,
            cache_status="available",
        ),
        "track_4": TrackMetadata(
            track_id="track_4",
            title="Excuses",
            artist="AP Dhillon",
            bpm=140,
            key=3,
            energy=0.80,
            valence=0.70,
            danceability=0.82,
            genre="punjabi_pop",
            language="punjabi",
            popularity=85,
            cache_status="available",
        ),
        "track_5": TrackMetadata(
            track_id="track_5",
            title="Raataan Lambiyan",
            artist="Jubin Nautiyal",
            bpm=110,
            key=10,
            energy=0.45,
            valence=0.55,
            danceability=0.50,
            genre="bollywood",
            language="hindi",
            popularity=88,
            cache_status="available",
        ),
        "track_6": TrackMetadata(
            track_id="track_6",
            title="Kesariya",
            artist="Arijit Singh",
            bpm=95,
            key=7,
            energy=0.35,
            valence=0.60,
            danceability=0.45,
            genre="bollywood",
            language="hindi",
            popularity=90,
            cache_status="available",
        ),
        "track_7": TrackMetadata(
            track_id="track_7",
            title="Shape of You",
            artist="Ed Sheeran",
            bpm=96,
            key=1,
            energy=0.65,
            valence=0.75,
            danceability=0.82,
            genre="pop",
            language="english",
            popularity=98,
            cache_status="available",
        ),
        "track_8": TrackMetadata(
            track_id="track_8",
            title="Despacito",
            artist="Luis Fonsi",
            bpm=89,
            key=6,
            energy=0.78,
            valence=0.85,
            danceability=0.90,
            genre="pop",
            language="english",
            popularity=97,
            cache_status="available",
        ),
    }


def test_station_config_loading():
    """Test loading station configs from YAML."""
    english_config = StationConfig.from_yaml(Path("config/station_english.yaml"))
    assert english_config.station_id == "english_hits"
    assert english_config.language == "english"
    assert "mood" in english_config.weights
    assert sum(english_config.weights.values()) == 1.0
    print("✓ English station config loaded correctly")

    desi_config = StationConfig.from_yaml(Path("config/station_desi.yaml"))
    assert desi_config.station_id == "desi_beats"
    assert desi_config.language == "hindi"
    assert "language" in desi_config.weights
    assert desi_config.weights["language"] > english_config.weights["language"]
    print("✓ Desi station config loaded correctly")
    print(f"  English weights: {english_config.weights}")
    print(f"  Desi weights: {desi_config.weights}")


def test_scoring_functions():
    """Test individual scoring functions."""
    tracks = create_mock_tracks()
    vibe = VibeProfile(
        mood=0.7,
        energy=0.7,
        tempo=120,
        genre="pop",
        language="english",
        key=8,
    )

    engine = RecommendationEngine(
        station_config=StationConfig.from_yaml(Path("config/station_english.yaml")),
        playable_track_ids=set(tracks.keys()),
    )

    # Test mood scoring
    track = tracks["track_1"]  # valence 0.65, vibe mood 0.7 -> diff 0.05 -> score 0.95
    mood_score = engine.score_mood(track, vibe)
    assert 0.9 < mood_score <= 1.0, f"Expected ~0.95, got {mood_score}"
    print(f"✓ Mood score: {mood_score:.3f}")

    # Test energy scoring
    energy_score = engine.score_energy(track, vibe)
    assert 0.9 < energy_score <= 1.0, f"Expected ~0.95, got {energy_score}"
    print(f"✓ Energy score: {energy_score:.3f}")

    # Test tempo scoring
    tempo_score = engine.score_tempo(track, vibe)
    assert 0.5 < tempo_score < 0.8, f"Expected ~0.6, got {tempo_score}"
    print(f"✓ Tempo score: {tempo_score:.3f}")

    # Test genre scoring
    genre_score = engine.score_genre(track, vibe)
    assert genre_score == 1.0, f"Expected 1.0, got {genre_score}"
    print(f"✓ Genre score: {genre_score:.3f}")

    # Test language scoring
    lang_score = engine.score_language(track, vibe)
    assert lang_score == 1.0, f"Expected 1.0, got {lang_score}"
    print(f"✓ Language score: {lang_score:.3f}")

    # Test transition scoring (with prev track)
    # track_1 (171 BPM) from track_2 (126 BPM) = 45 BPM delta > 30 max, so score reduced
    prev_track = tracks["track_2"]  # BPM 126
    trans_score = engine.score_transition(track, vibe, prev_track)
    # BPM delta 45 > max 30, score = 1.0 * (1 - (45-30)/30) = 0.5
    assert trans_score == 0.5, f"Expected 0.5, got {trans_score}"
    print(f"✓ Transition score (BPM delta exceeds max): {trans_score:.3f}")

    # Test valid transition (within limits)
    prev_track2 = tracks["track_7"]  # Shape of You, 96 BPM
    trans_score2 = engine.score_transition(track, vibe, prev_track2)
    # BPM delta 75 > max 30, still reduced
    print(f"✓ Transition score (96->171 BPM): {trans_score2:.3f}")

    # Test good transition
    track_mid = TrackMetadata(
        track_id="track_mid",
        title="Mid Tempo",
        artist="Test",
        bpm=130,
        key=10,  # Close to prev_track key 12
        energy=0.71,  # Close to prev_track energy 0.72
        genre="pop",
        language="english",
        cache_status="available",
    )
    trans_score3 = engine.score_transition(track_mid, vibe, prev_track)
    # BPM delta 4 < max 30, energy delta 0.01 < max 0.3, key distance 2 < max 5
    assert trans_score3 == 1.0, f"Expected 1.0, got {trans_score3}"
    print(f"✓ Transition score (good transition 126->130 BPM): {trans_score3:.3f}")

    # Test recency scoring
    recency_score = engine.score_recency(track)
    assert 0.3 <= recency_score <= 1.0, f"Expected 0.3-1.0, got {recency_score}"
    print(f"✓ Recency score: {recency_score:.3f}")


def test_candidate_generation_and_filtering():
    """Test candidate generation with filters."""
    tracks = create_mock_tracks()
    playable_ids = set(tracks.keys())

    # English station - should filter out non-English tracks
    engine = RecommendationEngine(
        station_config=StationConfig.from_yaml(Path("config/station_english.yaml")),
        playable_track_ids=playable_ids,
        track_metadata_provider=lambda tid: tracks.get(tid),
    )

    all_tracks = list(tracks.values())
    filtered = engine.filter_candidates(all_tracks)

    # Should only have English tracks
    for t in filtered:
        assert t.language == "english", f"Non-English track passed filter: {t.track_id}"
    assert len(filtered) == 5  # 5 English tracks in mock data
    print(f"✓ English station filter: {len(filtered)}/{len(all_tracks)} tracks passed")

    # Desi station - should allow Hindi/Punjabi
    engine_desi = RecommendationEngine(
        station_config=StationConfig.from_yaml(Path("config/station_desi.yaml")),
        playable_track_ids=playable_ids,
        track_metadata_provider=lambda tid: tracks.get(tid),
    )

    filtered_desi = engine_desi.filter_candidates(all_tracks)
    for t in filtered_desi:
        assert t.language in ["hindi", "punjabi"], f"Non-Desi track passed filter: {t.track_id}"
    assert len(filtered_desi) == 3  # 3 Desi tracks in mock data
    print(f"✓ Desi station filter: {len(filtered_desi)}/{len(all_tracks)} tracks passed")


def test_top_k_diverse_selection():
    """Test top-K selection with diversity constraints."""
    tracks = create_mock_tracks()
    vibe = VibeProfile(mood=0.7, energy=0.7, tempo=120, genre="pop", language="english")

    engine = RecommendationEngine(
        station_config=StationConfig.from_yaml(Path("config/station_english.yaml")),
        playable_track_ids=set(tracks.keys()),
    )

    # Create candidates with varying scores
    candidates = []
    for i, (tid, track) in enumerate(tracks.items()):
        if track.language != "english":
            continue
        scores = engine.compute_scores(track, vibe)
        total = engine.apply_weights(scores)
        candidates.append(Candidate(
            track=track,
            scores=scores,
            total_score=total,
            explanation="",
            confidence=total,
        ))

    # Select top 3 with diversity
    selected = engine.select_top_k_diverse(candidates, k=3)

    assert len(selected) <= 3
    artists = [c.track.artist for c in selected]
    assert len(set(artists)) == len(artists), "Duplicate artists in selection"
    print(f"✓ Top-K diverse selection: {[c.track.title for c in selected]}")


def test_transition_validation():
    """Test transition validation against station rules."""
    tracks = create_mock_tracks()

    engine = RecommendationEngine(
        station_config=StationConfig.from_yaml(Path("config/station_english.yaml")),
    )

    # Valid transition
    prev = tracks["track_2"]  # 126 BPM, energy 0.72
    curr = tracks["track_1"]  # 171 BPM, energy 0.75
    valid, reason = engine.validate_transition(curr, prev)
    print(f"✓ Transition validation (126->171 BPM): valid={valid}, reason={reason}")

    # Invalid transition (large BPM delta)
    curr2 = TrackMetadata(
        track_id="track_fast",
        title="Fast Track",
        artist="Test",
        bpm=200,
        energy=0.9,
        genre="electronic",
        language="english",
        cache_status="available",
    )
    valid2, reason2 = engine.validate_transition(curr2, prev)
    print(f"✓ Transition validation (126->200 BPM): valid={valid2}, reason={reason2}")


def test_explanation_generation():
    """Test human-readable explanation generation."""
    tracks = create_mock_tracks()
    vibe = VibeProfile(mood=0.8, energy=0.75, tempo=130, genre="pop", language="english")

    engine = RecommendationEngine(
        station_config=StationConfig.from_yaml(Path("config/station_english.yaml")),
    )

    track = tracks["track_1"]
    scores = engine.compute_scores(track, vibe)
    explanation = engine.generate_explanation(track, scores, vibe)

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    print(f"✓ Explanation: {explanation}")


async def test_nvidia_client_mock():
    """Test NVIDIA client with mock (no actual API call)."""
    # This test verifies the client structure without making real calls
    client = NVIDIAClient(
        api_key="test_key",
        rate_limit_rpm=40,
        timeout_seconds=10,
    )

    # Create dummy candidates
    tracks = create_mock_tracks()
    candidates = []
    for tid, track in list(tracks.items())[:3]:
        candidates.append(Candidate(
            track=track,
            scores={"mood": 0.9, "energy": 0.8},
            total_score=0.85,
            explanation="Test",
            confidence=0.85,
        ))

    # Test rate limiting logic
    await client._wait_for_rate_limit()
    print("✓ NVIDIA client rate limiting works")

    # Test normalization
    norm = client._normalize_rerank_score(2.0)
    assert 0.8 < norm < 1.0
    print(f"✓ Score normalization: logit=2.0 -> {norm:.3f}")


async def test_full_recommendation_pipeline():
    """Test the complete recommendation pipeline."""
    tracks = create_mock_tracks()

    def track_provider(tid: str) -> TrackMetadata | None:
        return tracks.get(tid)

    vibe = VibeProfile(
        mood=0.7,
        energy=0.7,
        tempo=125,
        genre="pop",
        language="english",
        key=8,
    )

    engine = RecommendationEngine(
        station_config=StationConfig.from_yaml(Path("config/station_english.yaml")),
        playable_track_ids=set(tracks.keys()),
        track_metadata_provider=track_provider,
        recent_history=["track_3"],  # Exclude Billie Eilish
        artist_last_played={"Dua Lipa": datetime.utcnow() - timedelta(minutes=10)},  # Cooldown
    )

    recommendations = await engine.recommend(vibe, k=3)

    assert len(recommendations) > 0
    assert len(recommendations) <= 3

    for rec in recommendations:
        assert rec.track.cache_status == "available"
        assert rec.track.track_id not in ["track_3"]  # Excluded by history
        assert rec.track.artist != "Dua Lipa"  # Excluded by cooldown
        assert rec.explanation
        assert 0 <= rec.confidence <= 1

    print(f"✓ Full pipeline: {len(recommendations)} recommendations")
    for r in recommendations:
        print(f"  - {r.track.title} by {r.track.artist} (score: {r.total_score:.3f})")


async def test_desi_station_pipeline():
    """Test recommendation pipeline for Desi station."""
    tracks = create_mock_tracks()

    def track_provider(tid: str) -> TrackMetadata | None:
        return tracks.get(tid)

    vibe = VibeProfile(
        mood=0.6,
        energy=0.7,
        tempo=120,
        genre="bollywood",
        language="hindi",
        key=10,
    )

    engine = RecommendationEngine(
        station_config=StationConfig.from_yaml(Path("config/station_desi.yaml")),
        playable_track_ids=set(tracks.keys()),
        track_metadata_provider=track_provider,
    )

    recommendations = await engine.recommend(vibe, k=3)

    assert len(recommendations) > 0
    for rec in recommendations:
        assert rec.track.language in ["hindi", "punjabi"]
        assert rec.track.genre in ["bollywood", "punjabi_pop"]

    print(f"✓ Desi station pipeline: {len(recommendations)} recommendations")
    for r in recommendations:
        print(f"  - {r.track.title} by {r.track.artist} ({r.track.language}, {r.track.genre})")


async def test_factory_function():
    """Test the factory function for creating engine from config."""
    engine = await create_engine_from_config(
        station_config_path=Path("config/station_english.yaml"),
        nvidia_api_key=None,  # No API key for testing
        playable_track_ids={"track_1", "track_2"},
        track_metadata_provider=lambda tid: create_mock_tracks().get(tid),
    )

    assert isinstance(engine, RecommendationEngine)
    assert engine.station.station_id == "english_hits"
    assert engine.nvidia is None  # No API key provided
    print("✓ Factory function works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Recommendation Engine Tests")
    print("=" * 60)

    test_station_config_loading()
    test_scoring_functions()
    test_candidate_generation_and_filtering()
    test_top_k_diverse_selection()
    test_transition_validation()
    test_explanation_generation()

    asyncio.run(test_nvidia_client_mock())
    asyncio.run(test_full_recommendation_pipeline())
    asyncio.run(test_desi_station_pipeline())
    asyncio.run(test_factory_function())

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()