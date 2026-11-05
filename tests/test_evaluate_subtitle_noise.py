from scripts.evaluate_subtitle_noise import delete_words, paired
from video_rag.schemas import VideoSegment


def test_deletion_is_deterministic_and_preserves_segment_times():
    segment = VideoSegment('s', 'v', 'local', 1, 3, transcript=' '.join(f'word{i}' for i in range(100)))
    original = [segment]
    first, second = delete_words(original), delete_words(original)
    assert first == second
    assert first[0].start_time == 1 and first[0].end_time == 3
    assert 0 < len(first[0].transcript.split()) < 100
    assert original[0].transcript == segment.transcript


def test_paired_counts_actual_top_three_changes():
    clean = {'_outcomes': {'a': (True, True), 'b': (False, True), 'c': (False, False)}}
    noisy = {'_outcomes': {'a': (False, False), 'b': (True, True), 'c': (True, True)}}
    assert paired(clean, noisy) == {'both_overlap': 1, 'clean_only': 1, 'noisy_only': 1, 'neither_overlap': 0}
