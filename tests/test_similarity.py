from riki_memory import cosine_similarity, trigram_similarity
from riki_memory.retrieval import fuse_scores, reciprocal_rank_fusion


def test_trigram_identical_is_one():
    assert trigram_similarity("Sub likes black coffee", "Sub likes black coffee") == 1.0


def test_trigram_case_insensitive():
    assert trigram_similarity("Riya works at Zerodha", "riya works at zerodha") == 1.0


def test_trigram_orders_correctly():
    near = trigram_similarity("the dentist appointment", "dentist appointment")
    far = trigram_similarity("the dentist appointment", "completely different words here")
    assert near > far
    assert 0.0 <= far <= 1.0


def test_trigram_empty():
    assert trigram_similarity("", "") == 1.0
    assert trigram_similarity("something", "") == 0.0


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_rrf_multi_channel_win():
    # found by two channels beats found by one, even at a worse rank
    ranked = reciprocal_rank_fusion(
        [["a", "b"], ["c", "a"]],
        weights=[1.0, 1.0],
    )
    assert ranked[0] == "a"


def test_rrf_weights_matter():
    ranked = reciprocal_rank_fusion([["a"], ["b"]], weights=[0.1, 10.0])
    assert ranked[0] == "b"
    assert fuse_scores([["a"]])["a"] > 0
