import torch

from measures import (
    binary_accuracy,
    convergence_score,
    cumulative_average,
    normalized_editdistance,
    pairwise_dedup,
    production_similarity,
    semantic_difference,
)

EPS = 1e-5


def test_semantic_difference():
    # maximum
    assert semantic_difference((1, 180), (2, 0)) == 2.0

    # middle
    assert semantic_difference((2, 135), (1, 0)) == 1.75
    assert semantic_difference((2, 90), (1, 0)) == 1.5
    assert semantic_difference((2, 90), (1, 45)) == 1.25
    assert semantic_difference((1, 180), (1, 0)) == 1.0
    assert semantic_difference((1, 180), (1, 90)) == 0.5
    assert semantic_difference((3, 90), (3, 0)) == 0.5
    assert semantic_difference((1, 45), (1, 90)) == 0.25

    # minimum
    assert semantic_difference((1, 180), (1, 180)) == 0.0


def test_binary_accuracy():

    acc, correct = binary_accuracy(["a", "b", "c"], ["a", "b", "c"])
    assert acc == 1.0
    assert all(correct)
    acc, correct = binary_accuracy(["aa", "bbb", "c"], ["aa", "bbb", "c"])
    assert acc == 1.0
    assert all(correct)
    acc, correct = binary_accuracy(["aa", "bxb"], ["aa", "bbb"])
    assert acc == 0.5
    assert correct[0] == 1
    assert correct[1] == 0


def test_normalized_editdistance():
    assert normalized_editdistance("a", "a") == 0
    assert normalized_editdistance("a", "b") == 1
    assert normalized_editdistance("abcd", "ab") == 2 / 4
    assert abs(normalized_editdistance("banana", "bahama") - 2 / 6) < EPS
    assert normalized_editdistance("cdefgah", "cdefgah") == 0


def test_production_similarity():
    assert production_similarity("a", "a") == 1
    assert production_similarity("a", "b") == 0
    assert production_similarity("abcd", "ab") == 2 / 4
    assert abs(production_similarity("banana", "bahama") - 4 / 6) < EPS
    assert production_similarity("cdefgah", "cdefgah") == 1


def test_pairwise_dedup():
    xs = []
    result = list(pairwise_dedup(lambda a, b: a - b, xs))
    assert result == []

    xs = [20, 10]
    result = list(pairwise_dedup(lambda a, b: a - b, xs))
    assert result == [10]

    xs = [5, 4, 3]
    result = list(pairwise_dedup(lambda a, b: a - b, xs))
    assert result == [1, 2, 1]


def test_pairwise_dedup_editdistance():
    """These tests reflect the behaviour of the orig R script
    > allpairslev(c("aa", "ab", "ac", "de"))
    [1] 0.5 0.5 1.0 0.5 1.0 1.0
    """

    xs = ["aa", "bb"]
    result = list(pairwise_dedup(normalized_editdistance, xs))
    assert result == [1.0]

    xs = ["aa", "bb", "cc"]
    result = list(pairwise_dedup(normalized_editdistance, xs))
    assert result == [1.0, 1.0, 1.0]

    xs = ["aa", "ab", "ac", "de"]
    result = list(pairwise_dedup(normalized_editdistance, xs))
    assert result == [0.5, 0.5, 1.0, 0.5, 1.0, 1.0]


def test_cumulative_average():
    x = [10, 20, 30, 40, 50]
    ca = cumulative_average(x)
    assert ca == 30

    x = range(1, 100)
    ca = cumulative_average(x)
    assert ca == 50


def test_convergence_score_with_prodsim():
    msgs = ["aaa", "aaa", "aaa", "aaa"]
    assert convergence_score(msgs, metric="production_similarity") == 1.0

    msgs = ["a", "b"]
    assert convergence_score(msgs, metric="production_similarity") == 0.0

    msgs = ["aa", "bb"]
    assert convergence_score(msgs, metric="production_similarity") == 0.0

    msgs = ["ab", "bb"]
    assert convergence_score(msgs, metric="production_similarity") == 0.5

    msgs = ["a", "b", "c", "d"]
    assert convergence_score(msgs, metric="production_similarity") == 0.0

    msgs = ["a", "a", "b", "b"]
    assert abs(convergence_score(msgs, metric="production_similarity") - (1 / 3)) < 1e-8

    msgs = ["aa", "aa", "bb", "bb"]
    assert abs(convergence_score(msgs, metric="production_similarity") - (1 / 3)) < 1e-8

    msgs = ["aa", "aa", "bb", "bb", "cc", "cc"]
    assert abs(convergence_score(msgs, metric="production_similarity") - 0.20) < 1e-8

    msgs = ["aa", "aa", "bb", "bb", "ccaa", "ccaa"]
    assert abs(convergence_score(msgs, metric="production_similarity") - (1 / 3)) < 1e-8


def test_convergence_score_with_normalized_editdistance():
    msgs = ["aaa", "aaa", "aaa", "aaa"]
    assert convergence_score(msgs, metric="normalized_editdistance") == 0.0

    msgs = ["a", "b"]
    assert convergence_score(msgs, metric="normalized_editdistance") == 1.0

    msgs = ["aa", "bb"]
    assert convergence_score(msgs, metric="normalized_editdistance") == 1.0

    msgs = ["ab", "bb"]
    assert convergence_score(msgs, metric="normalized_editdistance") == 0.5

    msgs = ["a", "b", "c", "d"]
    assert convergence_score(msgs, metric="normalized_editdistance") == 1.0

    msgs = ["a", "a", "b", "b"]
    assert (
        abs(convergence_score(msgs, metric="normalized_editdistance") - (2 / 3)) < 1e-8
    )

    msgs = ["aa", "aa", "bb", "bb"]
    assert (
        abs(convergence_score(msgs, metric="normalized_editdistance") - (2 / 3)) < 1e-8
    )

    msgs = ["aa", "aa", "bb", "bb", "cc", "cc"]
    assert abs(convergence_score(msgs, metric="normalized_editdistance") - 0.80) < 1e-8

    msgs = ["aa", "aa", "bb", "bb", "ccaa", "ccaa"]
    assert (
        abs(convergence_score(msgs, metric="normalized_editdistance") - (2 / 3)) < 1e-8
    )
