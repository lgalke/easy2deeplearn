import itertools
from typing import Callable, Generator, Iterable, List, Tuple, TypeVar, Union

import editdistance
import numpy as np
import scipy.stats as st
from numpy.typing import ArrayLike

Message = str
Messages = List[Message]
Meaning = Tuple[int, float]
Meanings = List[Meaning]



def form2meaning_ratio(meanings: Meanings, messages: Messages):
    """Computes the ratio between distinct meanings and distinct forms."""
    if isinstance(meanings, np.ndarray):
        assert meanings.ndim == 2
        num_meanings = np.unique(meanings, axis=0).shape[0]
    else:
        num_meanings = len(set(meanings))
    lexicon_size = len(set(messages))
    ratio = float(lexicon_size) / num_meanings
    # ambiguity_score = (1 - ratio) * 100
    # synonymity_score = (ratio - 1) * 100
    return ratio


def semantic_difference(meaning1: Meaning, meaning2: Meaning):
    """
    Sort-of simplified Hamming distance

    As used in: Compositional structure can emerge without generational transmission
    """
    shape1, angle1 = meaning1
    shape2, angle2 = meaning2
    diff_shape = 0.0 if (shape1 == shape2) else 1.0
    diff_angle = np.abs(angle1 - angle2) / 180.0
    return diff_shape + diff_angle


def normalized_editdistance(message: str, target_word: str) -> float:
    try:
        maxlen = max(len(message), len(target_word))
    except TypeError:
        print("Type error")
        print(message)
        print(type(message))
        print(target_word)
        exit(1)
    dist = editdistance.eval(message, target_word)

    try:
        dist /= maxlen
    except ZeroDivisionError:
        # Both sequences are empty
        return 0
    return dist


def production_similarity(message: str, target_word: str) -> float:
    """One minus length-normalized Levenshtein distance"""
    return 1 - normalized_editdistance(message, target_word)


def mean_production_similarity(messages: Messages, target_words: Messages):
    """Average production similarity across multiple messages/targets

    :messages: TODO
    :target_words: TODO
    :returns: TODO

    """
    assert len(messages) == len(target_words)
    return np.mean(
        [production_similarity(m, t) for m, t in zip(messages, target_words)]
    )


def binary_accuracy(
    messages: Messages, target_words: Messages
) -> Tuple[float, List[int]]:
    """Computes binary accuracy on a set of paired messages / targets"""
    assert len(messages) == len(target_words)
    N = len(target_words)
    hits = [1 if m == t else 0 for m, t in zip(messages, target_words)]
    return sum(hits) / N, hits


VALID_METRICS = {  # for generalization score
    "semantic_difference": semantic_difference,
    "editdistance": editdistance.eval,
    "normalized_editdistance": normalized_editdistance,
    "production_similarity": production_similarity,
}


R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T")


def pairwise(
    f: Callable[[R, S], T], xs: Iterable[R], ys: Iterable[S]
) -> Generator[T, None, None]:
    """Apply f to all combinations of x in xs and y in ys"""
    return itertools.starmap(f, itertools.product(xs, ys))


def pairwise_dedup(f: Callable[[S, S], T], xs: List[S]) -> Generator[T, None, None]:
    """Apply f to all combinations of x1, x2 in xs. Triangular without diagonal"""
    for i, a in enumerate(xs[:-1]):
        for b in xs[(i + 1) :]:
            yield f(a, b)


def cumulative_average(xs: Iterable) -> float:
    """Efficient computation of the cumulative average from an iterable, preferably a generator"""
    ca = 0.0
    for i, x in enumerate(xs):
        ca += (x - ca) / (i + 1)
    return ca


def generalization_score(
    familiar_scenes: List[Meaning],
    familiar_labels: List[Message],
    new_scenes: List[Meaning],
    new_labels: List[Message],
    scene_metric: Union[
        str, Callable[[Message, Message], float]
    ] = "semantic_difference",
    label_metric: Union[
        str, Callable[[Meaning, Meaning], float]
    ] = "normalized_editdistance",
    rescale: bool = True,
):
    """
    Generalization score (yet not normalized)
    As used in: What makes a language easy to learn

    """
    if not callable(scene_metric):
        scene_metric = VALID_METRICS[scene_metric]

    if not callable(label_metric):
        label_metric = VALID_METRICS[label_metric]

    # pairwise distances between each new scene and all familar scenes
    scene_distances = list(pairwise(scene_metric, familiar_scenes, new_scenes))
    label_distances = list(pairwise(label_metric, familiar_labels, new_labels))

    r, pval = st.pearsonr(scene_distances, label_distances)

    # we cannot normalize here because it's across participants/conditions...

    # Squash [-1, 1] into range [0,1]
    if rescale:
        print("Warning: legacy scaling of gen score.")
        r = (r + 1) / 2

    return r, pval


def normalize_correlation(x: List[float]):
    """Inspired by min-max normalization procedure (unity-based normalization / feature_scaling)
    As used in: What makes a language easy to learn
    """
    min_x = min(x)
    # TODO double check, don't forget to rescale
    return x - min_x / (max(x) - min_x)


def convergence_score(
    messages: Messages,
    metric: Union[str, Callable[[Message, Message], float]] = "production_similarity",
):
    """
    Edit distances between labels produced by different participants/NNs

    Limor's R Code:
        #levenstein distances between all pairs of words in a given list (convergence)
        allpairslev <- function(words) {
          l <- length(words)
          unlist(sapply(1:(l-1),function(x) sapply((x+1):l,function(y) normdist(words[x],words[y]))))
        }

        #calculate lev distances between the labels produced by different NNs at regularization test (final 100th iteration)
        conv <-lapply (langs, function(l) {sapply(reg_items, function(i)
              mean(allpairslev(regularization_test[regularization_test$Target_Item==i & regularization_test$Input_Condition==l,]$Word_Produced),na.rm=TRUE))})

        # THEN LATER do -1 while adding
        for (l in 1:10) {for (i in reg_items) {
        d[d$Task=="RegularizationTest" & d$Input_Condition==langs[l] & d$Target_Item==i & d$Iteration==100,"Reg_NN_Convergence"]<- 1- conv[[l]][match(i,reg_items)]}}


        for (l in 1:10) {for (i in reg_items) {
            d[d$Task=="RegularizationTest" & d$Input_Condition==langs[l] & d$Target_Item==i & d$Iteration==50,"Reg_NN_Convergence"]<- 1- conv_halftime[[l]][match(i,reg_items)]}}
    """
    if not callable(metric):
        metric = VALID_METRICS[metric]

    distances_gen = pairwise_dedup(metric, messages)
    avg = cumulative_average(distances_gen)

    return avg
