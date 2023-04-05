from collections import namedtuple


Config = namedtuple("Experiment", ["alphabet",
                                   "max_length",
                                   "num_distractors",
                                   "num_shapes",
                                   "num_angles"])


# "Larger Communities create more systematic languages"
Raviv1 = Config(
    frozenset("aeiou-wtpsfghknm"),
    16,
    7,
    4,
    None)

