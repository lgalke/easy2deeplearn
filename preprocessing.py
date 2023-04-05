import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import pandas as pd

from tokenization import Tokenizer
from language import Language
from data import Data

from collections import namedtuple

from config import Config, Raviv1


def sin_cos_angles(angles):
    """Transforms an array-like to sinus and cosinus values"""
    angles = torch.tensor(angles, dtype=torch.float)
    x_radial = angles * np.pi / 180  # Radial
    x = torch.stack([torch.sin(x_radial), torch.cos(x_radial)], dim=-1)
    return x


class Preprocessor(object):

    """Transform dataframe rows into tensors"""

    def __init__(self, config: Config, language: Language):
        """Preprocess data for various experimental settings

        :language: pd.DataFrame to resolve language[target] == label string
        :vocab: Mapping from symbols to int

        """
        self.language = language
        self.config = config
        self.tokenizer = Tokenizer(config)

        self._distractor_cols = ["Distr%d" % i for i in range(1, config.num_distractors)]


    def _one_hot_shapes(self, shapes):
        shapes = torch.tensor(shapes, dtype=torch.long)
        shapes = shapes - 1  # Indexing starts at 1...
        return F.one_hot(shapes, num_classes=self.config.num_shapes)

    def _sin_cos_angles(self, angles):
        """Transforms an array-like to sinus and cosinus values"""
        return sin_cos_angles(angles)

    def _preprocess_scene(self, shape:int, angle:int):
        shape = int(shape)
        angle = int(angle)
        shape_features = self._one_hot_shapes(shape)
        angle_features = self._sin_cos_angles(angle)
        features = torch.cat([shape_features, angle_features], dim=-1)
        return features

    def _preprocess_target(self, target):
        if not self.language.has_word_with_id(target):
            # In the regularization block, words are typically not in language
            return None
        word = self.language.get_word_by_id(target)
        tokens = self.tokenizer.encode(word, pad=True)
        target_word = torch.LongTensor(tokens)
        return target_word

    def _preprocess_distractors(self, row):
        distractors = row[self._distractor_cols]
        distractors = distractors[distractors.notna()]

        list_of_distractor_features = []
        # list_of_distractor_target_word = []

        for distr_id in distractors:
            # Distractor features
            scene = self.language.get_scene_by_id(distr_id)
            distr_feats = self._preprocess_scene(*scene)
            list_of_distractor_features.append(distr_feats)

            # Distractor targets
            # distr_tgt = self._preprocess_target(distr_id)
            # list_of_distractor_target_word.append(distr_tgt)

        if not list_of_distractor_features:
            return None

        distractors_features = torch.stack(list_of_distractor_features)  # [num_found_distractors, num_features]
        # distractors_target_word = torch.stack(list_of_distractor_target_word)
        
        return distractors_features

    def _preprocess_row(self, row):
        # Shape and Angle features
        features = self._preprocess_scene(row.Shape, row.Angle)
        # Target word
        target_word = self._preprocess_target(row.Target)
        # Distractors
        distractors = self._preprocess_distractors(row)

        example = Data(features, target_word, distractors=distractors)

        return example


    def __call__(self, data: pd.DataFrame):
        list_of_examples = []

        for __idx, row in tqdm(data.iterrows(), desc='Preprocessing'):
            example = self._preprocess_row(row)
            list_of_examples.append(example)

        return list_of_examples




