import torch

class Data(object):

    """Docstring for Data."""

    def __init__(
            self, features: torch.FloatTensor, target_word: torch.LongTensor, distractors: [torch.FloatTensor] = None
    ):
        """A data object (single or batch)

        :features: preprocessed shape angle features
        :target_word: preprocessed target word
        :distractors: preprocessed distractors

        """
        assert features.ndim
        self.features = features
        self.target_word = target_word
        self.distractors = distractors

    def __repr__(self):
        s = 'Data(\n'
        s += f'\tfeatures={self.features}\n'
        s += f'\ttarget_word={self.target_word}\n'
        s += f'\tdistractors={self.distractors}\n'
        if self.distractors is not None:
            s += f'\tdistractors.shape={self.distractors.shape}\n'
        s += ')'
        return s

    def cuda(self):
        self.features = self.features.cuda()
        if self.target_word is not None:
            self.target_word = self.target_word.cuda()
        if self.distractors is not None:
            self.distractors = self.distractors.cuda()
        return self

    def __len__(self):
        return self.features.size(0)


def collate(examples):
    features = torch.stack([ex.features for ex in examples])
    target_word = torch.stack([ex.target_word for ex in examples])
    return Data(features, target_word, distractors=None)

def collate_with_distractors(examples):
    features = torch.stack([ex.features for ex in examples])
    target_word = torch.stack([ex.target_word for ex in examples])
    distractors = torch.stack([ex.distractors for ex in examples])
    return Data(features, target_word, distractors=distractors)

def collate_only_features(examples):
    features = torch.stack([ex.features for ex in examples])
    return Data(features, target_word=None, distractors=None)

class DataLoader(torch.utils.data.DataLoader):

    """DataLoader subclass with our collate function as default"""

    def __init__(self, *args, **kwargs):
        """Init dataloader

        :*args: passed to dataloader
        :**kwargs: passed to dataloader

        """
        collate_fn = kwargs.pop('collate_fn', collate)
        torch.utils.data.DataLoader.__init__(self, *args, **kwargs, collate_fn=collate_fn)

