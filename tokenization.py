# Special token ID for End-of-Sequence
EOS_TOKEN_ID = 0
# Offset is the index at which ordinary vocabulary tokens start
VOCAB_OFFSET = 1


class Tokenizer(object):
    """Docstring for Vocab."""

    def __init__(self, config):
        """Mapping from symbols to numbers

        :vocab: dict-like from str to int
        :index2symbol: array-like from int to str

        """
        self.vocab = None
        self.index2token = None

        self.max_length = config.max_length + 1

        self.build_vocab(config.alphabet)

    def encode(self, tokens: str, pad=False) -> [int]:
        if len(tokens) > self.max_length:
            print("Warning: exceeded max length. Trimming...")
            tokens = tokens[: self.max_length]

        encoded_tokens = [self.vocab[t] for t in tokens] + [EOS_TOKEN_ID]

        if pad:
            # We use the EOS token to pad
            encoded_tokens += [EOS_TOKEN_ID] * (self.max_length - len(encoded_tokens))

        return encoded_tokens

    def decode(self, indices: [int]) -> str:
        indices = [int(i) for i in indices]
        tokens = []
        for i in indices:
            if i == EOS_TOKEN_ID:
                break
            tokens.append(self.index2token[i])

        return "".join(tokens)

    def encode_batch(self, tokens_batch:[str], pad:bool=True):
        return [self.encode(tokens, pad=pad) for tokens in tokens_batch]

    def decode_batch(self, indices_batch:[[int]]):
        return [self.decode(ind) for ind in indices_batch]
        # iterate via index to be safe to torch tensors
        # return [self.decode(indices_batch[i]) for i in range(len(indices_batch))]

    def build_vocab(self, alphabet):
        """Builds the mapping from symbol to int
        :alphabet: iterable of symbols
        :returns: Vocab

        """
        print("Building vocabulary")
        # Start with special symbols

        tokens = sorted(set(alphabet))

        vocab = dict()
        # Then insert
        for token in tokens:
            vocab[token] = VOCAB_OFFSET + len(vocab)

        self.vocab = vocab
        self.index2token = {tok_id: tok for (tok, tok_id) in vocab.items()}
