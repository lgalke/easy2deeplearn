from tokenization import Tokenizer
from config import Raviv1
DUMMY_WORDS = ['ha-ia', 'swaa', 'hiu', 'gesh', 'sket', 'sktuu']

def test_Tokenizer_single():
    tokenizer = Tokenizer(Raviv1)
    for word in DUMMY_WORDS:
        encoded_word = tokenizer.encode(word)
        decoded_word = tokenizer.decode(encoded_word)
        assert decoded_word == word

def test_Tokenizer_batch():
    tokenizer = Tokenizer(Raviv1)
    encoded_batch = tokenizer.encode_batch(DUMMY_WORDS)
    decoded_batch = tokenizer.decode_batch(encoded_batch)

    for inp, out in zip(DUMMY_WORDS, decoded_batch):
        assert out == inp
