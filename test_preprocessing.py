from preprocessing import Preprocessor

from tokenization import Tokenizer
from config import Raviv1
from language import load_input_language
from learning_experiment import LearningExp

def test_preprocessing():
    lang = load_input_language('S5')
    prep = Preprocessor(Raviv1, lang)

    lexp = LearningExp.load('data/LearningExp_190501_S5_001_log.txt')
    data = lexp.get_exposure_data()

    training_data = prep(data)

    assert len(training_data) == len(data)


