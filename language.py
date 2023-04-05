""" Module for a language """
import pandas as pd

INPUT_LANGUAGES_PATH = "data/input_languages.csv"
INPUT_LANGUAGES = None


def load_input_languages(path) -> pd.DataFrame:
    languages = pd.read_csv(path, sep=",", index_col=["InputCondition", "Item.ID"])
    return languages


def load_input_language(input_condition):
    # simple caching
    global INPUT_LANGUAGES
    if INPUT_LANGUAGES is None:
        #        print("Loading input languages...")
        INPUT_LANGUAGES = load_input_languages(INPUT_LANGUAGES_PATH)

    data = INPUT_LANGUAGES.loc[input_condition]

    #    print("input_condition:", input_condition)

    return Language(input_condition, data)


class Language(object):
    INPUT_LANG_PATH = "data/input_languages.csv"

    """Docstring for Language. """

    def __init__(self, name, data):
        """TODO: to be defined.

        :data: pd.DataFrame indexed by Item.ID

        """
        self.name = name
        self.data = data

    def get_scene_by_id(self, item_id: int) -> [int, int]:
        # used to resolve distractors
        item = self.data.loc[item_id]
        return int(item["Shape"]), int(item["Angle"])

    def get_word_by_id(self, item_id: int) -> str:
        # used to resolve targets
        item = self.data.loc[item_id]
        return item["Word"]

    def get_unique_attribute(self, key):
        assert key in ["GroupSize", "StructureBin", "StructureScore"]
        uniq_values = self.data[key].unique()
        assert len(uniq_values) == 1
        value = uniq_values[0]
        return value

    def has_word_with_id(self, word_id):
        return word_id in self.data.index

    def __repr__(self):
        s = f"Language {self.name}:\n"
        s += repr(self.data)
        s += "\n"
        return s
