""" Data from a Learning Experiment """
import itertools
import os.path as osp
import json
import datetime

from typing import List, Tuple, Dict
import pandas as pd

from language import load_input_language

# Line start that indicates start of data
START_OF_DATA_HEADER = "Round\tTask\tTrial"


def scenes(data:pd.DataFrame) -> List[Tuple]:
    """ Convert some data subset to list of shape,angle pairs """
    shapes = data.Shape.values
    angles = data.Angle.values
    scenes = list(zip(shapes, angles))
    return scenes


class LearningExp(object):
    """Class to deal with log files from learning experiments
    >>> e1 = LearningExp.load("path/to/learning-exp.txt")
    """

    EGP_ROUNDS = [1, 2, 3]
    MR_ROUNDS = [5]
    BLOCK_NAMES = [
        "Exposure",
        "Guessing",
        "Production",
        "MemorizationTest",
        "RegularizationTest",
    ]
    TRAIN_BLOCKS = BLOCK_NAMES[:3]

    def __init__(self, data: pd.DataFrame, lang: pd.DataFrame=None, info:Dict=None):
        """Initializes a learning exp

        :data: pandas.DataFrame
        :info: dict

        """
        self.data = data
        self.lang = lang
        self.info = info

    def __repr__(self) -> str:
        s = "LearningExp(\n"
        # s += f"\tdata=\n{self.data}\n"
        s += f"\tlang= {self.lang}\n"
        s += f"\tinfo= {self.info}\n"
        s += f")"
        return s

    def save(self, path):
        """ Save the learning experiment to disk: info and data"""
        self.info["Logfile written"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # First write metadata
        with open(path, 'w', encoding='utf8') as filehandle:
            print(json.dumps(self.info), file=filehandle)
        # Then write data
        self.data.to_csv(path, sep='\t', index=True, mode='a')

    def _get_data(self, task: str, round: int) -> pd.DataFrame:
        assert task in LearningExp.BLOCK_NAMES
        data = self.data.loc[(round, task)].reset_index()
        return data

    def _get_multi_round_data(self, task: str, rounds: [int]) -> pd.DataFrame:
        assert task in LearningExp.BLOCK_NAMES
        data_list = [self.data.loc[(r, task)] for r in rounds]
        data = pd.concat(data_list).reset_index()
        return data

    def get_exposure_data(self, round=None):
        """ Get all exposure data if `round=None`, or exposure data from a specific round """
        if round is None:
            return self._get_multi_round_data("Exposure", self.EGP_ROUNDS)

        assert round in self.EGP_ROUNDS
        return self._get_data("Exposure", round=round)

    def get_guessing_data(self, round=None):
        """ Get all guessing data if `round=None`, or guessing data from a specific round """
        if round is None:
            return self._get_multi_round_data("Guessing", self.EGP_ROUNDS)

        assert round in self.EGP_ROUNDS
        return self._get_data("Guessing", round=round)

    def get_production_data(self, round=None):
        """ Get all production data if `round=None`, or production data from a specific round """
        if round is None:
            return self._get_multi_round_data("Production", self.EGP_ROUNDS)

        assert round in self.EGP_ROUNDS
        return self._get_data("Production", round=round)

    def get_all_training_data(self):
        tasks = LearningExp.TRAIN_BLOCKS
        rounds = LearningExp.EGP_ROUNDS
        data_list = [self.data.loc[(r, t)] for r, t in itertools.product(rounds, tasks)]
        data = pd.concat(data_list).reset_index()
        return data

    def get_memorization_test_data(self, round:int=5):
        # round 5 is the sane default for human participant data
        return self._get_data("MemorizationTest", round=round)

    def get_regularization_test_data(self, round:int=5):
        # round 5 is the sane default for human participant data
        return self._get_data("RegularizationTest", round=round)

    def get_memorization_scenes(self):
        """ Return scenes from memorization data"""
        data = self.get_memorization_test_data()
        return scenes(data)

    def get_regularization_scenes(self):
        """ Return scenes from regularization data"""
        data = self.get_regularization_test_data()
        return scenes(data)

    @staticmethod
    def load(path, with_input_language=True):
        """Load a participant log file 

        :path: Path to language learnability experiment logfile
        :returns: LearningExp

        """
        info = {"filename": osp.basename(path)}
        with open(path, "r") as file:
            for linenumber, line in enumerate(file):
                line = line.strip()
                if line.startswith(START_OF_DATA_HEADER):
                    break

                if line.startswith('{'):  # Json indicator
                    new_info = json.loads(line)
                    info = {**info, **new_info}
                elif "-" in line:
                    key, value = line.split("-")
                    info[key.strip()] = value.strip()
                elif ":" in line:  # elif to not try to parse ':' within time stamps
                    key, value = line.split(":")
                    info[key.strip()] = value.strip()

        language_id = info["Language"]
        if with_input_language:
            language = load_input_language(language_id)
        else:
            language = None

        # data = pd.read_csv(path, sep='\t', skiprows=6) # Data starts in line 7
        data = pd.read_csv(
            path, sep="\t", skiprows=linenumber, index_col=["Round", "Task", "Trial"]
        )  # Now generic.

        return LearningExp(data, language, info=info)

    @staticmethod
    def empty_like(other):
        columns = list(other.data.reset_index().columns)
        # print("Columns", columns)
        # index = pd.MultiIndex.from_tuples([], names=other.data.index.names)
        data = pd.DataFrame(data=None, columns=columns)
        data.set_index(['Round', 'Task', 'Trial'], drop=True, inplace=True)
        info = {"Language": other.info["Language"], "_orig_info_": other.info}
        return LearningExp(data, lang=None, info=info)

    def append_results(
        self,
        round_number: int,
        task: str, 
        orig_data: pd.DataFrame,
        generated_messages: [str],
        correct_messages=None,
        producer=-1,
    ):
        """Add a bunch of results ignoring any indices"""
        # Current data has [Round, Task, Trial] index, reset first
        old_data = self.data.reset_index()

        # Orig data already comes with default range index
        new_data = orig_data.copy(deep=True)

        # Indexy-columns
        new_data["Round"] = round_number
        new_data["Task"] = task
        new_data["Trial"] = list(range(1, len(new_data) + 1))

        # Store human input in separate column
        new_data["OrigInput"] = new_data["Input"]
        new_data["Input"] = generated_messages
        new_data["Correct"] = correct_messages
        new_data["Producer"] = producer

        data = pd.concat([old_data, new_data], ignore_index=True)

        # Restore indices for compat with other methods
        data.set_index(['Round', 'Task', 'Trial'], drop=True, inplace=True)

        self.data = data
