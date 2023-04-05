""" Test suite for the learning_experiment.py file """
from learning_experiment import LearningExp


def test_LearningExp():
    e1 = LearningExp.load("data/LearningExp_190501_S5_001_log.txt")

    N = len(e1.data)

    all_exposure_data = e1.get_exposure_data()

    all_guessing_data = e1.get_guessing_data()

    all_production_data = e1.get_guessing_data()

    memorization_test_data = e1.get_memorization_test_data()

    regularization_test_data = e1.get_regularization_test_data()

    assert (
        sum(
            len(d)
            for d in [
                all_exposure_data,
                all_guessing_data,
                all_production_data,
                memorization_test_data,
                regularization_test_data,
            ]
        )
        == N
    )


def test_LearningExp_empty_like():
    e1 = LearningExp.load("data/LearningExp_190501_S5_001_log.txt")

    e2 = LearningExp.empty_like(e1)

    assert (e2.data.columns == e1.data.columns).all()
    assert e2.data.index.names == e1.data.index.names
    assert e2.info["_orig_info_"] == e1.info
    assert len(e2.data) == 0


def test_LearningExp_save_load():
    e1 = LearningExp.load("data/LearningExp_190501_S5_001_log.txt")
    path = "/tmp/TMP-learningexp-saveload-test.txt"
    e1.save(path)
    e1_loaded = LearningExp.load(path)

    assert e1_loaded.info == e1.info
    assert len(e1_loaded.data) == len(e1.data)
    for col in e1.data.columns:
        print(col)
        assert (e1_loaded.data[col].fillna("XXX") == e1.data[col].fillna("XXX")).all()


def test_LearningExp_append_results():
    e1 = LearningExp.load("data/LearningExp_190501_S5_001_log.txt")

    log = LearningExp.empty_like(e1)

    mem_data = e1.get_memorization_test_data()
    reg_data = e1.get_regularization_test_data()

    mem_dummy_messages = ["wuseldusel"] * len(mem_data)
    mem_dummy_correct = [0] * len(mem_data)

    reg_dummy_messages = ["hupiflup"] * len(reg_data)
    reg_dummy_correct = None

    log.append_results(
        100,
        "MemorizationTest",
        mem_data,
        mem_dummy_messages,
        correct_messages=mem_dummy_correct,
    )
    log.append_results(
        100,
        "RegularizationTest",
        reg_data,
        reg_dummy_messages,
        correct_messages=reg_dummy_correct,
    )

    log.append_results(
        101,
        "MemorizationTest",
        mem_data,
        mem_dummy_messages,
        correct_messages=mem_dummy_correct,
    )
    log.append_results(
        101,
        "RegularizationTest",
        reg_data,
        reg_dummy_messages,
        correct_messages=reg_dummy_correct,
    )

    assert len(log.data) == (2 * len(mem_data) + 2 * len(reg_data))
