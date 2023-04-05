import argparse

import pandas as pd

from measures import form2meaning_ratio


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_languages_path")
    args = parser.parse_args()

    # Ambiguity measures: 1 - ( #num_unique_messages / #num_meanings)

    data = pd.read_csv(args.input_languages_path)

    for groupname, group in data.groupby("InputCondition"):
        # num_meanings = len(group.drop_duplicates(subset=["Shape", "Angle"]))
        # num_unique_messages = len(group.drop_duplicates(subset=["Word"]))
        # ratio = float(num_unique_messages) / num_meanings
        ratio = form2meaning_ratio(group[["Shape", "Angle"]].values, group.Word.values)
        ambiguity_score = (1 - ratio) * 100
        print(f"form2meaning-ratio({groupname}): {ratio:.4f}")
        print(f"ambiguity({groupname}): {ambiguity_score:.2f}%")


if __name__ == "__main__":
    main()
