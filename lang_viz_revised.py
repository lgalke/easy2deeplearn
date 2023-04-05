import argparse
import os
import os.path as osp
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# def tabularize_languages(mem_data, reg_data, human_data, epoch=100, participant_id=None):
#     pass
#
#
# def visualize_languages(mem_data, reg_data, epoch=100, participant_id=None):
#     mem_data_subset = mem_data[(mem_data.Round == epoch) & mem_data.]
#     reg_data_subset = reg_data[reg_data.Round == epoch]
#
#     plt.figure(1)
#
#     plt.arrow(0,0, 10, 10)
#     pass


def prepare_data(df, epoch=100):
    subset = df[df.Round == 100]
    unused_columns = ["Distr%d" % i for i in range(1, 8)] + [
        "SelectedItem",
        "Trial",
        "Target",
        "Task",
        "Round",
        "Correct",
    ]
    return subset.drop(unused_columns, axis=1)


def select_conditions(data, condition="Producer", criterion="ProdSim_Humans"):
    """Selects a subset of rows of `df` corresponding to the round number and the criterion
    Returns a dataframe like this:
          ProdSim_Humans  Producer
    0.00        0.247826      4082
    0.25        0.700932      1093
    0.50        0.883282      3090
    0.75        0.956522      1020
    1.00        1.000000      1008
    """
    condition_dtype = data[condition].dtype

    mean_by_condition = data.groupby(condition)[criterion].mean()

    print(mean_by_condition)

    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    quantiles = mean_by_condition.quantile(qs, interpolation="nearest")
    print("Quantiles:\n", quantiles)

    selected_conditions = pd.Series(dtype=condition_dtype).reindex_like(quantiles)
    for q, qval in quantiles.items():
        conditions = mean_by_condition[mean_by_condition == qval]
        # Deterministic
        # selected_conditions.loc[q] = np.unique(conditions.index)[0]
        # Random
        selected_conditions.loc[q] = conditions.sample(1).index

    selected_conditions = selected_conditions.astype(condition_dtype)

    # Assemble conditions and values in dataframe
    df = pd.DataFrame(
        {f"{criterion}-mean-by-{condition}": quantiles, condition: selected_conditions},
        index=quantiles.index,
    )

    return df


def merge_with_data(conditions_df, data, condition="Producer"):
    df = pd.merge(conditions_df, data, on=condition, how="inner")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mem_data_path", help="Path to mem_data.csv")
    parser.add_argument("reg_data_path", help="Path to reg_data.csv")
    parser.add_argument(
        "--shape_images_dir", default="./shape-images", help="Path to shape_images dir"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="At which epoch to show results"
    )
    parser.add_argument("--output_dir", default=".", help="Where to write output")
    parser.add_argument(
        "--condition", default="Producer", choices=["Producer", "InputCondition"]
    )
    parser.add_argument(
        "--criterion",
        default="ProdSim_Humans",
        help="What criterions to use for quantiles (default: ProdSim_Humans)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    criterion = args.criterion

    print(f"Output will be written to `{output_dir}`")

    # human_data = pd.read_csv(args.human_data_path)
    mem_data = pd.read_csv(args.mem_data_path)
    reg_data = pd.read_csv(args.reg_data_path)

    mem_data = prepare_data(mem_data)
    reg_data = prepare_data(reg_data)

    # print(mem_data.columns)
    print(reg_data.columns)

    grouped_by_input_condition = reg_data.groupby("InputCondition")
    mean_prodsim = grouped_by_input_condition["ProdSim_Humans"].mean().sort_values()
    print(mean_prodsim)
    mean_structscore = grouped_by_input_condition["StructureScore"].mean().sort_values()
    print(mean_structscore)

    s5 = reg_data[reg_data["InputCondition"] == "S5"]

    print("Selecting S5")

    os.makedirs(output_dir, exist_ok=True)

    # Select and gather mem results
    print(s5)
    s5.to_csv(osp.join(output_dir, f"s5.csv"))


if __name__ == "__main__":
    main()
