import argparse
import glob
import os
from typing import Tuple, Union

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300


import matplotlib.pyplot as plt

plt.rcParams["axes.facecolor"] = "white"

import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.factorplots import interaction_plot
from joblib import Memory
from patsy import dmatrices, dmatrix
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold


# from sklearn.preprocessing import scale
from tqdm import tqdm

from learning_experiment import LearningExp, scenes
from measures import (
    convergence_score,
    generalization_score,
    production_similarity,
    mean_production_similarity,
)

# from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


# GLOBALS
COLORMAP = plt.get_cmap(
    "copper"
).reversed()  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
ERRORBAR = "se"

LINEWIDTH = 2.5
STARSIZE = 250

CACHEDIR = "./__stats_cache__"
MEMORY = Memory(CACHEDIR)


INDEX_COLUMNS = ["Producer", "Round", "Trial"]


@MEMORY.cache
def load_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = glob.glob(os.path.join(path, "*_log.txt"))

    dfs = []
    for file in tqdm(files, desc="Loading files"):
        lexp = LearningExp.load(file, with_input_language=False)
        # Reset index because it's not unique over multiple files
        lexp_data = lexp.data.reset_index()
        # Store input language in each row
        lexp_data["InputCondition"] = lexp.info["Language"]
        dfs.append(lexp_data)

    data = pd.concat(dfs, ignore_index=True)

    print(f"N = {len(data)}")

    # Make sure to handle produced empty sequences correctly
    data["Input"].fillna("", inplace=True)

    # Split into memorization and regularization data
    # and set useful indices for later
    mem_data = data[data.Task == "MemorizationTest"].set_index(INDEX_COLUMNS)
    reg_data = data[data.Task == "RegularizationTest"].set_index(INDEX_COLUMNS)

    print("Sorting index")
    mem_data.sort_index(inplace=True)
    reg_data.sort_index(inplace=True)

    print(f"N_mem = {len(mem_data)}")
    print(f"N_reg = {len(reg_data)}")

    return mem_data, reg_data


def add_structure_score(data: pd.DataFrame, input_languages_file: str):
    input_languages_struct = pd.read_csv(
        input_languages_file, usecols=["InputCondition", "StructureScore"]
    )
    input_languages_struct.drop_duplicates(inplace=True)
    input_languages_struct.set_index("InputCondition", inplace=True)
    data = data.join(input_languages_struct, on=["InputCondition"])
    return data


def add_structure_bin(data: pd.DataFrame):
    assert "InputCondition" in data, "InputCondition not found"
    bins = data["InputCondition"].map(
        {
            "B1": 1,
            "S1": 1,
            "B2": 2,
            "S2": 2,
            "B3": 3,
            "S3": 3,
            "B4": 4,
            "S4": 4,
            "B5": 5,
            "S5": 5,
        }
    )
    return bins


def get_genscore_norm_values(
    data: pd.DataFrame, genscore_col="GenScore"
) -> Tuple[float, dict]:
    global_minimum = data[genscore_col].min()
    local_maxima = {}
    for key, group in data.groupby("InputCondition"):
        local_maxima[key] = group[genscore_col].max()
    return global_minimum, local_maxima


def normalize_genscore(
    reg_data: pd.DataFrame,
    genscore_col="GenScore",
    norm_values: Tuple[float, dict] = None,
) -> pd.Series:
    """
    Formula: x’ = (x-min(x))/(max(x)-min(x)), where min(x) in the lowest value for x
    achieved by a participant across all conditions (− 0.069), and max(x) is
    the highest value for x achieved by a participant in a specific condition
    (i.e., max(x) varied for different input languages, with each input lan­
    guage having a different maximal value).
    """
    # Temporary dataframe for the computations
    new_series = pd.Series(dtype=float).reindex_like(reg_data)

    # This is actually not done.
    # df['_GenScore_01'] = (df['GenScore'] + 1) / 2
    if norm_values is None:
        # Compute if not given
        norm_values = get_genscore_norm_values(reg_data, genscore_col=genscore_col)

    print("Using norm values to normalize genscore:", norm_values)

    global_min, local_maxima = norm_values

    # ..and local max, is used to scale to [0,1]
    for key, group in reg_data.groupby("InputCondition"):
        values = group[genscore_col]
        local_max = local_maxima[key]
        new_series[group.index] = (values - global_min) / (local_max - global_min)

    return new_series


@MEMORY.cache
def calc_generalization_score(
    mem_data: pd.DataFrame, reg_data: pd.DataFrame, return_pval: bool = False
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    seeds = pd.unique(reg_data.index.get_level_values("Producer"))
    rounds = pd.unique(reg_data.index.get_level_values("Round"))

    genscore_series = pd.Series(dtype=float).reindex_like(reg_data)
    pval_series = pd.Series(dtype=float).reindex_like(reg_data)

    # Assumes index levels ['Producer', 'Round', 'Trial']
    # Requirement: Langs are uniquely identified by Producer (Seed)!
    for i in tqdm(seeds, desc="Calculate generalization score"):
        for j in rounds:
            mem_round_data = mem_data.loc[(i, j)]
            familiar_scenes = scenes(mem_round_data)
            familiar_labels = mem_round_data["Input"]

            reg_round_data = reg_data.loc[(i, j)]
            new_scenes = scenes(reg_round_data)
            new_labels = reg_round_data["Input"]

            genscore, pval = generalization_score(
                familiar_scenes,
                familiar_labels,
                new_scenes,
                new_labels,
                scene_metric="semantic_difference",
                label_metric="normalized_editdistance",
                rescale=False,  # Do not rescale to [0,2] yet
            )

            genscore_series[(i, j)] = genscore
            pval_series[(i, j)] = pval

    if return_pval:
        return genscore_series, pval_series

    return genscore_series


def calc_generalization_score_for_humans(
    mem_data: pd.DataFrame,
    reg_data: pd.DataFrame,
    return_pval: bool = False,
    at_round: int = 100,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    # get seeds from data because we have multiple humans
    seeds = pd.unique(reg_data["Producer"])
    # Select subset of rounds b/c only one for humans
    mem_subset = mem_data[mem_data.Round == at_round]
    reg_subset = reg_data[reg_data.Round == at_round]

    results = []

    # Requirement: Langs are uniquely identified by Producer (Seed)!
    for i in tqdm(seeds, desc="Calculate generalization score"):
        # Prepare mem data
        mem_data_of_seed = mem_subset[mem_subset.Producer == i]
        familiar_scenes = scenes(mem_data_of_seed)
        familiar_labels = mem_data_of_seed["OrigInput"]

        # Prepare reg data
        reg_data_of_seed = reg_subset[reg_subset.Producer == i]
        new_scenes = scenes(reg_data_of_seed)
        new_labels = reg_data_of_seed["OrigInput"]

        genscore, pval = generalization_score(
            familiar_scenes,
            familiar_labels,
            new_scenes,
            new_labels,
            scene_metric="semantic_difference",
            label_metric="normalized_editdistance",
            rescale=False,  # Do not rescale to [0,2] yet
        )
        results.append(
            {
                "InputCondition": reg_data_of_seed[
                    "InputCondition"
                ].max(),  # same anyways
                "Producer": i,
                "StructureScore": reg_data_of_seed[
                    "StructureScore"
                ].max(),  # same anyways
                "GenScore_of_Humans": genscore,
            }
        )

    df = pd.DataFrame(results)
    return df  # Does contain some redundancy


@MEMORY.cache
def calc_convergence_score(reg_data: pd.DataFrame, word_column="Input"):
    """
    This continuous measure reflects the degree of similarity between
    the labels produced during the generalization test by different partici­
    pants who learned the same input language. For each of the new scenes
    in the ten input languages, we calculated the normalized Levenshtein
    distances between all pairs of labels produced by different participants
    for the same new scenes. The average distance between all pairs of labels
    was subtracted from 1 to represent string similarity, i.e., how much the
    labels of different participants resembled each other. A high conver­
    gence score indicates that participants who learned the same language
    also produced similar labels for the unfamiliar scenes during the
    generalization test. A low convergence score indicates that participants
    who learned the same language produced different labels for unfamiliar
    scenes during the generalization test.
    """
    convscore_series = pd.Series(dtype=float).reindex_like(reg_data)
    for name, group in tqdm(
        reg_data.groupby(["InputCondition", "Round", "Target"]),
        desc="Calculating convergence scores",
    ):
        convscore = 1 - convergence_score(
            group[word_column], metric="normalized_editdistance"
        )

        convscore_series[group.index] = convscore

    # OLD variant (errornous)
    # Iterate over items
    # Items are a function of Trial in convergence
    # rounds = pd.unique(reg_data.index.get_level_values("Round"))
    # trials = pd.unique(reg_data.index.get_level_values("Trial"))
    # Use Target to make sure despite Trial would be faster to access
    # items = pd.unique(reg_data["Target"])
    # for i in tqdm(rounds, desc="Calculate convergence score"):
    #     for j in trials:
    #         subset = reg_data.loc[:, i, j]

    #         messages = subset["Input"].values

    #         # Exactly as in the orig R script:
    #         # 1. calc conv score with normalized edit distance
    #         # 2. then calculate 1 - conv score
    #         # ( we could also do prodsim right away, but better keep same )

    #         convscore = convergence_score(messages, metric="normalized_editdistance")

    #         convscore_series.loc[:, i, j] = 1 - convscore

    return convscore_series


def find_intersection(reference_value: float, data: pd.DataFrame, column: str):
    assert "Round" in data, "No Round column found"
    all_rounds = np.unique(data.Round)
    for i in all_rounds:
        subset = data[data.Round == i]
        if subset[column].mean() >= reference_value:
            return i

    return None


def lineplot_with_stars(
    ax,
    data: pd.DataFrame,
    y: str,
    with_stars=True,
    ref_data: pd.DataFrame = None,
    ref_y: str = None,
    x="Round",
    hue="StructureBin",
    hue_norm=(1, 5),
    legend="auto",
) -> None:
    if ref_data is None:
        ref_data = data
    if ref_y is None:
        ref_y = y

    sns.lineplot(
        x=x,
        y=y,
        hue=hue,
        hue_norm=hue_norm,
        errorbar=ERRORBAR,
        palette=COLORMAP,
        linewidth=LINEWIDTH,
        legend=legend,
        data=data,
        ax=ax,
    )

    if not with_stars:
        return

    for groupname, group in ref_data.groupby(hue):
        star_y = group[ref_y].mean()
        corresponding_subset = data[data[hue] == groupname]
        intersection_point = find_intersection(star_y, corresponding_subset, y)

        star_x = (
            intersection_point if intersection_point is not None else max(data.Round)
        )

        ax.scatter(
            star_x,
            star_y,
            marker="*",
            s=STARSIZE,
            c=groupname,
            cmap=COLORMAP,
            vmin=hue_norm[0],
            vmax=hue_norm[1],
        )
        print(f"{y}: NNs exceed humans group value {star_y} at round {star_x}.")


def postprocess_legend(src_ax, dst_ax=None, title="Structure", **kwargs):
    dst_ax = src_ax if dst_ax is None else dst_ax

    h, l = src_ax.get_legend_handles_labels()
    desc = ["Low", "Mid-low", "Mid", "Mid-high", "High"]
    dst_ax.legend(
        reversed(h),
        reversed([desc[int(i) - 1] for i in l]),
        title=title,
        **kwargs,
    )


def make_memorization_plots(mem_data: pd.DataFrame, outdir: str = ".", cut=None):
    figsize = (10, 6)
    colormap = COLORMAP

    if cut is not None:
        mem_data = mem_data[mem_data.Round <= cut]

    if "StructureBin" not in mem_data:
        mem_data["StructureBin"] = add_structure_bin(mem_data)
    # with plt.style.context("seaborn-paper"):

    ###### ACCURACY TO GROUNDTRUTH ######
    fig, ax = plt.subplots(1, figsize=figsize)
    mem_data.loc[:, "Correct_Humans_GroundTruth"] = mem_data.apply(
        lambda row: row["OrigInput"] == row["Word"], axis=1
    )
    lineplot_with_stars(ax, mem_data, "Correct", ref_y="Correct_Humans_GroundTruth")

    postprocess_legend(ax)
    sns.move_legend(ax, "upper left")
    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, f"mem-accuracy_lineplot.png"))

    ###### PRODSIM TO GROUNDTRUTH ######
    fig, ax = plt.subplots(1, figsize=figsize)
    mem_data.loc[:, "ProdSim_Humans_GroundTruth"] = mem_data.apply(
        lambda row: production_similarity(row["OrigInput"], row["Word"]), axis=1
    )
    lineplot_with_stars(
        ax, mem_data, "ProdSim_GroundTruth", ref_y="ProdSim_Humans_GroundTruth"
    )
    postprocess_legend(ax)
    sns.move_legend(ax, "lower right")
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(outdir, f"mem-prodsim_lineplot.png"))

    ###### PRODSIM TO HUMANS ######
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(1, figsize=figsize)
    lineplot_with_stars(ax, mem_data, "ProdSim_Humans", with_stars=False)
    postprocess_legend(ax)
    sns.move_legend(ax, "lower right")
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(outdir, f"mem-prodsim-humans_lineplot.png"))


def make_memorization_panel(mem_data: pd.DataFrame, outdir: str = ".", cut=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
    if cut is not None:
        mem_data = mem_data[mem_data.Round <= cut]
    if "StructureBin" not in mem_data:
        mem_data["StructureBin"] = add_structure_bin(mem_data)

    # Plot prodsim groundtruth in ax1
    mem_data.loc[:, "ProdSim_Humans_GroundTruth"] = mem_data.apply(
        lambda row: production_similarity(row["OrigInput"], row["Word"]), axis=1
    )
    lineplot_with_stars(
        ax1,
        mem_data,
        "ProdSim_GroundTruth",
        ref_y="ProdSim_Humans_GroundTruth",
        legend=False,
    )
    ax1.set_title("Production Similarity to Ground Truth")
    ax1.set_ylabel("Prod. Sim.")

    # Prodsim to humans in ax2
    lineplot_with_stars(ax2, mem_data, "ProdSim_Humans", with_stars=False)
    ax2.set_title("Production Similarity to Human Learners")

    postprocess_legend(ax2)
    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, f"mem-panel.png"))
    print("Mem Panel done")


def make_generalization_plots(
    reg_data: pd.DataFrame,
    human_raw_genscores: pd.DataFrame = None,
    outdir: str = ".",
    cut=None,
):
    figsize = (10, 6)
    if "StructureBin" not in reg_data:
        reg_data["StructureBin"] = add_structure_bin(reg_data)
    if "GenScore_normalized" not in reg_data:
        reg_data["GenScore_normalized"] = normalize_genscore(reg_data)

    ## CUT DATA FOR PLOTTING
    if cut is not None:
        reg_data = reg_data[reg_data.Round <= cut]

    ## GEN SCORE NORMALIZED (humans not comparable, due to norm)
    if human_raw_genscores is not None:
        human_raw_genscores["StructureBin"] = add_structure_bin(human_raw_genscores)
        human_raw_genscores["GenScore_of_Humans_normalized"] = normalize_genscore(
            human_raw_genscores, genscore_col="GenScore_of_Humans"
        )

    fig, ax = plt.subplots(1, figsize=figsize)
    lineplot_with_stars(
        ax,
        reg_data,
        "GenScore_normalized",
        with_stars=(human_raw_genscores is not None),
        ref_data=human_raw_genscores,
        ref_y="GenScore_of_Humans_normalized",
    )
    postprocess_legend(ax)
    sns.move_legend(ax, "lower right")
    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, f"reg-genscore_lineplot.png"))

    ## GEN SCORE NON-NORMALIZED
    fig, ax = plt.subplots(1, figsize=figsize)
    lineplot_with_stars(
        ax,
        reg_data,
        "GenScore",
        with_stars=(human_raw_genscores is not None),
        ref_data=human_raw_genscores,
        ref_y="GenScore_of_Humans",
    )

    postprocess_legend(ax)
    sns.move_legend(ax, "lower right")
    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, f"reg-genscore-pre-norm_lineplot.png"))

    ### GEN SCORE v3:  NORMALIZED BY HUMAN MIN MAX ###
    if human_raw_genscores is not None:
        print("Plotting Gen score normalized by humans")
        human_norm_values = get_genscore_norm_values(
            human_raw_genscores, genscore_col="GenScore_of_Humans"
        )
        human_raw_genscores["GenScore_of_Humans_normalized"] = normalize_genscore(
            human_raw_genscores,
            genscore_col="GenScore_of_Humans",
            norm_values=human_norm_values,
        )
        human_raw_genscores["StructureBin"] = add_structure_bin(human_raw_genscores)

        reg_data["GenScore_norm_by_humans"] = normalize_genscore(
            reg_data, norm_values=human_norm_values
        )
        fig, ax = plt.subplots(1, figsize=figsize)
        lineplot_with_stars(
            ax,
            reg_data,
            "GenScore_norm_by_humans",
            ref_data=human_raw_genscores,
            ref_y="GenScore_of_Humans_normalized",
        )
        postprocess_legend(ax)
        sns.move_legend(ax, "lower right")
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(outdir, f"reg-genscore-norm-by-humans_lineplot.png"))

    ## CONV SCORE
    fig, ax = plt.subplots(1, figsize=figsize)
    lineplot_with_stars(ax, reg_data, "ConvScore", ref_y="ConvScore_of_Humans")
    postprocess_legend(ax)
    sns.move_legend(ax, "lower right")
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(outdir, f"reg-convscore_lineplot.png"))

    ## PRODSIM HUMANS
    fig, ax = plt.subplots(1, figsize=figsize)
    lineplot_with_stars(ax, reg_data, "ProdSim_Humans", with_stars=False)
    postprocess_legend(ax)
    sns.move_legend(ax, "lower right")
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(outdir, f"reg-prodsim-humans_lineplot.png"))


def make_generalization_panel(
    reg_data: pd.DataFrame,
    human_raw_genscores: pd.DataFrame = None,
    outdir: str = ".",
    cut=None,
):
    if "StructureBin" not in reg_data:
        reg_data["StructureBin"] = add_structure_bin(reg_data)
    if "StructureBin" not in human_raw_genscores:
        human_raw_genscores["StructureBin"] = add_structure_bin(human_raw_genscores)

    ## CUT DATA FOR PLOTTING
    if cut is not None:
        reg_data = reg_data[reg_data.Round <= cut]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=True, sharey=False, figsize=(10, 6)
    )

    ### Ax1: Generalization score
    lineplot_with_stars(
        ax1,
        reg_data,
        "GenScore",
        with_stars=(human_raw_genscores is not None),
        ref_data=human_raw_genscores,
        ref_y="GenScore_of_Humans",
        legend=False,
    )
    ax1.set_title("Generalization")
    ax1.set_ylabel("Gen. Score")

    ## Ax2: Prodsim to Humans
    lineplot_with_stars(ax2, reg_data, "ProdSim_Humans", with_stars=False, legend=False)
    ax2.set_title("Similarity to Human Learners")
    ax2.set_ylabel("Prod. Sim.")

    ## Ax 3: ConvScore
    lineplot_with_stars(
        ax3, reg_data, "ConvScore", ref_y="ConvScore_of_Humans", legend="auto"
    )
    ax3.set_title("Convergence")
    ax3.set_ylabel("Conv. Score")

    h, l = ax3.get_legend_handles_labels()
    print("Handles", h)
    print("Labels", l)
    desc = ["Low", "Mid-low", "Mid", "Mid-high", "High"]
    ax4.legend(
        reversed(h),
        reversed([desc[int(i) - 1] for i in l]),
        frameon=False,
        title="Structure of Input Language",
        loc="upper left",
    )
    ax4.axis("off")
    ax3.get_legend().remove()
    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, f"reg-panel.png"))
    print("Gen Panel done")


def make_big_panel(
    mem_data: pd.DataFrame,
    reg_data: pd.DataFrame,
    human_genscores: pd.DataFrame = None,
    outdir: str = ".",
    cut: int = None,
):
    if cut is not None:
        mem_data = mem_data[mem_data.Round <= cut].copy()
        reg_data = reg_data[reg_data.Round <= cut].copy()
    if "StructureBin" not in mem_data:
        mem_data["StructureBin"] = add_structure_bin(mem_data)
    if "StructureBin" not in reg_data:
        reg_data["StructureBin"] = add_structure_bin(reg_data)
    if "GenScore_normalized" not in reg_data:
        reg_data["GenScore_normalized"] = normalize_genscore(reg_data)
    if human_genscores is not None:
        human_genscores["StructureBin"] = add_structure_bin(human_genscores)
        human_genscores["GenScore_of_Humans_normalized"] = normalize_genscore(
            human_genscores, genscore_col="GenScore_of_Humans"
        )

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, sharex=True, figsize=(12, 10)
    )
    # Ax1: prodsim groundtruth
    mem_data["ProdSim_Humans_GroundTruth"] = mem_data.apply(
        lambda row: production_similarity(row["OrigInput"], row["Word"]), axis=1
    )
    lineplot_with_stars(
        ax1,
        mem_data,
        "ProdSim_GroundTruth",
        ref_y="ProdSim_Humans_GroundTruth",
        legend=False,
    )
    ax1.set_title("(A) Similarity to Input Language during Memorization")
    ax1.set_ylabel("Production Similarity")

    # ax2: Mem Prodsim to humans
    lineplot_with_stars(ax2, mem_data, "ProdSim_Humans", with_stars=False, legend=False)
    ax2.set_title("(B) Similarity to Humans during Memorization")
    ax2.set_ylabel("Production Similarity")

    ### ax3: Generalization score
    lineplot_with_stars(
        ax3,
        reg_data,
        "GenScore",
        with_stars=(human_genscores is not None),
        ref_data=human_genscores,
        ref_y="GenScore_of_Humans",
        legend=False,
    )
    ax3.set_title("(C) Generalization Systematicity")
    ax3.set_ylabel("Generalization Score")

    ## Ax4: Prodsim to Humans
    lineplot_with_stars(ax4, reg_data, "ProdSim_Humans", with_stars=False, legend=False)
    ax4.set_title("(D) Similarity to Humans during Generalization")
    ax4.set_ylabel("Production Similarity")

    ## Ax 5: ConvScore
    lineplot_with_stars(
        ax5, reg_data, "ConvScore", ref_y="ConvScore_of_Humans", legend="auto"
    )
    ax5.set_title("(E) Convergence between Networks during Generalization")
    ax5.set_ylabel("Convergence Score")

    postprocess_legend(
        ax5,
        dst_ax=ax6,
        frameon=False,
        loc="upper left",
        title="Structure of Input Language",
    )
    ax6.axis("off")
    ax5.get_legend().remove()

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(outdir, f"big-panel.png"))
    print("BIG Panel done")


def poly(x, p):
    """Equivalent to the R function poly(...)

    It already does the QR decomposition, which sklearn.preprocessing.PolynomialFeatures doesn't.

    See: https://stackoverflow.com/questions/41317127/python-equivalent-to-r-poly-function
    """
    x = np.asarray(x)
    X = np.transpose(np.vstack((x**k for k in range(p + 1))))
    return np.linalg.qr(X)[0][:, 1:]


def save_summary_and_plot(number, results, folder, extra_desc: str = ""):
    summary = results.summary()
    print(summary)
    with open(
        os.path.join(folder, f"model_{number}{extra_desc}_summary.txt"),
        mode="w",
        encoding="utf-8",
    ) as filehandle:
        print(summary, file=filehandle)

    with open(
        os.path.join(folder, f"model_{number}{extra_desc}_summary.tex"),
        mode="w",
        encoding="utf-8",
    ) as filehandle:
        print(summary.as_latex(), file=filehandle)

    with plt.style.context("seaborn-paper"):
        # Plot partregress grid
        fig = plt.figure(figsize=(8, 6))
        fig = sm.graphics.plot_partregress_grid(results, fig=fig)
        plt.tight_layout(pad=1.0)
        plt.savefig(
            os.path.join(folder, f"model_{number}{extra_desc}_partregress_grid.png")
        )

        # Plot ccpr grid
        fig = plt.figure(figsize=(8, 6))
        fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(folder, f"model_{number}{extra_desc}_ccpr_grid.png"))

        fig = plt.figure(figsize=(8, 6))


def run_model_1(data: pd.DataFrame, epoch=100, outdir="."):
    """Final Accuracy
    :data: Dataframe holding memorization data
    :epoch: The epoch at which the test should be conducted
    :returns: None
    :outdir: base directory to write outputs to

    Plot 1 Final Accuracy
    # model with 1 degree
    m1<- glmer(Raw_ACC ~
            c.Structure_Score +
            (1|Seed)+(1|Target_Item),
            data=memory_test, family="binomial")

    # model with 2 degree (selected)
    m1_poly <- glmer(Raw_ACC ~
                     poly(c.Structure_Score,2) +
                     (1|Seed)+(1|Target_Item),
                     data=memory_test, family="binomial")

    """
    print(f"Running model 1 on data from epoch {epoch}")

    # BinomialBayesMixedGLM needs vc_formulas ? random?
    # https://www.statsmodels.org/stable/generated/statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.html#statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM
    # Example from page would translate to (1 + Year | Village)

    subset = data[data.Round == epoch]

    random = {"a": "0 + C(Producer)", "b": "0 + C(Target)"}

    model = sm.BinomialBayesMixedGLM.from_formula(
        "Correct ~ scale(StructureScore)", random, subset
    )

    results = model.fit_vb()

    print(result.summary())
    save_summary_and_plot(1, results, outdir, extra_desc=f"_at{epoch}_")


def run_model_2(data: pd.DataFrame, epoch=100, outdir="."):
    """Final Production Similarity (with nested random intercepts)

    :data: Dataframe holding memorization data
    :epoch: The epoch at which the test should be conducted
    :returns: None
    :outdir: base directory to write outputs to

    """
    print(f"Running model 2 on data from epoch {epoch}")
    subset = data[data.Round == epoch]

    groups = nested_group_labels(subset["Producer"], subset["Target"])
    model = sm.MixedLM.from_formula(
        "ProdSim_GroundTruth ~ scale(StructureScore)", groups=groups, data=subset
    )
    results = model.fit()

    save_summary_and_plot(2, results, outdir, extra_desc=f"_at{epoch}")


def make_interaction_plot(
    data: pd.DataFrame,
    response_variable: str = "ProdSim_GroundTruth",
    outdir: str = ".",
    prefix="",
):
    """Plot the interaction with Round x to `response_variable` as y and Structure as different colors"""
    with plt.style.context("seaborn-paper"):
        plt.figure(figsize=(6, 8))
        fig, ax = plt.subplots(figsize=(6, 8))
        fig = interaction_plot(
            x=data["Round"],  # x
            trace=data["StructureScore"],  # trace
            response=data[response_variable],  # response
            # colors=["red", "blue"],
            # markers=["D", "^"],
            ms=10,
            ax=ax,
            plottype="both",
        )

        # NO RANDOM EFFECTS
        plt.tight_layout(pad=1.0)
        plt.savefig(
            os.path.join(outdir, f"{prefix}{response_variable}_interaction_plot.png")
        )


def run_model_3(mem_data: pd.DataFrame):
    """Learning trajectory"""
    print("Running model 3 -- Learning trajectory")
    print(mem_data.columns)

    raise NotImplementedError


def nested_group_labels(outer: pd.Series, inner: pd.Series):
    return outer.astype(str) + "_" + inner.astype(str)


def run_model_4_with_crossed_random_effects(data: pd.DataFrame, outdir="."):
    """Learning trajectory PROD SIM to ground truth
    Adapted from: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/regression/tests/test_lme.py#L284
    """
    print(
        "Running model 4 with more random effects -- Learning trajectory -- PROD SIM to Ground truth "
    )

    vcf = {"item": "0 + C(Target)", "seed": "0 + C(Producer)"}
    groups = np.ones(len(data))
    print(data.columns)
    model = sm.MixedLM.from_formula(
        "ProdSim_GroundTruth ~ scale(StructureScore) * scale(np.log(Round))",
        re_formula="0+C(Target)+C(Producer)",
        vc_formula=vcf,
        groups=groups,
        data=data,
    )
    # Killed (Out of memory)
    results = model.fit()

    save_summary_and_plot(41, results, outdir)

    with plt.style.context("seaborn-paper"):
        plt.figure(figsize=(8, 6))
        sns.relplot(
            x="Round",
            y="ProdSim_GroundTruth",
            size="StructureScore",
            hue="StructureBin",
            kind="line",
            data=data,
            errorbar=ERRORBAR,
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(outdir, "model_41_relplot.png"))


def run_model_4_with_nested_random_effects(data: pd.DataFrame, outdir="."):
    """Learning trajectory PROD SIM to ground truth"""
    print(
        "Running model 4 with nested random effects -- Learning trajectory -- PROD SIM to Ground truth "
    )

    # vcf = {"item": "0 + C(Target)", "seed": "0 + C(Producer)"}
    # groups = nested_group_labels(data['Producer'], data['Target'])
    # print(data.columns)
    # model = sm.MixedLM.from_formula("ProdSim_GroundTruth ~ scale(StructureScore) * scale(np.log(Round))",
    #             re_formula="0+C(Target)+C(Producer)",
    #             vc_formula=vcf,
    #             groups=groups,
    #             data=data
    #         )
    # results = model.fit()
    # -> killed

    groups = nested_group_labels(data["Producer"], data["Target"])
    model = sm.MixedLM.from_formula(
        "ProdSim_GroundTruth ~ scale(StructureScore) * scale(np.log(Round))",
        groups=groups,
        data=data,
    )
    results = model.fit()

    save_summary_and_plot(42, results, outdir)

    with plt.style.context("seaborn-paper"):
        plt.figure(figsize=(8, 6))
        sns.relplot(
            x="Round",
            y="ProdSim_GroundTruth",
            size="StructureScore",
            kind="line",
            data=data,
            errorbar=ERRORBAR,
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(outdir, "model_42_relplot.png"))


def run_model_4(data: pd.DataFrame, outdir=".", nested=False):
    """Learning trajectory PROD SIM to ground truth"""
    print("Running model 4 -- Learning trajectory -- PROD SIM to Ground truth ")

    endog, exog = dmatrices(
        "ProdSim_GroundTruth ~ scale(StructureScore) * scale(np.log(Round))",
        data=data,
        return_type="dataframe",
    )

    if nested:
        groups = nested_group_labels(data["Producer"], data["Target"])
    else:
        groups = data["Producer"]

    model = sm.MixedLM(endog, exog, groups)  # converged w/ log round
    results = model.fit()

    save_summary_and_plot(4, results, outdir)

    with plt.style.context("seaborn-paper"):
        plt.figure(figsize=(8, 6))
        sns.relplot(
            x="Round",
            y="ProdSim_GroundTruth",
            size="StructureScore",
            kind="line",
            data=data,
            errorbar=ERRORBAR,
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(outdir, "model_4_relplot.png"))


def run_model_6(data: pd.DataFrame, outdir=".", nested=False):
    """Learning trajectory GEN SCORE"""
    print("Running model 6 -- GenScore trajectory")

    print(data.columns)
    data["GenScore_normalized"] = normalize_genscore(data)

    print("Removing NA values")
    subset = data[data.GenScore.notna()]

    endog, exog = dmatrices(
        "GenScore_normalized ~ scale(StructureScore) * scale(np.log(Round))",  # this works
        data=subset,
        return_type="dataframe",
    )

    if nested:
        groups = nested_group_labels(subset["Producer"], subset["Target"])
    else:
        groups = subset["Producer"]

    # model = sm.OLS(endog, exog) # Converges nicely, R^2=0.52 or something
    model = sm.MixedLM(endog, exog, groups=groups)  # converges with linear round
    # model = sm.MixedLM(endog, exog, subset["Target"]) # converges with linear round
    results = model.fit()

    save_summary_and_plot(6, results, outdir)
    with plt.style.context("seaborn-paper"):
        plt.figure(figsize=(8, 6))
        sns.relplot(
            x="Round",
            y="GenScore_normalized",
            size="StructureScore",
            kind="line",
            data=data,
            errorbar=ERRORBAR,
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(outdir, f"model_6_relplot.png"))


def run_model_6b(data: pd.DataFrame, outdir=".", nested=False):
    """Learning trajectory GEN SCORE not normalized"""
    print("Running model 6 -- GenScore trajectory")
    subset = data[data.GenScore.notna()]

    endog, exog = dmatrices(
        "GenScore ~ scale(StructureScore) * scale(np.log(Round))",  # this works
        data=subset,
        return_type="dataframe",
    )

    if nested:
        groups = nested_group_labels(subset["Producer"], subset["Target"])
    else:
        groups = subset["Producer"]

    # model = sm.OLS(endog, exog) # Converges nicely, R^2=0.52 or something
    model = sm.MixedLM(endog, exog, groups=groups)  # converges with linear round
    # model = sm.MixedLM(endog, exog, subset["Target"]) # converges with linear round
    results = model.fit()

    save_summary_and_plot("6b", results, outdir, extra_desc="no-norm")
    with plt.style.context("seaborn-paper"):
        plt.figure(figsize=(8, 6))
        sns.relplot(
            x="Round",
            y="GenScore",
            size="StructureScore",
            kind="line",
            data=data,
            errorbar=ERRORBAR,
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(outdir, f"model_6b_relplot.png"))


def run_model_7(data: pd.DataFrame, outdir=".", nested=False):
    """Convergence score ~ Structure"""
    print("Running model 7 -- ConvScore trajectory")

    endog, exog = dmatrices(
        "ConvScore ~ scale(StructureScore) * scale(np.log(Round))",
        data=data,
        return_type="dataframe",
    )

    if nested:
        groups = nested_group_labels(data["Producer"], data["Target"])
    else:
        groups = data["Producer"]

    md = sm.MixedLM(endog, exog, groups=groups)
    mdf = md.fit()
    save_summary_and_plot(7, mdf, outdir)

    with plt.style.context("seaborn-paper"):
        plt.figure(figsize=(8, 6))
        sns.relplot(
            x="Round",
            y="ConvScore",
            size="StructureScore",
            kind="line",
            data=data,
            errorbar=ERRORBAR,
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(outdir, "model_7_relplot.png"))


def run_model_8(data: pd.DataFrame, outdir=".", nested=False):
    """Learning trajectory -- PROD SIM to Humans in Generalization"""
    print(
        "Running model 8 -- Learning trajectory -- ProdSim to Humans in Generalization"
    )

    endog, exog = dmatrices(
        "ProdSim_Humans ~ scale(StructureScore) * center(np.log(Round))",
        data=data,
        return_type="dataframe",
    )

    if nested:
        groups = nested_group_labels(data["Producer"], data["Target"])
    else:
        groups = data["Producer"]

    # Note:
    # The critical thing is to put np.log(Round) without scaling.
    # Whether groups are nested Producer/Target or only Producer is not so important
    # (same covered variance)

    md = sm.MixedLM(endog, exog, groups=groups)
    mdf = md.fit()

    save_summary_and_plot(8, mdf, outdir)

    # PLOT
    with plt.style.context("seaborn-paper"):
        plt.figure(figsize=(8, 6))
        sns.relplot(
            x="Round",
            y="ProdSim_Humans",
            size="StructureScore",
            kind="line",
            data=data,
            errorbar=ERRORBAR,
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(outdir, "model_8_relplot.png"))


def run_model_9(data: pd.DataFrame, outdir=".", nested=False):
    """Learning trajectory -- PROD SIM to Humans in Memorization"""
    print(
        "Running model 8 -- Learning trajectory -- ProdSim to Humans in Generalization"
    )

    endog, exog = dmatrices(
        "ProdSim_Humans ~ scale(StructureScore) * scale(np.log(Round))",
        data=data,
        return_type="dataframe",
    )

    if nested:
        groups = nested_group_labels(data["Producer"], data["Target"])
    else:
        groups = data["Producer"]

    md = sm.MixedLM(endog, exog, groups=groups)
    mdf = md.fit()

    save_summary_and_plot(9, mdf, outdir)

    # PLOT
    with plt.style.context("seaborn-paper"):
        plt.figure(figsize=(8, 6))
        sns.relplot(
            x="Round",
            y="ProdSim_Humans",
            size="StructureScore",
            kind="line",
            data=data,
            errorbar=ERRORBAR,
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(outdir, "model_9_relplot.png"))


def main():
    """Run the stats on a folder of results"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_folder", help="Path to results folder with participant files"
    )
    parser.add_argument(
        "--input_languages_file",
        help="Path to input_languges.csv",
        default="data/input_languages.csv",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        help="Path to output folder",
        default="stats_output",
    )
    parser.add_argument(
        "--models_subdir",
        default="models",
        help="Subdir in output folder to put model summaries and plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Output folder: {args.output_folder}")
    mem_data_path = os.path.join(args.output_folder, "mem_data.csv")
    reg_data_path = os.path.join(args.output_folder, "reg_data.csv")

    # Load data
    if not all([os.path.exists(mem_data_path), os.path.exists(reg_data_path)]):
        mem_data, reg_data = load_data(args.results_folder)

    ### COMPUTE ALL SCORES FOR MEMORIZATION DATA ###
    if not os.path.exists(mem_data_path):
        # Add the column 'Word' referring to the ground truth word
        input_languages = pd.read_csv(
            args.input_languages_file, index_col=["InputCondition", "Item.ID"]
        )
        mem_data = mem_data.join(
            input_languages["Word"], on=["InputCondition", "Target"]
        )

        # Calculate prodsim between Input to ground truth (memorization only)
        mem_data["ProdSim_GroundTruth"] = mem_data.apply(
            lambda row: production_similarity(row["Input"], row["Word"]), axis=1
        )
        print(mem_data["ProdSim_GroundTruth"].describe())

        mem_data["ProdSim_Humans"] = mem_data.apply(
            lambda row: production_similarity(row["Input"], row["OrigInput"]), axis=1
        )
        print(mem_data["ProdSim_Humans"].describe())

        mem_data = add_structure_score(mem_data, args.input_languages_file)
        print(f"Saving computed mem scores to {mem_data_path}")
        mem_data.to_csv(mem_data_path, index=True)
    else:
        print(f"Loading precomputed mem scores from {mem_data_path}")
        mem_data = pd.read_csv(mem_data_path, index_col=INDEX_COLUMNS)

    print(mem_data)

    ### COMPUTE ALL SCORES FOR REGULARIZATION DATA ###
    if not os.path.exists(reg_data_path):
        # Calculate convergence score
        reg_data["ConvScore"] = calc_convergence_score(reg_data)
        print(reg_data.ConvScore.describe())
        print(reg_data.head())

        # Calculate generalization score
        genscore, genscore_pval = calc_generalization_score(
            mem_data, reg_data, return_pval=True
        )
        reg_data["GenScore"] = genscore
        reg_data["GenScore_pval"] = genscore_pval
        print(reg_data["GenScore"].describe())
        print(reg_data.head())

        # Calculate prodsim between Input and the input from human learners
        reg_data["ProdSim_Humans"] = reg_data.apply(
            lambda row: production_similarity(row["Input"], row["OrigInput"]), axis=1
        )
        print(reg_data["ProdSim_Humans"].describe())

        reg_data = add_structure_score(reg_data, args.input_languages_file)
        print(f"Saving computed reg scores to {reg_data_path}")
        reg_data.to_csv(reg_data_path, index=True)
    else:
        print(f"Loading precomputed reg scores from {reg_data_path}")
        reg_data = pd.read_csv(reg_data_path, index_col=INDEX_COLUMNS)

    print(reg_data)
    #  mem_data, regdata defined and all scores ready ###

    # THIS NEEDS TO STAY BEFORE resetting index
    reg_data["ConvScore_of_Humans"] = calc_convergence_score(
        reg_data, word_column="OrigInput"
    )
    ### Stats ###

    # Reset indices such that we have access to variables on the index

    mem_data.reset_index(inplace=True)
    reg_data.reset_index(inplace=True)

    # Adding struct bin
    assert "StructureBin" not in mem_data
    mem_data["StructureBin"] = add_structure_bin(mem_data)
    assert "StructureBin" not in reg_data
    reg_data["StructureBin"] = add_structure_bin(reg_data)

    # Run stats models
    model_output_folder = os.path.join(args.output_folder, args.models_subdir)
    print("Will save models to", model_output_folder)
    os.makedirs(model_output_folder, exist_ok=True)

    # run_model_4_with_crossed_random_effects(mem_data, outdir='/tmp') # -> Killed (OOM)
    # run_model_4_with_nested_random_effects(mem_data, outdir='/tmp') # -> With slope per item -> Killed, else ok

    # run_model_1(mem_data)  # Does not work yet
    # run_model_4(mem_data, outdir=model_output_folder)
    # run_model_6(reg_data, outdir=model_output_folder)
    # run_model_7(reg_data, outdir=model_output_folder)
    # run_model_8(reg_data, outdir=model_output_folder)  # Does not converge with scaling, but centering is ok
    # run_model_9(mem_data, outdir=model_output_folder)  # Needs scaling to converge

    # FINAL MODELS
    # run_model_1(mem_data, epoch=100)  # not converged
    # run_model_1(mem_data, epoch=50)   # not converged

    # Added CPPR plots 2022-08-10
    # run_model_2(mem_data, epoch=10, outdir=model_output_folder)
    # run_model_2(mem_data, epoch=40, outdir=model_output_folder)
    # run_model_2(mem_data, epoch=70, outdir=model_output_folder)
    # run_model_2(mem_data, epoch=100, outdir=model_output_folder)

    # run_model_4(mem_data, outdir=model_output_folder, nested=True)
    # run_model_6(reg_data, outdir=model_output_folder, nested=True)

    # TODO 6b updated?
    # run_model_6b(reg_data, outdir=model_output_folder, nested=True)

    # run_model_7(reg_data, outdir=model_output_folder, nested=True)
    # run_model_8(reg_data, outdir=model_output_folder, nested=True)
    # run_model_9(mem_data, outdir=model_output_folder, nested=True)

    # old plots
    # reg_data["GenScore_normalized"] = normalize_genscore(reg_data)
    # make_interaction_plot(mem_data, 'Correct', outdir=model_output_folder, prefix='mem_')
    # make_interaction_plot(mem_data, 'ProdSim_GroundTruth', outdir=model_output_folder, prefix='mem_')
    # make_interaction_plot(mem_data, 'ProdSim_Humans', outdir=model_output_folder, prefix='mem_')

    # make_interaction_plot(reg_data, 'ProdSim_Humans', outdir=model_output_folder, prefix='reg_')
    # make_interaction_plot(reg_data, 'GenScore_normalized', outdir=model_output_folder, prefix='reg_')
    # make_interaction_plot(reg_data, 'ConvScore', outdir=model_output_folder, prefix='reg_')

    print("Calculating human gen scores")
    human_raw_genscores = calc_generalization_score_for_humans(mem_data, reg_data)

    # CUT PLOTS AT 60
    cut = 60
    plots_dir = "plots_binned_at_60"
    os.makedirs(plots_dir, exist_ok=True)
    # make_memorization_plots(mem_data, outdir=plots_dir, cut=cut)
    # make_memorization_panel(mem_data, outdir=plots_dir, cut=cut)
    # make_generalization_plots(
    #     reg_data, outdir=plots_dir, human_raw_genscores=human_raw_genscores, cut=cut
    # )
    # make_generalization_panel(
    #     reg_data, human_raw_genscores=human_raw_genscores, outdir=plots_dir, cut=cut
    # )
    make_big_panel(mem_data, reg_data, human_raw_genscores, outdir=plots_dir, cut=cut)

    # UNCUT PLOTS (at 100)
    cut = None
    plots_dir = "plots_binned_at_100"
    os.makedirs(plots_dir, exist_ok=True)
    # make_memorization_plots(mem_data, outdir=plots_dir, cut=cut)
    # make_memorization_panel(mem_data, outdir=plots_dir, cut=cut)
    # make_generalization_plots(
    #     reg_data, outdir=plots_dir, human_raw_genscores=human_raw_genscores, cut=cut
    # )
    # make_generalization_panel(
    #     reg_data, human_raw_genscores=human_raw_genscores, outdir=plots_dir, cut=cut
    # )
    make_big_panel(mem_data, reg_data, human_raw_genscores, outdir=plots_dir, cut=cut)


if __name__ == "__main__":
    main()
