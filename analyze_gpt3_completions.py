import os
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from language import load_input_language
from measures import production_similarity, generalization_score
from learning_experiment import scenes
from tqdm import tqdm

from stats import calc_generalization_score_for_humans

# mpl.rcParams["figure.dpi"] = 300
sns.set_theme(context="paper", style="whitegrid", font_scale=2.0)

GPT3_COMPLETIONS_DIR = "../gpt3-completions/"

NN_RESULTS_DIR = "../results-v1-stats-output-v2/"

ALL_LANGUAGES = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4", "B5"]
STRUCTURE_BINS = {
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


def analyze_gpt3_results(lang_ids):
    mem_test_frames = []
    reg_test_frames = []
    for lang_id in tqdm(lang_ids):
        lang = load_input_language(lang_id)

        ### MEMORIZATION TEST ###
        mem_test_data = pd.read_csv(
            os.path.join(GPT3_COMPLETIONS_DIR, f"gpt3-{lang_id}-mem-test.csv")
        )

        mem_test_data["Task"] = "MemorizationTest"
        mem_test_data["InputCondition"] = lang_id
        mem_test_data["StructureScore"] = lang.get_unique_attribute("StructureScore")
        mem_test_data["StructureBin"] = STRUCTURE_BINS[lang.name]
        mem_test_data["GroundTruth"] = mem_test_data["Target"].map(lang.get_word_by_id)

        mem_test_data["Correct"] = mem_test_data.apply(
            lambda row: 1.0 if row["Input"] == row["GroundTruth"] else 0.0,
            axis=1,
        )
        mem_test_data["ProdSim_GroundTruth"] = mem_test_data.apply(
            lambda row: production_similarity(row["Input"], row["GroundTruth"]),
            axis=1,
        )

        reg_test_data = pd.read_csv(
            os.path.join(GPT3_COMPLETIONS_DIR, f"gpt3-{lang_id}-reg-test.csv")
        )

        reg_test_data["Task"] = "RegularizationTest"
        reg_test_data["InputCondition"] = lang_id
        reg_test_data["StructureScore"] = lang.get_unique_attribute("StructureScore")
        reg_test_data["StructureBin"] = STRUCTURE_BINS[lang.name]

        familiar_scenes = scenes(mem_test_data)
        familiar_labels = mem_test_data["Input"]
        new_scenes = scenes(reg_test_data)
        new_labels = reg_test_data["Input"]

        genscore, __ = generalization_score(
            familiar_scenes,
            familiar_labels,
            new_scenes,
            new_labels,
            scene_metric="semantic_difference",
            label_metric="normalized_editdistance",
            rescale=False,
        )
        reg_test_data["GenScore"] = genscore

        mem_test_frames.append(mem_test_data)
        reg_test_frames.append(reg_test_data)

    return (
        pd.concat(mem_test_frames, ignore_index=True),
        pd.concat(reg_test_frames, ignore_index=True),
    )


def compare_gpt3_to_humans(gpt3_results: pd.DataFrame, human_results: pd.DataFrame):
    sim = []
    N = 0
    for __idx, gpt3_row in gpt3_results.iterrows():
        gpt3_word, lang, shape, angle = (
            gpt3_row.Input,
            gpt3_row.InputCondition,
            gpt3_row.Shape,
            gpt3_row.Angle,
        )
        same_scene = human_results[
            (human_results.InputCondition == lang)
            & (human_results.Shape == shape)
            & (human_results.Angle == angle)
        ]
        N_match = len(same_scene)
        print(f"Found {N_match} matching human generations")

        tmp = []
        for __idx2, human_production in same_scene.iterrows():
            human_word = human_production.OrigInput  # OrigInput is human production
            prodsim = production_similarity(gpt3_word, human_word)
            tmp.append(prodsim)

        # one result per gpt-3 generation
        sim.append(sum(tmp) / len(tmp))

        N += N_match

    print(f"---\n{N} total comparisons")
    return sim


def compare_humans_to_humans(human_results: pd.DataFrame):
    """ Added 2024-07-01, lg -- compare human productions to each other
    
    Make sure to not compute the same participant with itself.
    We can reconstruct the human participant id by using the participant id modulo 1000"""
    sim = []
    N = 0
    for __idx, row in human_results.iterrows():
        word, lang, shape, angle, producer = (
            row.OrigInput,
            row.InputCondition,
            row.Shape,
            row.Angle,
            row.Producer
        )
        same_scene = human_results[
            (human_results.InputCondition == lang)
            & (human_results.Shape == shape)
            & (human_results.Angle == angle)
            & ((human_results.Producer % 1000) != (producer % 1000))  # NO SELF COMPARISON!
        ]
        N_match = len(same_scene)
        print(f"Found {N_match} matching human generations")

        tmp = []
        for __idx2, human_production in same_scene.iterrows():
            human_word = human_production.OrigInput  # OrigInput is human production
            prodsim = production_similarity(word, human_word)
            tmp.append(prodsim)

        # one result per human generation
        sim.append(sum(tmp) / len(tmp))

        N += N_match

    print(f"---\n{N} total comparisons")
    return sim


def compare_RNNs_to_humans(rnn_results: pd.DataFrame, human_results: pd.DataFrame):
    """ Added 2024-07-01, lg -- compare RNN productions to each other
    
    Make sure to not compute the same participant with itself.
    We can reconstruct the human participant id by using the participant id modulo 1000"""
    sim = []
    N = 0
    for __idx, row in tqdm(rnn_results.iterrows(), desc="Calculating RNNs2Human similarity"):
        word, lang, shape, angle = (
            row.Input,
            row.InputCondition,
            row.Shape,
            row.Angle
        )
        same_scene = human_results[
            (human_results.InputCondition == lang)
            & (human_results.Shape == shape)
            & (human_results.Angle == angle)
        ]
        N_match = len(same_scene)
        print(f"Found {N_match} matching human generations")

        tmp = []
        for __idx2, human_production in same_scene.iterrows():
            human_word = human_production.OrigInput  # OrigInput is human production
            prodsim = production_similarity(word, human_word)
            tmp.append(prodsim)

        # one result per human generation
        sim.append(sum(tmp) / len(tmp))

        N += N_match

    print(f"---\n{N} total comparisons")
    return sim


def plot_results(mem_results: pd.DataFrame, reg_results: pd.DataFrame, name: str = ""):
    colormap = plt.get_cmap(
        "copper"
    ).reversed()  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    hue_norm = (1, 5)
    errorbar = "se"
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(10, 4))

    sns.regplot(
        x="StructureScore",
        y="ProdSim_GroundTruth",
        # x_bins=10,
        # hue="StructureBin",
        # hue_norm=hue_norm,
        # palette=colormap,
        data=mem_results,
        ax=ax1,
    )
    ax1.set_title(f"(a) {name} Memorization")
    ax1.set_ylabel("Prod. Sim.")
    sns.regplot(
        x="StructureScore",
        y="GenScore",
        # x_bins=10,
        # hue="StructureBin",
        # hue_norm=hue_norm,
        # palette=colormap,
        data=reg_results,
        ax=ax2,
    )
    ax2.set_title(f"(b) {name} Generalization")
    ax2.set_ylabel("Generalization Score")

    fig.tight_layout(pad=1.0)
    plt.show()


def error_analysis(mem_results, name=""):
    mem_results_subset = mem_results[mem_results["ProdSim_GroundTruth"] < 1.0]
    p_error = 100 * (len(mem_results_subset) / len(mem_results))

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(10, 4))
    # fig, ax = plt.figure(figsize=(10, 4))
    sns.regplot(
        x="StructureScore", y="ProdSim_GroundTruth", data=mem_results_subset, ax=ax
    )
    ax.set_title(f"Error Analysis of {name} Memorization ({p_error:.2f}%)")
    ax.set_ylabel("Prod. Sim.")
    fig.tight_layout(pad=1.0)
    plt.show()


def main():
    gpt3_mem_results, gpt3_reg_results = analyze_gpt3_results(ALL_LANGUAGES)

    # plot_results(gpt3_mem_results, gpt3_reg_results, name="GPT-3.5")
    # error_analysis(gpt3_mem_results)

    print(f"Trying to load processed results from '{NN_RESULTS_DIR}'")
    print("(Make sure that stats.py was run before.)")
    mem_results = pd.read_csv(os.path.join(NN_RESULTS_DIR, "mem_data.csv"))
    reg_results = pd.read_csv(os.path.join(NN_RESULTS_DIR, "reg_data.csv"))

    at_round = 100

    # Reduce all data to final round
    mem_results = mem_results[mem_results.Round == at_round]
    reg_results = reg_results[reg_results.Round == at_round]

    # One seed to fetch original human data
    human_mem_results = mem_results[mem_results.Producer < 2000].copy()
    human_reg_results = reg_results[reg_results.Producer < 2000].copy()

    # fig, ax1 = plt.subplots(
    #     1,
    #     1,
    #     sharex=True,
    #     sharey=True,
    #     figsize=(12, 6),
    # )
    # sns.regplot(x="StructureScore", y="Correct", data=mem_results, ax=ax1)
    # plt.show()
    # exit(0)

    human_genscores = calc_generalization_score_for_humans(
        human_mem_results, human_reg_results, at_round=at_round
    )
    kwargs = {
        "x_jitter": 0.02,
        "y_jitter": 0.02,
        "marker": ".",
        "line_kws": {"color": "r"},
        "scatter_kws": {"alpha": 0.4},
    }

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        sharex=True,
        sharey=True,
        figsize=(12, 6),
        dpi=300,
    )
    # fig.suptitle("Systematic Generalization")

    sns.regplot(
        x="StructureScore",
        y="GenScore_of_Humans",
        data=human_genscores,
        ax=ax1,
        **kwargs,
    )
    N = len(human_genscores)
    ax1.set_title(f"(A) Humans")
    print(f"Sys. Gen.: Humans N={N}")
    ax1.set_ylabel("Generalization Score")
    ax1.set_xlabel("Struct. Score")

    # print(gpt3_reg_results)
    rows = []
    for lang_id, group in reg_results.groupby("InputCondition"):
        row = pd.DataFrame(
            {
                "StructureScore": group["StructureScore"].mean(),
                "GenScore": group["GenScore"].mean(),
            },
            index=[lang_id],
        )
        rows.append(row)
    gpt3_genscores = pd.concat(rows)
    N = len(gpt3_genscores)
    sns.regplot(x="StructureScore", y="GenScore", data=gpt3_genscores, ax=ax2, **kwargs)
    ax2.set_title(f"(B) GPT-3.5")
    print(f"Sys. Gen.: GPT-3.5 N={N}")
    # ax2.set_ylabel("Gen. Score")
    ax2.set_xlabel("Struct. Score")
    ax2.set(ylabel=None)

    rows = []
    for seed, group in reg_results.groupby("Producer"):
        row = pd.DataFrame(
            {
                "InputCondition": group["InputCondition"].max(),
                "StructureScore": group["StructureScore"].mean(),
                "GenScore": group["GenScore"].mean(),
            },
            index=[seed],
        )
        rows.append(row)
    nn_genscores = pd.concat(rows)
    N = len(nn_genscores)
    sns.regplot(x="StructureScore", y="GenScore", data=nn_genscores, ax=ax3, **kwargs)
    ax3.set_title(f"(C) RNNs")
    print(f"Sys. Gen.: RNNs N={N}")
    # ax3.set_ylabel("Gen. Score")
    ax3.set_xlabel("Struct. Score")
    ax3.set(ylabel=None)

    fig.tight_layout(pad=1.0)
    plt.savefig("gpt3-generalization-plot.pdf")

    ### ERROR ANALYSIS ###

    print("Computing prod sim between humans and ground truth")
    human_mem_results["ProdSim_bw_Humans_n_GroundTruth"] = human_mem_results.apply(
        lambda row: production_similarity(row["OrigInput"], row["Word"]),
        axis=1,
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        sharex=True,
        sharey=True,
        figsize=(12, 6),
        dpi=300,
    )
    # fig.suptitle("Memorization Error Analysis")

    # Humans
    human_errors = human_mem_results[
        human_mem_results["ProdSim_bw_Humans_n_GroundTruth"] < 1.0
    ]
    p_error = 100 * (len(human_errors) / len(human_mem_results))
    sns.regplot(
        x="StructureScore",
        y="ProdSim_bw_Humans_n_GroundTruth",
        data=human_errors,
        ax=ax1,
        **kwargs,
    )
    N = len(human_errors)
    ax1.set_title(f"(A) Humans")
    print(f"Human error rate: {p_error:.2f}% ({N} errors)")
    ax1.set_ylabel("True Label Similarity")
    ax1.set_xlabel("Struct. Score")
    # sns.regplot(x="StructureScore", y="ProdSim_GroundTruth", data=mem_results, ax=ax1)
    # ax1.set_title(f"Humans")
    # ax1.set_ylabel("Prod. Sim.")
    # ax1.set_xlabel("Struct. Score")

    # GPT-3.5
    gpt3_mem_errors = gpt3_mem_results[gpt3_mem_results["ProdSim_GroundTruth"] < 1.0]
    p_error = 100 * (len(gpt3_mem_errors) / len(gpt3_mem_results))
    sns.regplot(
        x="StructureScore",
        y="ProdSim_GroundTruth",
        data=gpt3_mem_errors,
        ax=ax2,
        **kwargs,
    )
    N = len(gpt3_mem_errors)
    ax2.set_title(f"(B) GPT-3.5")
    print(f"GPT-3.5 error rate: {p_error:.2f}% ({N} errors)")
    ax2.set_ylabel(None)
    ax2.set_xlabel("Struct. Score")

    # RNNs
    mem_results_errors = mem_results[mem_results["ProdSim_GroundTruth"] < 1.0]
    p_error = 100 * (len(mem_results_errors) / len(mem_results))
    sns.regplot(
        x="StructureScore",
        y="ProdSim_GroundTruth",
        data=mem_results_errors,
        ax=ax3,
        **kwargs,
    )
    N = len(mem_results_errors)
    ax3.set_title(f"(C) RNNs")
    print(f"RNNs error rate: {p_error:.2f}% ({N} errors)")
    ax3.set_ylabel(None)
    ax3.set_xlabel("Struct. Score")

    fig.tight_layout(pad=1.0)
    plt.savefig("gpt3-error-analysis-plot.pdf")

    ############################
    ### Similarity to Humans ###
    ############################

    ## NEW, 2023-08-15, lg
    ## Updated with human/human similarity, 2024-07-01, lg

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        sharex=True,
        sharey=True,
        figsize=(12, 6),
        dpi=300,
    )
    # fig.suptitle("Similarity to Humans during Generalization")


    assert "ProdSim_Humans2humans" not in reg_results
    print("Computing human2human similarity")
    human_reg_results["ProdSim_Humans2humans"] = compare_humans_to_humans(human_reg_results)
    sns.regplot(
        x="StructureScore",
        y="ProdSim_Humans2humans",
        data=human_reg_results,
        ax=ax1,
        **kwargs
    )
    ax1.set_title(f"(A) Humans")
    ax1.set_ylabel("Human Label Similarity")
    ax1.set_xlabel("Struct. Score")

    assert "ProdSim_Humans" not in gpt3_reg_results.columns
    gpt3_reg_results["ProdSim_Humans"] = compare_gpt3_to_humans(
        gpt3_reg_results, human_reg_results
    )

    sns.regplot(
        x="StructureScore",
        y="ProdSim_Humans",
        data=gpt3_reg_results,
        ax=ax2,
        **kwargs,
    )
    N = len(gpt3_reg_results)
    ax2.set_title(f"(B) GPT-3.5")
    ax2.set_ylabel(None)
    ax2.set_xlabel("Struct. Score")


    assert "Prodsim_RNNs2Humans" not in reg_results
    reg_results["ProdSim_RNNs2Humans"] = compare_RNNs_to_humans(reg_results, human_reg_results)

    sns.regplot(
        x="StructureScore",
        y="ProdSim_RNNs2Humans",
        data=reg_results,
        ax=ax3,
        **kwargs
    )
    N = len(reg_results)
    ax3.set_title(f"(C) RNNs")
    ax3.set_ylabel(None)
    ax3.set_xlabel("Struct. Score")

    fig.tight_layout(pad=1.0)
    plt.savefig("gpt3-sim2humans-plot.pdf")


if __name__ == "__main__":
    main()
