import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

script_name = os.path.splitext(os.path.basename(__file__))[0]

sns.set(font_scale=1.5)


def plot_sac_ablation():
    cache_dir_path_ant = ".cache/sac_ablation/AntGoal-v0/"
    ant_df = pd.read_csv(os.path.join(cache_dir_path_ant, "processed.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    # axes.set_title("")
    sns.lineplot(ax=axes, data=ant_df, x="env steps", y="success rate", hue="method")
    axes.get_legend().remove()
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes.get_legend_handles_labels()
    labels = ['Only $\pi_R$', '$\pi_R$ and Teacher', 'Ours']
    fig.legend(handles, labels, loc='lower center', ncol=4)

    plt.show()
    print("done")


def plot_robustness_ablation():
    cache_dir_path_crossing = ".cache/robustness_ablation/MiniGrid-LavaCrossingS15N10-v0"
    crossing_df = pd.read_csv(os.path.join(cache_dir_path_crossing, "processed.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    axes.set_title("Lava Crossing")
    sns.lineplot(ax=axes, data=crossing_df, x="env steps", y="success rate", hue="method")
    axes.get_legend().remove()
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)

    plt.show()
    print("done")


def plot_hand_results():
    cache_dir_path_crossing = ".cache/hand_res/HandManipulatePen_ContinuousTouchSensors-v1/"
    hand_df = pd.read_csv(os.path.join(cache_dir_path_crossing, "processed.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    # axes.set_title("Tactile Sensing Pen Reorientation")
    sns.lineplot(ax=axes, data=hand_df, x="env steps", y="success rate", hue="method")
    axes.axhline(0.47, ls='--', c="gray", label="SAC, 16M steps")
    axes.axhline(0.78, ls='--', c="black", label="Teacher")
    axes.get_legend().remove()
    box = axes.get_position()
    # axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.ticklabel_format(axis='x', style='sci', scilimits=(6, 6))

    handles, labels = axes.get_legend_handles_labels()
    legend = axes.legend(handles, labels, loc='upper left', ncol=3)
    legend.get_frame().set_facecolor('white')

    plt.show()
    print("done")


def plot_data_collection():
    cache_dir_path_tiger_door = ".cache/data_collection_ablation/MiniGrid-TigerDoorEnv-v0"
    tiger_door_df = pd.read_csv(os.path.join(cache_dir_path_tiger_door, "processed.csv"))
    cache_dir_path_crossing = ".cache/data_collection_ablation/MiniGrid-LavaCrossingS15N10-v0"
    crossing_df = pd.read_csv(os.path.join(cache_dir_path_crossing, "processed.csv"))
    dfs = [tiger_door_df, crossing_df]
    titles = ['Tiger Door', 'Lava Crossing']
    hue_order = [
        "only_student",
        "all",
        "start_student_than_teacher",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(30, 5))

    for i in range(2):
        axes[i].set_title(titles[i])
        sns.lineplot(ax=axes[i], data=dfs[i], x="env steps", y="success rate", hue="method", hue_order=hue_order)
        axes[i].get_legend().remove()
        box = axes[i].get_position()
        axes[i].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
        axes[i].ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, columnspacing=10)

    plt.show()
    print("done")


def plot_main_res():
    # cache_dir_path = ".cache/data_collection_ablation/MiniGrid-TigerDoorEnv-v0"
    cache_dir_path_tiger_door = ".cache/main_res/MiniGrid-TigerDoorEnv-v0"
    tiger_door_df = pd.read_csv(os.path.join(cache_dir_path_tiger_door, "processed.csv"))
    cache_dir_path_memory = ".cache/main_res/MiniGrid-MemoryS11-v0"
    memory_df = pd.read_csv(os.path.join(cache_dir_path_memory, "processed.csv"))
    cache_dir_path_crossing = ".cache/main_res/MiniGrid-LavaCrossingS15N10-v0"
    crossing_df = pd.read_csv(os.path.join(cache_dir_path_crossing, "processed.csv"))
    cache_dir_path_ant = ".cache/main_res/AntGoal-v0"
    ant_df = pd.read_csv(os.path.join(cache_dir_path_ant, "processed.csv"))
    dfs = [tiger_door_df, memory_df, crossing_df, ant_df]
    titles = ['Tiger Door', 'Memory', 'Lava Crossing', 'Light-Dark Ant']
    hue_order = [
        "Ours",
        "COSIL_best",
        "COSIL_avg",
        "ADVISOR",
        "PBRS"
    ]
    IL_res = [0.5, 0.5, 0.88, 0.79]
    fig, axes = plt.subplots(1, 4, figsize=(30, 5))

    for i in range(4):
        axes[i].set_title(titles[i])
        sns.lineplot(ax=axes[i], data=dfs[i], x="env steps", y="success rate", hue="method", hue_order=hue_order)
        axes[i].axhline(IL_res[i], ls='--', c="gray", label="IL")
        axes[i].get_legend().remove()
        box = axes[i].get_position()
        axes[i].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
        axes[i].ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6)#, columnspacing=10)

    plt.show()
    print("done")


if __name__ == "__main__":
    # plot_hand_results()
    plot_main_res()
    # plot_data_collection()
    # plot_robustness_ablation()
    # plot_sac_ablation()
