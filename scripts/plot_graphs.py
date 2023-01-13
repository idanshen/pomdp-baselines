import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

script_name = os.path.splitext(os.path.basename(__file__))[0]

sns.set(font_scale=1.5)


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
        "start_beta_student_aux_than_teacher",
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
    dfs = [tiger_door_df, memory_df, crossing_df]
    titles = ['Tiger Door', 'Memory', 'Lava Crossing']
    fig, axes = plt.subplots(1, 4, figsize=(30, 5))

    for i in range(3):
        axes[i].set_title(titles[i])
        sns.lineplot(ax=axes[i], data=dfs[i], x="env steps", y="success rate", hue="method")
        axes[i].get_legend().remove()
        box = axes[i].get_position()
        axes[i].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
        axes[i].ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    axes[3].set_title('Ant')
    box = axes[3].get_position()
    axes[3].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, columnspacing=10)

    plt.show()
    print("done")


if __name__ == "__main__":
  plot_data_collection()