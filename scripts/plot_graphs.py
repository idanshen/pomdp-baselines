import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

script_name = os.path.splitext(os.path.basename(__file__))[0]

sns.set(font_scale=1.5)

def plot_fix_ablation2():
    cache_dir_path_ant = ".cache/fix_ablation2/HandManipulatePen_ContinuousTouchSensors-v1/"
    ant_df = pd.read_csv(os.path.join(cache_dir_path_ant, "processed.csv"))
    ant_df["coefficient"] = np.clip(ant_df["coefficient"], 0, 2.5)
    for i in [0.1, 0.3, 0.5, 0.7, 1.0]:
        ant_df = ant_df.append(pd.DataFrame({"env steps": 1, "coefficient": i, "method": "Fixed, "+str(i)}, index={-1}))
        ant_df = ant_df.append(pd.DataFrame({"env steps": 4000000, "coefficient": i, "method": "Fixed, "+str(i)}, index={-1}))
    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    # axes.set_title("")
    order = ["TGRL", "Fixed, 0.1", "Fixed, 0.3", "Fixed, 0.5", "Fixed, 0.7", "Fixed, 0.9", "Fixed, 1.0"]
    # dashes = [False, (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
    ax = sns.lineplot(ax=axes, data=ant_df, x="env steps", y="coefficient", hue="method", hue_order=order)
    for i in range(1,7):
        ax.lines[i].set_linestyle("--")
        ax.lines[i].set_alpha(0.5)
    # axes.axhline(0.1, ls='--', c="gray", label="0.1")
    # axes.axhline(0.3, ls='--', c="gray", label="0.3")
    # axes.axhline(0.5, ls='--', c="gray", label="0.5")
    # axes.axhline(0.7, ls='--', c="gray", label="0.7")
    # axes.axhline(1.0, ls='--', c="gray", label="1.0")
    axes.get_legend().remove()
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes.get_legend_handles_labels()
    # labels = ['Only $\pi_R$', '$\pi_R$ and Teacher', 'Ours']
    # axes.legend(handles, labels, loc='upper right')#, ncol=4)

    plt.show()
    print("done")

def plot_fix_ablation():
    cache_dir_path_ant = ".cache/fix_ablation/HandManipulatePen_ContinuousTouchSensors-v1/"
    ant_df = pd.read_csv(os.path.join(cache_dir_path_ant, "processed.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    # axes.set_title("")
    order = ["TGRL", "Fixed, 0.1", "Fixed, 0.3", "Fixed, 0.5", "Fixed, 0.7","Fixed, 0.9", "Fixed, 1.0"]
    sns.lineplot(ax=axes, data=ant_df, x="env steps", y="success rate", hue="method", hue_order=order)
    axes.get_legend().remove()
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes.get_legend_handles_labels()
    # labels = ['Only $\pi_R$', '$\pi_R$ and Teacher', 'Ours']
    axes.legend(handles, labels, loc='lower right')#, ncol=4)

    plt.show()
    print("done")


def plot_sac_ablation():
    cache_dir_path_ant = ".cache/sac_ablation/AntGoal-v0/"
    ant_df = pd.read_csv(os.path.join(cache_dir_path_ant, "processed.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    # axes.set_title("")
    sns.lineplot(ax=axes, data=ant_df, x="env steps", y="success rate", hue="method", style="policy", hue_order=['shared', 'seperate'])
    axes.get_legend().remove()
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes.get_legend_handles_labels()
    # labels = ['Only $\pi_R$', '$\pi_R$ and Teacher', 'Ours']
    axes.legend(handles, labels, loc='lower right')#, ncol=4)

    plt.show()
    print("done")


def plot_demo_ablation():
    cache_dir_path_ant = ".cache/demo_ablation/AntGoal-v0/"
    ant_df = pd.read_csv(os.path.join(cache_dir_path_ant, "processed.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    # axes.set_title("")
    sns.lineplot(ax=axes, data=ant_df, x="env steps", y="success rate", hue="method")
    axes.get_legend().remove()
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes.get_legend_handles_labels()
    # labels = ['Only $\pi_R$', '$\pi_R$ and Teacher', 'Ours']
    axes.legend(handles, labels, loc='lower right', ncol=2)

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
    axes.legend(handles, labels, loc='lower right', ncol=4)

    plt.show()
    print("done")


def plot_hand_results():
    cache_dir_path_crossing = ".cache/hand_res/HandManipulatePen_ContinuousTouchSensors-v1/"
    hand_df = pd.read_csv(os.path.join(cache_dir_path_crossing, "processed.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    # axes.set_title("Tactile Sensing Pen Reorientation")
    sns.lineplot(ax=axes, data=hand_df, x="env steps", y="success rate", hue="method", hue_order=["Ours", "IL", "SAC"])
    axes.axhline(0.47, ls='--', c="gray", label="SAC, 16M steps")
    axes.axhline(0.78, ls='--', c="black", label="Teacher")
    axes.get_legend().remove()
    box = axes.get_position()
    # axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.set_ylim(0.0, 1.0)
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


def plot_new_res():
    # cache_dir_path = ".cache/data_collection_ablation/MiniGrid-TigerDoorEnv-v0"
    cache_dir_path_hopper = ".cache/new_res/Hopper-v3"
    hopper_df = pd.read_csv(os.path.join(cache_dir_path_hopper, "processed.csv"))
    cache_dir_path_cheetah = ".cache/new_res/HalfCheetah-v3"
    cheetah_df = pd.read_csv(os.path.join(cache_dir_path_cheetah, "processed.csv"))
    cache_dir_path_walker = ".cache/new_res/Walker2d-v3"
    walker_df = pd.read_csv(os.path.join(cache_dir_path_walker, "processed.csv"))
    dfs = [hopper_df, cheetah_df, walker_df]
    titles = ['Hopper-P', 'HalfCheetah-P', 'Walker2d-P']
    hue_order = [
        "TGRL",
        "IL",
    ]
    teacher_reward = [1250, 9000, 3800]
    fig, axes = plt.subplots(1, 3, figsize=(30, 5))

    for i in range(3):
        axes[i].set_title(titles[i])
        dfs[i]['reward'] = dfs[i]['reward'] / teacher_reward[i]
        sns.lineplot(ax=axes[i], data=dfs[i], x="env steps", y="reward", hue="method", hue_order=hue_order)
        axes[i].axhline(1.0, ls='--', c="gray", label="Teacher")
        axes[i].get_legend().remove()
        box = axes[i].get_position()
        axes[i].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
        axes[i].ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)#, columnspacing=10)

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

def plot_ablations():
    fig, axes = plt.subplots(2, 2, figsize=(30, 5))
    cache_dir_path_crossing = ".cache/robustness_ablation/MiniGrid-LavaCrossingS15N10-v0"
    crossing_df = pd.read_csv(os.path.join(cache_dir_path_crossing, "processed.csv"))

    sns.lineplot(ax=axes[0,0], data=crossing_df, x="env steps", y="success rate", hue="method")
    axes[0,0].get_legend().remove()
    box = axes[0,0].get_position()
    axes[0,0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.8])
    axes[0,0].ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes[0,0].get_legend_handles_labels()
    axes[0,0].legend(handles, labels, loc='lower right', ncol=4)

    cache_dir_path_ant = ".cache/sac_ablation/AntGoal-v0/"
    ant_df = pd.read_csv(os.path.join(cache_dir_path_ant, "processed.csv"))

    # axes.set_title("")
    sns.lineplot(ax=axes[0,1], data=ant_df, x="env steps", y="success rate", hue="method", style="policy", hue_order=['shared', 'seperate'])
    axes[0,1].get_legend().remove()
    box = axes[0,1].get_position()
    axes[0,1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.8])
    axes[0,1].ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes[0,1].get_legend_handles_labels()
    # labels = ['Only $\pi_R$', '$\pi_R$ and Teacher', 'Ours']
    axes[0,1].legend(handles, labels, loc='lower right')#, ncol=4)

    cache_dir_path_ant = ".cache/fix_ablation/HandManipulatePen_ContinuousTouchSensors-v1/"
    ant_df = pd.read_csv(os.path.join(cache_dir_path_ant, "processed.csv"))

    # axes.set_title("")
    order = ["TGRL", "Fixed, 0.1", "Fixed, 0.3", "Fixed, 0.5", "Fixed, 0.7","Fixed, 0.9", "Fixed, 1.0"]
    c = sns.color_palette("rocket")
    ax = sns.lineplot(ax=axes[1,1], data=ant_df, x="env steps", y="success rate", hue="method", hue_order=order, palette=[sns.color_palette()[0], c[0], c[1], c[2], c[3], c[4], c[5]])
    axes[1, 1].get_legend().remove()
    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc='lower right')
    for i in range(1,7):
        ax.lines[i].set_linestyle("--")
        ax.lines[i].set_alpha(0.7)

    box = axes[1,1].get_position()
    axes[1,1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.8])
    axes[1,1].ticklabel_format(axis='x', style='sci', scilimits=(4, 4))


    # labels = ['Only $\pi_R$', '$\pi_R$ and Teacher', 'Ours']
    #, ncol=4)


    cache_dir_path_ant = ".cache/fix_ablation2/HandManipulatePen_ContinuousTouchSensors-v1/"
    ant_df = pd.read_csv(os.path.join(cache_dir_path_ant, "processed.csv"))
    ant_df["coefficient"] = np.clip(ant_df["coefficient"], 0, 2.3)
    for i in [0.1, 0.3, 0.5, 0.7, 1.0]:
        ant_df = ant_df.append(pd.DataFrame({"env steps": 1, "coefficient": i, "method": "Fixed, "+str(i)}, index={-1}))
        ant_df = ant_df.append(pd.DataFrame({"env steps": 4000000, "coefficient": i, "method": "Fixed, "+str(i)}, index={-1}))

    # axes.set_title("")
    order = ["TGRL", "Fixed, 0.1", "Fixed, 0.3", "Fixed, 0.5", "Fixed, 0.7","Fixed, 0.9", "Fixed, 1.0"]
    ax = sns.lineplot(ax=axes[1,0], data=ant_df, x="env steps", y="coefficient", hue="method", hue_order=order, palette=[sns.color_palette()[0], c[0], c[1], c[2], c[3], c[4], c[5]])
    for i in range(1,7):
        ax.lines[i].set_linestyle("--")
        ax.lines[i].set_alpha(0.5)
    axes[1,0].get_legend().remove()
    box = axes[1,0].get_position()
    axes[1,0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.8])
    axes[1,0].ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes[1,0].get_legend_handles_labels()
    # labels = ['Only $\pi_R$', '$\pi_R$ and Teacher', 'Ours']
    # axes[1,1].legend(handles, labels, loc='lower right')#, ncol=4)

    plt.show()
    print("done")

def plot_rebuttal_tiger():
    cache_dir_path_crossing = ".cache/rebuttal_tiger/MiniGrid-TigerDoorEnv-v0/"
    hand_df = pd.read_csv(os.path.join(cache_dir_path_crossing, "processed.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    # axes.set_title("Tactile Sensing Pen Reorientation")
    sns.lineplot(ax=axes, data=hand_df, x="env steps", y="success rate", hue="method", hue_order=["Ours"])
    axes.axhline(0.8, ls='--', c="black", label="Teacher")
    axes.axhline(0.5, ls='--', c="gray", label="IL")
    axes.get_legend().remove()
    box = axes.get_position()
    # axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.set_ylim(0.0, 1.01)
    axes.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

    handles, labels = axes.get_legend_handles_labels()
    legend = axes.legend(handles, labels, loc='upper left', ncol=3)
    legend.get_frame().set_facecolor('white')

    plt.show()
    print("done")

def plot_rebuttal_hand():
    cache_dir_path_crossing = ".cache/rebuttal_hand/HandManipulatePen_ContinuousTouchSensors-v1/"
    hand_df = pd.read_csv(os.path.join(cache_dir_path_crossing, "processed.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(30, 5))

    # axes.set_title("Tactile Sensing Pen Reorientation")
    sns.lineplot(ax=axes, data=hand_df, x="env steps", y="success rate", hue="method", hue_order=["Ours", "IL", "SAC", "ADVISOR", "COSIL_avg", "COSIL_best"])
    # axes.axhline(0.47, ls='--', c="gray", label="SAC, 16M steps")
    axes.axhline(0.78, ls='--', c="black", label="Teacher")
    axes.get_legend().remove()
    box = axes.get_position()
    # axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.7])
    axes.set_ylim(0.0, 1.0)
    axes.ticklabel_format(axis='x', style='sci', scilimits=(6, 6))

    handles, labels = axes.get_legend_handles_labels()
    legend = axes.legend(handles, labels, loc='upper left', ncol=3)
    legend.get_frame().set_facecolor('white')

    plt.show()
    print("done")

if __name__ == "__main__":
    # plot_hand_results()
    # plot_main_res()
    # plot_data_collection()
    # plot_robustness_ablation()
    # plot_sac_ablation()
    # plot_demo_ablation()
    # plot_fix_ablation2()
    # plot_new_res()
    # plot_ablations()
    # plot_rebuttal_tiger()
    plot_rebuttal_hand()