from matplotlib import pyplot as plt
import numpy as np
import tensorboard_reducer as tbr

mean = {}
std = {}
input_event_dirs = {0.1: ["/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.1/gamma-0.9/11-07:14-54:14.28/",
                          ],
                    0.5: ["/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.5/gamma-0.9/11-06:22-31:33.27/",
                          ],
                    1.0: ["/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-1.0/gamma-0.9/11-06:23-08:18.89/",
                          ],
                    }
for key in input_event_dirs.keys():
    # where to write reduced TB events, each reduce operation will be in a separate subdirectory
    tb_events_output_dir = "/home/idanshen/gridworld/"
    csv_out_path = "/home/idanshen/gridworld/reduced_"+str(key)+".csv"
    # whether to abort or overwrite when csv_out_path already exists
    overwrite = False
    reduce_ops = ("mean", "std")

    events_dict = tbr.load_tb_events(input_event_dirs[key])

    # number of recorded tags. e.g. would be 3 if you recorded loss, MAE and R^2
    n_scalars = len(events_dict)
    n_steps, n_events = list(events_dict.values())[0].shape

    print(
        f"Loaded {n_events} TensorBoard runs with {n_scalars} scalars and {n_steps} steps each"
    )
    print(", ".join(events_dict))

    reduced_events = tbr.reduce_events(events_dict, reduce_ops)
    # mean[str(key)] = reduced_events["mean"]["metrics/return_eval_avg"]
    mean[str(key)] = events_dict['metrics/return_eval_avg'].mean(axis=1)
    std[str(key)] = reduced_events["std"]["metrics/return_eval_avg"]
    # for op in reduce_ops:
    #     print(f"Writing '{op}' reduction to '{tb_events_output_dir}-{op}'")

    # tbr.write_tb_events(reduced_events, tb_events_output_dir, overwrite)

    # print(f"Writing results to '{csv_out_path}'")
    #
    # tbr.write_data_file(reduced_events, csv_out_path, overwrite)
    #
    # print("Reduction complete")
plt.clf()
for j, color, facecolor in zip(['0.1', '0.5', '1.0'], ['#3F7F4C','#1B2ACC','#CC4F1B'], ['#7EFF99', '#089FFF', '#FF9848']):
    x = np.array(mean[j].index)
    y = np.array(mean[j].values)
    error = np.array(std[j].values)
    plt.plot(x, y, 'k', color =color,label=j)
    plt.fill_between(x, y-error, y+error,
        alpha=0.3, facecolor=facecolor,
        linewidth=0)
plt.legend()
plt.show()
print("end")