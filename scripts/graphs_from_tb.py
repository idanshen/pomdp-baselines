from matplotlib import pyplot as plt
import numpy as np
import tensorboard_reducer as tbr
from scipy.interpolate import interp1d

mean = {}
std = {}
# TigerDoor
# input_event_dirs = {0.01: ["/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.01/gamma-0.9/11-08:17-22:07.16/",
#                            "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.01/gamma-0.9/11-08:17-25:56.00/",
#                            "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.01/gamma-0.9/11-08:17-40:23.03/",
#                            "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.01/gamma-0.9/11-08:17-41:00.29/",
#                           ],
#                     0.3: ["/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.3/gamma-0.9/11-08:18-57:50.32/",
#                           "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.3/gamma-0.9/11-08:19-01:00.00/",
#                           "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.3/gamma-0.9/11-08:19-16:21.90/",
#                           "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-0.3/gamma-0.9/11-08:19-16:31.79/",
#                           ],
#                     1.0: ["/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-1.0/gamma-0.9/11-08:18-02:38.71/",
#                           "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-1.0/gamma-0.9/11-08:18-21:00.99/",
#                           "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/TigerDoorEnv/Markovian_eaacd/alpha-1.0/gamma-0.9/11-08:18-33:32.36/",
#                           ],
#                     }

# LavaGapS7
input_event_dirs = {0.01: ["/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.01/gamma-0.9/11-10:23-14:20.24/",
                           "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.01/gamma-0.9/11-11:10-50:16.53/",
                           "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.01/gamma-0.9/11-10:23-34:18.32/",
                           "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.01/gamma-0.9/11-11:11-05:27.67/",
                           "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.01/gamma-0.9/11-11:11-18:36.97/",
                          ],
                    0.3: ["/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.3/gamma-0.9/11-11:00-49:55.34/",
                          "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.3/gamma-0.9/11-11:01-06:31.34/",
                          "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.3/gamma-0.9/11-11:01-20:49.00/",
                          "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.3/gamma-0.9/11-11:12-14:34.05/",
                          "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-0.3/gamma-0.9/11-11:12-30:04.74/",
                          ],
                    1.0: ["/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-1.0/gamma-0.9/11-10:23-47:38.18/",
                          "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-1.0/gamma-0.9/11-11:00-11:48.61/",
                          "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-1.0/gamma-0.9/11-11:00-31:03.68/",
                          "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-1.0/gamma-0.9/11-11:11-28:22.02/",
                          "/home/idanshen/pomdp-baselines/gridworld/MiniGrid/LavaGapS7/Markovian_eaacd/alpha-1.0/gamma-0.9/11-11:11-52:42.94/",
                          ],
                    }
x = np.linspace(0, 40000, 21)
stats = {}
for key in input_event_dirs.keys():
    stats[key] = []
    for exp in input_event_dirs[key]:
        events_dict = tbr.load_tb_events([exp])
        data = events_dict["metrics/return_eval_avg"]
        index = np.concatenate([[0],np.array(data.index)], axis=0)
        values = np.concatenate([[-200],np.array(data.values).squeeze()], axis=0)
        fit = interp1d(index, values)
        stats[key].append(fit(x))
    stats[key] = np.array(stats[key])
    mean[str(key)] = stats[key].mean(axis=0)
    std[str(key)] = stats[key].std(axis=0)

plt.clf()
for j, color, facecolor in zip(['0.01', '0.3', '1.0'], ['#3F7F4C','#1B2ACC','#CC4F1B'], ['#7EFF99', '#089FFF', '#FF9848']):
    y = np.array(mean[j])
    error = np.array(std[j])
    plt.plot(x, y, 'k', color =color,label=j)
    plt.fill_between(x, y-error, y+error,
        alpha=0.3, facecolor=facecolor,
        linewidth=0)
    plt.xlabel("env steps")
    plt.ylabel("cumulative reward")
plt.legend()
plt.show()
print("end")