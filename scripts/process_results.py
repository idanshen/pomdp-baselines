import time
import os
import wandb
from functools import reduce
import itertools
import pandas as pd
import numpy as np
import glob

from tqdm import *

if __name__ == "__main__":


  all_runs = []

  cache_dir_path = ".cache/rebuttal_hand/HandManipulatePen_ContinuousTouchSensors-v1"
  max_steps = 5000000
  interval_size = 50000

  # cache_dir_path = ".cache/rebuttal_tiger/MiniGrid-TigerDoorEnv-v0"
  # max_steps = 30000
  # interval_size = 1000

  # cache_dir_path = ".cache/new_res/Hopper-v3"
  # max_steps = 2000000
  # interval_size = 100000
  # cache_dir_path = ".cache/new_res/HalfCheetah-v3"
  # max_steps = 5000000
  # interval_size = 50000
  # cache_dir_path = ".cache/new_res/Walker2d-v3"
  # max_steps = 4000000
  # interval_size = 50000

  # cache_dir_path = ".cache/fix_ablation/HandManipulatePen_ContinuousTouchSensors-v1"
  # max_steps = 4000000
  # interval_size = 50000

  # cache_dir_path = ".cache/hand_res/HandManipulatePen_ContinuousTouchSensors-v1"
  # max_steps = 16000000
  # interval_size = 50000

  # cache_dir_path = ".cache/sac_ablation/AntGoal-v0"
  # max_steps = 5000000
  # interval_size = 50000

  # cache_dir_path = ".cache/demo_ablation/AntGoal-v0"
  # max_steps = 7500000
  # interval_size = 50000

  # cache_dir_path = ".cache/robustness_ablation/MiniGrid-LavaCrossingS15N10-v0"

  # cache_dir_path = ".cache/data_collection_ablation/MiniGrid-TigerDoorEnv-v0"
  # cache_dir_path = ".cache/data_collection_ablation/MiniGrid-LavaCrossingS15N10-v0"

  # cache_dir_path = ".cache/main_res/MiniGrid-TigerDoorEnv-v0"
  # max_steps = 25000
  # interval_size = 2000
  # cache_dir_path = ".cache/main_res/MiniGrid-MemoryS11-v0"
  # max_steps = 100000
  # interval_size = 10000
  # cache_dir_path = ".cache/main_res/AntGoal-v0"
  # max_steps = 5000000
  # interval_size = 50000
  # cache_dir_path = ".cache/main_res/MiniGrid-LavaCrossingS15N10-v0"
  # max_steps = 175000
  # interval_size = 8000

  prefix_len = len(cache_dir_path) + 1
  for file in tqdm(glob.glob(f"{cache_dir_path}/**/*.csv", recursive=True)):
    cache_file_path = file[prefix_len:-4]
    print(cache_file_path)
    method, seed = cache_file_path.split("/")
    raw_run_df = pd.read_csv(file)

    try:
      raw_run_df = raw_run_df[["z/log_step", "metrics/success_rate_eval"]].dropna()\
        .reset_index()\
        .drop(columns=["index"])\
        .rename(columns={
          "z/log_step": "env steps",
          "metrics/success_rate_eval": "success rate"})
      raw_run_df = raw_run_df[raw_run_df["env steps"] <= max_steps]
      # raw_run_df = raw_run_df.append(pd.DataFrame({"env steps": max_steps, "success rate": raw_run_df["success rate"].values[-1]}, index={raw_run_df.index[-1]}))
      raw_run_df = raw_run_df.append(pd.DataFrame({"env steps": 1, "success rate": 0.0}, index={-1}))
      # raw_run_df = raw_run_df.append(pd.DataFrame({"env_step": max_steps, "success_rate": 1.0}, index={raw_run_df.index[-1]}))

      raw_run_df["env steps"] = pd.cut(raw_run_df['env steps'],
                                      bins=np.linspace(0, max_steps, max_steps // interval_size +1),
                                      labels=np.linspace(0, max_steps, max_steps // interval_size +1)[:-1])
      raw_run_df["method"] = method
      all_runs.append(raw_run_df)
    except KeyError:
      print(f"Invalid file: {file}")

  df = pd.concat(all_runs)
  df.to_csv(os.path.join(cache_dir_path, "processed.csv"))
