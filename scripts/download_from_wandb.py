import time
import os
import wandb
from functools import reduce
import itertools
import pandas as pd

from tqdm import *
api = wandb.Api(timeout=19)

if __name__ == "__main__":
    dir_name = "rebuttal_hand"

    envs = [
        # "MiniGrid-TigerDoorEnv-v0",
        # "MiniGrid-MemoryS11-v0",
        # "MiniGrid-LavaCrossingS15N10-v0",
        # "AntGoal-v0"
        "HandManipulatePen_ContinuousTouchSensors-v1",
        # "HalfCheetah-v3",
        # "Walker2d-v3",
        # "Hopper-v3"
    ]
    data_collection_methods = [
        # "only_student",
        # "all",
        # "start_beta_student_aux_than_teacher",
        "start_student_than_teacher",
        # "both"
    ]
    algo_names = [
        # "advisord",
        # "advisorc",
        # "eaacd",
        # "eaac",
        # "sac",
        # "elfd",
        "DAggerc"
        # "elfc",
    ]

    tuning_methods = [
        # "EIPO",
        # "Target",
        "Fixed"
    ]

    # initial_coefficients = [0.01, 0.3, 0.6, 1]
    os.makedirs(".cache", exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    cache_dir_path = f".cache/{dir_name}"
    os.makedirs(cache_dir_path, exist_ok=True)

    for env_name, method, algo, tuning in (itertools.product(envs, data_collection_methods, algo_names, tuning_methods)):
        runs = api.runs("tsrl/test-project",
                        filters={
                            # "config.seed": 2,
                            # "config.policy.teacher_dir": "/data/pulkitag/models/idanshen/pomdp-baselines/mujoco/HandManipulatePen_ContinuousTouchSensors-v1/Markovian_sac/alpha-0.1/gamma-0.9/01-10:12-01:15.62/",
                            "config.env.env_name": env_name,
                            "config.train.data_collection_method": method,
                            "config.policy.algo_name": algo,
                            # "config.policy.eaac.coefficient_tuning": tuning,
                            "config.policy.seq_model": 'mlp',
                            # "config.policy.sac.entropy_alpha": 0.01,
                            "config.env.obseravibility": "partial",
                        })

        title = "IL"

        cache_path = os.path.join(cache_dir_path, env_name, title)
        os.makedirs(cache_path, exist_ok=True)

        print(f"Checking {cache_path}: {len(runs)}")
        for run in runs:
            run_cache_dir_path = os.path.join(cache_path,
                                              f"{run.id}.csv")
            # os.makedirs(run_cache_dir_path, exist_ok=True)
            # run_cache_file_path = os.path.join(run_cache_dir_path, f"{run.id}.csv")
            if not os.path.exists(run_cache_dir_path):
                print(f"Download {run_cache_dir_path}")
                metrics_dataframe = pd.DataFrame(run.scan_history())
                metrics_dataframe.to_csv(run_cache_dir_path)
