import time
import os
import wandb
from functools import reduce
import itertools
import pandas as pd

from tqdm import *
api = wandb.Api(timeout=19)

if __name__ == "__main__":
    dir_name = "sac_ablation"

    envs = [
        # "MiniGrid-TigerDoorEnv-v0",
        # "MiniGrid-MemoryS11-v0",
        # "MiniGrid-LavaCrossingS15N10-v0",
        "AntGoal-v0"
    ]
    data_collection_methods = [
        "only_student",
        # "all",
        # "start_beta_student_aux_than_teacher",
    ]
    algo_names = [
        # "advisord",
        # "eaacd",
        # "eaac",
        "sac",
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
                            "config.env.env_name": env_name,
                            "config.train.data_collection_method": method,
                            "config.policy.algo_name": algo,
                            "config.policy.eaac.coefficient_tuning": tuning,
                            # "config.policy.eaacd.initial_coefficient": init_coeff,
                        })

        # if algo == "advisord":
        #     title = "ADVISOR"
        # elif algo == "eaacd":
        #     if tuning == "EIPO":
        #         title = "Ours"
        #     # elif tuning == "Fixed":
        #     #     title = "ADVISOR"
        #     else:
        #         title = "COSIL"
        # else:
        #     raise ValueError

        title = tuning

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
