

"""Evaluator class."""

import numpy as np


# library imports
import os
import sys
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class GateEvaluator():

    def __init__(self, gate_cfg, training_ts=1E8, num_envs=100, device="cuda:0") -> None:
        self.gate_cfg = gate_cfg

        self.training_ts = training_ts
        self.num_envs = num_envs

        self.id_counter = 0
        self.evaluation_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "gate_train.py"))

        self.device = device

    def run_subprocess(self, individual_save_dir, num=None):
        env = os.environ.copy()

        # Make sure the subprocess gets the same PYTHONPATH and venv path
        venv_bin = os.path.dirname(sys.executable)
        env["PATH"] = f"{venv_bin}:{env['PATH']}"
        
        # Add site-packages if needed (important for editable installs)
        site_packages_path = os.path.join(venv_bin, "..", "lib", "python3.10", "site-packages")
        env["PYTHONPATH"] = f"{site_packages_path}:{env.get('PYTHONPATH', '')}"

        # print(env)
        # Spawn a subprocess to evaluate the morphology
        if num is not None:
            process = subprocess.Popen(
                ["python3", self.evaluation_file, individual_save_dir, "--training_timesteps", str(self.training_ts), "--num_envs", str(self.num_envs), "--gate_cfg", self.gate_cfg, "--device", self.device, "--num", str(num)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            process = subprocess.Popen(
                ["python3", self.evaluation_file, individual_save_dir, "--training_timesteps", str(self.training_ts), "--num_envs", str(self.num_envs), "--gate_cfg", self.gate_cfg, "--device", self.device],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        return process

    def evaluate_population(self, population, gen_save_dir=None):
        individual_save_dirs = []
        for i in range(len(population)):
            individual = population[i]
            individual_save_dir = gen_save_dir + f"ind{self.id_counter + i}/"
            individual_save_dirs.append(individual_save_dir)

            if not os.path.exists(individual_save_dir):
                os.makedirs(individual_save_dir)
            np.save(individual_save_dir + "individual.npy", individual)

        self.id_counter += len(population)

        batch_size = 3
        batches = [individual_save_dirs[i:i + batch_size] for i in range(0, len(individual_save_dirs), batch_size)]
        results = []
        for batch in batches:
            start = time.time()
            processes = []
            for individual_save_dir in batch:
                processes.append((individual_save_dir, self.run_subprocess(individual_save_dir)))

            for individual_save_dir, process in processes:
                stdout, stderr = process.communicate()  # Wait for process to complete and get output
                if process.returncode == 0:
                    results.append((individual_save_dir, stdout.strip()))
                else:
                    print("Return code: ", process.returncode)
                    print(individual_save_dir) 
                    print(stdout.strip())
                    raise Exception(f"Error: {stderr.strip()}")
            end = time.time()
            print(f"Batch time: {end-start} Seconds", flush=True)
        # Sort results by individual_save_dir using numpy
        results = np.array(sorted(results, key=lambda x: x[0]))
        fitnesses = np.array(results[:,1], dtype=float)

        return fitnesses
    

class NSGAIIGateEvaluator():

    def __init__(self, task1, task2, training_ts1=1E8, num_envs1=100, training_ts2=1E8, num_envs2=100, device="cuda:0") -> None:
        self.device = device
        self.evaluator1 = GateEvaluator(gate_cfg=task1, training_ts=training_ts1, num_envs=num_envs1, device=device)
        self.evaluator2 = GateEvaluator(gate_cfg=task2, training_ts=training_ts2, num_envs=num_envs2, device=device)

        self.id_counter = 0

    def evaluate_population(self, population, gen_save_dir=None):

        individual_save_dirs = []
        for i in range(len(population)):
            individual = population[i]
            individual_save_dir = gen_save_dir + f"ind{self.id_counter + i}/"
            individual_save_dirs.append(individual_save_dir)

            if not os.path.exists(individual_save_dir):
                os.makedirs(individual_save_dir)
            np.save(individual_save_dir + "individual.npy", individual)

        self.id_counter += len(population)

        batch_size = 4
        batches = [individual_save_dirs[i:i + batch_size] for i in range(0, len(individual_save_dirs), batch_size)]
        results = []
        for batch in batches:
            start = time.time()
            task_fitnesses = []
            for task_num in range(2):
                
                if task_num == 0:
                    evaluator = self.evaluator1
                else:
                    evaluator = self.evaluator2

                processes = []
                for individual_save_dir in batch:

                    processes.append((individual_save_dir, evaluator.run_subprocess(individual_save_dir, num=task_num)))

                for individual_save_dir, process in processes:
                    stdout, stderr = process.communicate()
                    if process.returncode == 0:
                        task_fitnesses.append(stdout.strip())
                        
                    else:
                        raise Exception(f"Error: {stderr.strip()}")
            task_fitnesses = np.array(task_fitnesses, dtype=float).reshape(2, -1).T
            for individual_save_dir, fitnesses in zip(batch, task_fitnesses):
                results.append([individual_save_dir, fitnesses])
                
            end = time.time()
            print(f"Batch time: {end-start} Seconds", flush=True)

        results = sorted(results, key=lambda x: x[0])
        sorted_results = np.array([result[1] for result in results])

        fitnesses1 = sorted_results[:, 0]
        fitnesses2 = sorted_results[:, 1]

        return fitnesses1, fitnesses2, np.zeros_like(fitnesses1)