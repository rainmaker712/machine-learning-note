import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
import matplotlib.cm as cm
import seaborn as sns
sns.set_style("dark")
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import mlrose_hiive

# Models
from sklearn import tree
from sklearn.neural_network import MLPClassifier

from utils import learning_curve_plotter, model_param_curve, metrics

# base param
seed = 712
max_iters = 200
max_attempt = 10

fitness_dict = {"fourpeaks":  mlrose_hiive.FourPeaks(), 'continouspeaks': mlrose_hiive.ContinuousPeaks(), "onemax": mlrose_hiive.OneMax()}

def rhc(problem, problem_name, max_iters=max_iters, restarts=10, algo_name="rhc"):
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mlrose_hiive.random_hill_climb(problem,
                                                max_iters=max_iters, max_attempts=max_attempt,
                                                curve=False, random_state=seed, restarts=restarts)
    end_time = time.time()
    time_duration = end_time - start_time
    return [problem_name, algo_name, max_iters, time_duration, best_fitness]

def sa(problem, problem_name, max_iters=max_iters, init_temp=1, decay=0.1, min_temp=0.001, algo_name="sa"):
    start_time = time.time()
    decay = mlrose_hiive.GeomDecay(init_temp = init_temp, decay=decay, min_temp=min_temp)
    best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem,
                                                max_iters=max_iters, max_attempts=max_attempt,
                                                curve=False, random_state=seed, schedule=decay)
    end_time = time.time()
    time_duration = end_time - start_time
    return [problem_name, algo_name, max_iters, time_duration, best_fitness]

def ga(problem, problem_name, max_iters=max_iters, population=100, mutation=0.1, algo_name="ga"):
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem,
                                                max_iters=max_iters, max_attempts=max_attempt,
                                                curve=False, random_state=seed, pop_size=population,
                                                                            mutation_prob=mutation)
    end_time = time.time()
    time_duration = end_time - start_time
    return [problem_name, algo_name, max_iters, time_duration, best_fitness]


def mimic(problem, problem_name, max_iters=max_iters, keep_pct=0.2, algo_name="mimic"):
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mlrose_hiive.mimic(problem,
                                                max_iters=max_iters, max_attempts=max_attempt,
                                                curve=False, random_state=seed, keep_pct=keep_pct)
    end_time = time.time()
    time_duration = end_time - start_time
    return [problem_name, algo_name, max_iters, time_duration, best_fitness]

iteration = range(max_iters+1)
problem_list = ["fourpeaks", "continouspeaks", "onemax"]
problem_name = problem_list[2]

results = []

problem = mlrose_hiive.DiscreteOpt(fitness_fn=fitness_dict[problem_name], maximize=True, length=100, max_val = 2)
for step in range(1,max_iters+2, 20):
    rhc_result = rhc(problem, problem_name, max_iters=step)
    print("rhc_finished")
    sa_result = sa(problem, problem_name, max_iters=step)
    print("sa_finished")
    ga_result = ga(problem, problem_name, max_iters=step)
    print("ga_finished")
    mimic_result = mimic(problem, problem_name, max_iters=step)
    print("mimic_finished")
    results.append(rhc_result)
    results.append(sa_result)
    results.append(ga_result)
    results.append(mimic_result)

df = pd.DataFrame(results, columns=["problem_name", "algo", "iteration", "time_duration", "best_fitness"])

algo_list = ["rhc", "sa", "ga", "mimic"]
df["time_duration_old"] = df['time_duration']
df["time_duration"] = np.log2(df['time_duration'])

def plotting_algo(df):

    rhc = df[df.algo == algo_list[0]]
    iters = rhc["iteration"]
    
    rhc_fit = rhc["best_fitness"]
    rhc_time = rhc["time_duration"]
    
    sa = df[df.algo == algo_list[1]]
    sa_fit = sa["best_fitness"]
    sa_time = sa["time_duration"]
    
    ga = df[df.algo == algo_list[2]]
    ga_fit = ga["best_fitness"]
    ga_time = ga["time_duration"]
    
    mimic = df[df.algo == algo_list[3]]
    mimic_fit = mimic["best_fitness"]
    mimic_time = mimic["time_duration"]    
    
    plt.plot(iters, rhc_fit, label=algo_list[0], color="b")
    plt.plot(iters, sa_fit, label=algo_list[1], color="g")
    plt.plot(iters, ga_fit, label=algo_list[2], color="r")
    plt.plot(iters, mimic_fit, label=algo_list[3], color="y")
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("fitness value")
    plt.savefig(f"{problem_name}_fitness.png")
    plt.close()

    plt.plot(iters, rhc_time, label=algo_list[0], color="b")
    plt.plot(iters, sa_time, label=algo_list[1], color="g")
    plt.plot(iters, ga_time, label=algo_list[2], color="r")
    plt.plot(iters, mimic_time, label=algo_list[3], color="y")
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Time-log_scale (ms)")
    plt.savefig(f"{problem_name}_duration.png")    
    plt.close()

plotting_algo(df)
