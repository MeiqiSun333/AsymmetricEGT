# drawing attempt

import pandas as pd
import matplotlib.pyplot as plt
import os, json
import numpy as np
from scipy.stats import spearmanr
from linearmodels import PanelOLS
import statsmodels.api as sm



def compute_avg():
    results_dir = "experiment_extre"
    filename_list = ["regular_deg_8_rew_0.json", "scale-free_deg_8_rew_0.json", "watts-strogatz_deg_8_rew_0.json"]
    for filename in filename_list:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)


        keys = ['wealth', 'U', 'V', 'eta', 'ration', 'recent_wealth', 'deg', 'cluster', 'path_length']
        averages = {key: sum(agent[key] for agent in data) / len(data) for key in keys}

        print(filename,averages)



def build_degree_from_edge_list(edge_list, num_nodes):
    degrees = [0] * num_nodes
    for (u, v) in edge_list:
        degrees[u] += 1
        degrees[v] += 1
    return degrees

def flatten_degree_wealth(num_nodes=300):
    results_dir = "experiment1"
    filename = "scale-free_degree_8.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath,'r') as f:
        data = json.load(f)

    rows = []
    for rep_i, rep_data in enumerate(data):
        for step_dict in rep_data:
            step_val = step_dict["Step"]
            edge_list = step_dict.get("Edges", None)
            wdist = step_dict.get("Wealth_distribution", None)
            if edge_list is None or wdist is None:
                continue

            deg_arr = build_degree_from_edge_list(edge_list, num_nodes)

            for agent_id in range(num_nodes):
                row = {
                    "Repetition": rep_i,
                    "Step": step_val,
                    "AgentID": agent_id,
                    "Degree": deg_arr[agent_id],
                    "Wealth": wdist[agent_id]
                }
                rows.append(row)
    df = pd.DataFrame(rows)
    return df

def spearman_correlation_degree_wealth(df):

    sub = df.dropna(subset=["Degree", "Wealth"])
    corr_all, p_all = spearmanr(sub["Degree"], sub["Wealth"])
    print(f"[Global Spearman] corr={corr_all:.4f}, p={p_all:.3g}")

    # 2) 分 Step
    print("\nStep-wise Spearman:")
    for step_val, grp in sub.groupby("Step"):
        if len(grp) > 2:  # 确保数据量
            c, p = spearmanr(grp["Degree"], grp["Wealth"])
            print(f"  Step={step_val}, n={len(grp)}, corr={c:.4f}, p={p:.3g}")

    # 3) 分 Repetition
    print("\nRepetition-wise Spearman:")
    for rep_val, grp in sub.groupby("Repetition"):
        if len(grp) > 2:
            c, p = spearmanr(grp["Degree"], grp["Wealth"])
            print(f"Repetition={rep_val}, n={len(grp)}, corr={c:.4f}, p={p:.3g}")

def panel_regression_degree_wealth(df):

    df = df.sort_values(["AgentID","Step"])
    df = df.set_index(["AgentID","Step"])

    df["Wealth_next"] = df.groupby(level=["AgentID"])["Wealth"].shift(-1)
    df["Degree_next"] = df.groupby(level=["AgentID"])["Degree"].shift(-1)
    df["Wealth_lag"]  = df["Wealth"]
    df["Degree_lag"]  = df["Degree"]

    # remove rows with NaN => the last step for each agent won't have next
    panel_df = df.dropna(subset=["Wealth_next","Degree_lag","Wealth_lag"])

    y = panel_df["Wealth_next"]  # dependent var
    X = panel_df[["Degree_lag","Wealth_lag"]]
    X = sm.add_constant(X)

    mod = PanelOLS(y, X, entity_effects=False, time_effects=False, drop_absorbed=True)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    print(res)


from experiment_stratch import run_experiment_increase_degree
def example_plot():

    # series = run_experiment_increase_degree("scale-free", num_steps=600,
    #                                        avg_degree=8,
    #                                        rewiring_prob=0.6,
    #                                        pick_count=30)
    results_dir = "experiment_add"
    filename = "scale-free_degree_8.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'r') as f:
        series = json.load(f)
    plt.plot(range(len(series)), series, marker='o')
    plt.title("Average Wealth of boosted-degree nodes vs time")
    plt.xlabel("Step")
    plt.ylabel("Average wealth (the 30 low-degree => forcibly 30-degree at step=100)")
    plt.show()


if __name__=="__main__":
    # compute_avg()

    # example_plot()


    df = flatten_degree_wealth(num_nodes=500)
    spearman_correlation_degree_wealth(df)
    panel_regression_degree_wealth(df)

