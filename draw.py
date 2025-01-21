import os
import json
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def read_json_results(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def average_gini_per_step(data):
    steps = [item['Step'] for item in data[0]]
    step2ginis = {step: [] for step in steps}

    for rep_i in range(len(data)):
        for item in data[rep_i]:
            st = item['Step']
            gi = item['Gini']
            step2ginis[st].append(gi)

    step2avg = {}
    for st, ginis in step2ginis.items():
        step2avg[st] = np.mean(ginis)
    return step2avg

# def get_gini_dicts_for_all_reps(data):
#
#     steps = [item['Step'] for item in data[0]]
#     steps_list = sorted(steps)
#
#     rep2dict = []
#     for rep_i in range(len(data)):
#         step_gini_map = {}
#         for item in data[rep_i]:
#             st = item['Step']
#             gi = item['Gini']
#             step_gini_map[st] = gi
#         rep2dict.append(step_gini_map)
#     return rep2dict, steps_list
#
# def compute_diff_stats(rep2dict_A, rep2dict_B, steps_list):
#     n_rep = len(rep2dict_A)
#     result = {}
#     for st in steps_list:
#         diffs = []
#         for i in range(n_rep):
#             gA = rep2dict_A[i].get(st, 0)
#             gB = rep2dict_B[i].get(st, 0)
#             diffs.append(gA - gB)
#         diffs = np.array(diffs)
#         mean_diff = diffs.mean()
#         std_diff = diffs.std(ddof=1)
#         # 95% CI
#         ci = 1.96 * std_diff / sqrt(n_rep)
#         lower = mean_diff - ci
#         upper = mean_diff + ci
#         result[st] = (mean_diff, lower, upper)
#     return result


def main():
    avg_degrees = [6, 8, 10]
    results_dir = "experiment1"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, avg_degree in enumerate(avg_degrees):
        ax = axes[idx]

        file_reg = os.path.join(results_dir, f"regular_avg_degree_{avg_degree}.json")
        file_ws  = os.path.join(results_dir, f"watts-strogatz_avg_degree_{avg_degree}.json")
        file_sf  = os.path.join(results_dir, f"scale-free_avg_degree_{avg_degree}.json")

        data_reg = read_json_results(file_reg)
        data_ws  = read_json_results(file_ws)
        data_sf  = read_json_results(file_sf)

        # reg_rep2dict, steps_list = get_gini_dicts_for_all_reps(data_reg)
        # ws_rep2dict, _ = get_gini_dicts_for_all_reps(data_ws)
        # sf_rep2dict, _ = get_gini_dicts_for_all_reps(data_sf)
        # ws_diff_stats = compute_diff_stats(ws_rep2dict, reg_rep2dict, steps_list)
        # sf_diff_stats = compute_diff_stats(sf_rep2dict, reg_rep2dict, steps_list)
        # ws_mean = []
        # ws_lower = []
        # ws_upper = []
        # sf_mean = []
        # sf_lower = []
        # sf_upper = []
        # for st in steps_list:
        #     m1, l1, u1 = ws_diff_stats[st]
        #     ws_mean.append(m1)
        #     ws_lower.append(l1)
        #     ws_upper.append(u1)
        #     m2, l2, u2 = sf_diff_stats[st]
        #     sf_mean.append(m2)
        #     sf_lower.append(l2)
        #     sf_upper.append(u2)
        # ax.plot(steps_list, ws_mean, label='WS - Regular', color='tab:blue')
        # ax.fill_between(steps_list, ws_lower, ws_upper, color='tab:blue', alpha=0.2)
        # ax.plot(steps_list, sf_mean, label='SF - Regular', color='tab:orange')
        # ax.fill_between(steps_list, sf_lower, sf_upper, color='tab:orange', alpha=0.2)

        # without CI
        reg_dict = average_gini_per_step(data_reg)
        ws_dict = average_gini_per_step(data_ws)
        sf_dict = average_gini_per_step(data_sf)
        steps_list = sorted(reg_dict.keys())
        ws_minus_rr = [ws_dict[st] - reg_dict[st] for st in steps_list]
        sf_minus_rr = [sf_dict[st] - reg_dict[st] for st in steps_list]
        ax.plot(steps_list, ws_minus_rr, label='WS - Regular')
        ax.plot(steps_list, sf_minus_rr, label='SF - Regular')

        ax.set_title(f"Avg Degree = {avg_degree}")
        ax.set_xlabel("Step")
        if idx == 0:
            ax.set_ylabel("Gini difference")
        ax.legend()

    plt.tight_layout()
    plt.savefig("experiment1/compare_ws_sf_regular.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()


